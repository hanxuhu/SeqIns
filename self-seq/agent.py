import json
import os.path
import random

from openai import OpenAI
from tqdm import tqdm, trange
from template import *
import argparse
# from token_store import API_KEYs
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import vllm
import cohere
import time
from transformers import StoppingCriteria

# Support Hf, vLLM, GPT and cohere
class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)
    
class HfAgent:
    def __init__(self, model_name, model_kwargs, generation_kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        if 'gptq' in model_name.lower():
            from auto_gptq import exllama_set_max_input_length
            self.model = exllama_set_max_input_length(self.model, 8092)
        self.generation_kwargs = generation_kwargs

    @torch.no_grad()
    def generate(self, prompt, stop_id_sequences=None):
        tokenized_prompts = self.tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True)
        batch_input_ids = tokenized_prompts['input_ids']
        tokenized_prompts = tokenized_prompts.to(self.model.device)
        with torch.cuda.amp.autocast():
            batch_outputs = self.model.generate(**tokenized_prompts, **self.generation_kwargs, stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences is not None else None)

        if stop_id_sequences:
            for output_idx in range(batch_outputs.shape[0]):
                for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                    if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                        batch_outputs[output_idx, token_idx:] = self.tokenizer.pad_token_id
                        break

        batch_outputs = self.tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        batch_prompts = self.tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
        batch_generations = [
            output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
        ]
        return batch_generations
    
class VllmAgent:
    def __init__(self, model_name, model_kwargs, generation_kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = vllm.LLM(
            model_name,
            tokenizer=model_name,
            **model_kwargs
        )
        self.sampling_params = vllm.SamplingParams(
            **generation_kwargs
        )

    def generate(self, prompt, stop=None):
        generations = self.model.generate(prompt, self.sampling_params)
        prompt_to_output = {
                g.prompt: g.outputs[0].text for g in generations
            }
        outputs = [prompt_to_output[p] if p in prompt_to_output else "" for p in prompt]
        return outputs

class GptAgent:
    def __init__(self, api_key, model_name):
        self.client = OpenAI(
            api_key=api_key,
            max_retries=3,
        )
        self.model_name = model_name

    def generate(self, prompt):
        print('Querying GPT-3.5-turbo...')

        chat_completion = self.client.chat.completions.create(
            messages=prompt['messages'],
            model=self.model_name,
            temperature=0,
            max_tokens=4096,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        result = chat_completion.choices[0].message.content
        return result.strip()
    
class CohereAgent:
    def __init__(self, api_key, max_attempt=10):
        self.co = cohere.Client(api_key)
        self.max_attempt = max_attempt

    def generate(self, prompt):
        systems = [p for p in prompt['messages'] if p['role'] == "system"]
        if len(systems) == 0:
            system_message = ""
        else:
            system_message = systems[-1]['content']
        message = prompt['messages'][-1]['content']
        
        if message == "":
            return ""
        cur_attempt = 0
        while cur_attempt < self.max_attempt:
            cur_attempt += 1
            try:
                if system_message == "":
                    response = self.co.chat(
                        model="command-r-plus",
                        message=message,
                    )
                else:
                    response = self.co.chat(
                        model="command-r-plus",
                        preamble=system_message,
                        message=message,
                    )
                text = response.text.strip()
                if text:
                    return text
                else:
                    print(response, flush=True)
                    print("Translated text is empty or None. Retrying ...", flush=True)
                    time.sleep(random.uniform(2, 4))
            except Exception as e:
                print(e, flush=True)
                print(" Retrying ...", flush=True)
                time.sleep(random.uniform(2, 4))

        print(f"Failed {self.max_attempt} times. Returning a placeholder.", flush=True)
        return f"THIS TRANSLATION FAILED AFTER {self.max_attempt} ATTEMPTS."