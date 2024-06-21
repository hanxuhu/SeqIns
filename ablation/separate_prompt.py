import datasets
import pandas as pd
import argparse
import random
import time
from functools import partial
from tqdm import tqdm
from transformers import AutoTokenizer
import vllm
import torch

PROMPT_TEMPLATE = """Given the instruction, can you split the instruction into two or more? (usually they are connected with words "and then", "Then, ", "then" etc.) and pair the split instruction given the response. Give your answer in the format "Instruction N: [INSTRUCTION]\\nResponse N: [RESPONSE]" Report the whole split instruction and its corresponding response, don't return something like "the rest of response".
Instruction: "{instruction}"
Response: "{response}"
"""

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

def get_cohere(prompt, client, model="command-r", max_tokens=3999, max_attempt=5):
    cur_attempt = 0
    while cur_attempt < max_attempt:
        cur_attempt += 1
        try:
            response = client.chat(
                message=prompt,
                model=model,
                max_tokens=max_tokens,
            )
            response_content = response.text
            if response_content:
                return response_content
            else:
                print(response, flush=True)
                print("Response text is empty/none or not following the template. Retrying ...", flush=True)
                time.sleep(random.uniform(0.25, 1))

        except Exception as e:
            print(e, flush=True)
            print(" Retrying ...", flush=True)
            time.sleep(random.uniform(0.25, 1))

    print(f"Failed {max_attempt} times. Returning a placeholder.", flush=True)
    return f"THIS EVALUATION FAILED AFTER {max_attempt} ATTEMPTS."

def separate_prompt(instruction, input, response):
    if input != "":
        instruction = instruction + " Input: " + input
    return PROMPT_TEMPLATE.format(instruction=instruction, response=response)

def parse_response(response):
    parsed = []
    for i in range(1, 10):
        # search for the instruction i
        instruction = response.find(f"Instruction {i}:")
        if instruction == -1:
            instruction = response.find(f"Split instruction {i}:")
        output = response.find(f"Response {i}:")
        if instruction == -1 or output == -1:
            break
        instruction = response[instruction+len(f"Instruction {i}:"):output].strip()
        next_instruction = response.find(f"Instruction {i+1}:")
        if next_instruction == -1:
            output = response[output+len(f"Response {i}:"):].strip()
        else:
            output = response[output+len(f"Response {i}:"):next_instruction].strip()
        parsed.append((instruction, output))
    if parsed:
        return parsed
    # search for Instruction N and Response N
    instruction_idx = response.find("Instruction N:")
    output_idx = response.find("Response N:")
    loop = 0
    while (instruction_idx != -1) and (output_idx != -1):
        # if loop > 10:
        #     break
        loop += 1
        instruction = response[instruction_idx+len("Instruction N:"):output_idx].strip()
        next_instruction = response.find("Instruction N:", output_idx)
        if next_instruction == -1:
            output = response[output_idx+len("Response N:"):].strip()
        else:
            output = response[output_idx+len("Response N:"):next_instruction].strip()
        parsed.append((instruction, output))
        response = response[output_idx+len("Response N:"):]
        instruction_idx = response.find("Instruction N:")
        output_idx = response.find("Response N:")
    
    return parsed

def main(args):
    input = args.input
    df = pd.read_json(input, lines=True)
    df['prompt'] = df.apply(lambda x: separate_prompt(x['instruction'], x['input'], x['output']), axis=1)

    generation_kwargs = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "max_tokens": 2048,
    }
    vllm_kwargs = {
        "tokenizer_mode": "auto",
        "tensor_parallel_size": torch.cuda.device_count(),
        "gpu_memory_utilization": 0.95,
    }
    # if args.model == "command-r-plus":
    #     import cohere
    #     client = cohere.Client(COHERE_KEY)
    #     judge = partial(get_cohere, client=client, model="command-r-plus", max_tokens=3999)
    # else:
    client = VllmAgent(args.model, vllm_kwargs, generation_kwargs)

    prompts = df['prompt'].tolist()
    responses = client.generate(prompts)
    df['response'] = responses
    # save the responses
    df.to_json(input.replace(".jsonl", "-separated.jsonl"), lines=True, orient='records')
    df['parsed_response'] = df['response'].apply(parse_response)
    df.to_json(input.replace(".jsonl", "-parsed.jsonl"), lines=True, orient='records')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='alpaca_llama70b_iteration_2-iter-filtered.jsonl')
    parser.add_argument('--model', type=str, default="/mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct")

    args = parser.parse_args()
    main(args)