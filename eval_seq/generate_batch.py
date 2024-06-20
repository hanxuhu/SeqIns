import os
import sys
import json
import warnings
import fire
import torch
import transformers
from tqdm import tqdm, trange
from peft import PeftModel
from transformers import GenerationConfig
from transformers import AutoModelForCausalLM #, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, LlamaTokenizer
# from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
from utils.generate import *
import vllm

lora_base_map = {"baichuan-13b":"baichuan-inc/Baichuan-13B-Base",
                 "baichuan-7b":"baichuan-inc/baichuan-7B",
                 "baichuan-2-7b":"baichuan-inc/Baichuan2-7B-Base",
                 "bloom-560m":"bigscience/bloom-560m",
                 "bloom-1b1":"bigscience/bloom-1b1",
                 "bloom-1b7":"bigscience/bloom-1b7",
                 "bloom-3b":"bigscience/bloom-3b",
                 "bloom-7b1":"bigscience/bloom-7b1",
                 "mistral-7b": "mistralai/Mistral-7B-v0.1",
                 "llama-7b": "huggyllama/llama-7b",#"baffo32/decapoda-research-llama-7B-hf", #"decapoda-research/llama-7b-hf",
                 "llama-13b":"../patrick/llama-13b-base",#"decapoda-research/llama-13b-hf",
                 "llama-30b":"decapoda-research/llama-30b-hf",
                 "ollama-3b":"openlm-research/open_llama_3b",
                 "ollama-7b":"openlm-research/open_llama_7b",
                 "ollama-13b":"openlm-research/open_llama_13b",
                 "phi-2-2.7b":"microsoft/phi-2",
                 "pythia-70m":"EleutherAI/pythia-70m-deduped",
                 "pythia-160m":"EleutherAI/pythia-160m-deduped",
                 "pythia-410m":"EleutherAI/pythia-410m-deduped",
                 "pythia-1b":"EleutherAI/pythia-1b-deduped",
                 "pythia-1b4":"EleutherAI/pythia-1.4b-deduped",
                 "pythia-2b8":"EleutherAI/pythia-2.8b-deduped",
                 "pythia-6b9":"EleutherAI/pythia-6.9b-deduped",
                 "pythia-12b":"EleutherAI/pythia-12b-deduped"}

def main(
    length: int = 100,
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "",
    batch_size: int = 4,
    test_file: str = "",
    save_file: str = "",
    samples: int = None,
    prompt_template: str = "alpaca",  # The prompt template to use, will default to alpaca.
    use_vllm: bool = False,
    is_chat: bool = False,
    chat_template: str = "alpaca",
):
    print(1)
    print(test_file)
    # if lora_weights == "":
    #     assert base_model
    #     print("\n\n******WARNING: LoRA module is not specified. Loading only the base model for inference.******\n\n", flush=True)
    # if lora_weights != "" and lora_weights[-1] == "/":
    #     lora_weights = lora_weights[:-1]
    #     print(lora_weights)
    # if not base_model:
    #     print("no base model")
    #     for suffix in lora_base_map:
    #         if lora_weights.endswith(suffix):
    #             base_model = lora_base_map[suffix]
    #             continue
    #     print(base_model)
    #     assert base_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    if test_file:
        test_lang = test_file.split(".json")[0].split("_")[-1]
    if not save_file:
        save_file = "data/test-" + test_lang + "_decoded_by_" + lora_weights.split("/")[-1] + ".jsonl"

    prompter = Prompter(prompt_template)

    if use_vllm:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = vllm.LLM(
            model=base_model,
            tokenizer=base_model,
            tokenizer_mode="slow",
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.97
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, token='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', trust_remote_code=True)

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
            trust_remote_code=True,
            device_map="auto",
            token='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',
        )
    if chat_template == "tulu":
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
        
    if device == "cuda":
        print("Using " + str(torch.cuda.device_count())  + " GPU devices", flush=True)

    def read_data(filename):
        data = []
        # with open(filename) as f:
        #     data = json.load(f)
        # '''
        # with open(filename) as f:
        #     for line in f:
        #         line = json.loads(line.strip())
        #         #print(line)
        #         data.append({"instruction": line["instruction"], "input": line["input"]})
        # '''
        with open(filename) as f:
            if filename.endswith(".json"):
                data = json.load(f)
            elif filename.endswith(".jsonl"):
                for line in f:
                    line = json.loads(line.strip())
                    data.append(line)
                    # data.append({
                    #     "instruction": line["instruction"], 
                    #     "input": line["input"] if "input" in line else None,
                    #     "target": line["target"] if "target" in line else None,
                    #     })
            else:
                raise ValueError("File format not supported. Please provide a .json or .jsonl file.")
        return data


    def evaluate(
        instruction,
        input=None,
        label=None,
        temperature=0,
        top_p=1,
        top_k=50,
        num_beams=1, # perhaps can experiment with this
        max_new_tokens=256,
        no_repeat_ngram_size=6,
        **kwargs,
    ):
        prompts = []
        for i in range(len(instruction)):
            generate_prompt_func = prompter.generate_chat_prompt if is_chat else prompter.generate_prompt
            if input:
                prompt = generate_prompt_func(instruction[i], input[i])
            else:
                prompt = generate_prompt_func(instruction[i], '')
            if is_chat:
                prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)

            prompts.append(prompt)
        
        if use_vllm:
            sampling_params = vllm.SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_new_tokens,
                # stop=["\n\n"],
            )
            generations = model.generate(prompts, sampling_params)
            prompt_to_output = {
                g.prompt: g.outputs[0].text for g in generations
            }
            outputs = [prompt_to_output[p] if prompt in prompt_to_output else "" for p in prompts]

        else:
            # new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1]
            # generation_config = GenerationConfig(
            #     temperature=temperature,
            #     top_p=top_p,
            #     top_k=top_k,
            #     num_beams=num_beams,
            #     max_new_tokens=max_new_tokens,
            #     no_repeat_ngram_size=no_repeat_ngram_size,
            #     # stop_id_sequences=[[new_line_token]]
            #     **kwargs,
            # )
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_beams": num_beams,
                "max_new_tokens": max_new_tokens,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                **kwargs,
            }
            
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                batch_size=batch_size,
                **generation_config,
            )
            
        return [
            {'output': prompter.get_response(out),
             'prompt': prompt}
               for out, prompt in zip(outputs, prompts)
        ]

    if test_file:
        # test_lang = test_file.split(".jsonl")[0].split("_")[-1]
        data = read_data(test_file)
        write_data = []

        if samples is not None:
            data = data[:samples]

        instructions = [item["instruction"] for item in data]
        inputs = [item["input"] for item in data]
        labels = None
        if "translated_input" in data[0]:
            labels = [item["translated_input"] for item in data]
        responses = evaluate(instructions, 
                             input=inputs, 
                             label=labels,
                             max_new_tokens=length)
        for i in range(len(data)):
            data[i]['target'] = data[i]['output']
            data[i]["output"] = responses[i]['output']
            data[i]["prompt"] = responses[i]['prompt']

        with open(save_file, "w", encoding='utf-8') as out_f:
            for p in data:
                out_f.write(json.dumps(p, ensure_ascii=False) + "\n")
    else:
        print("No test file provided, will test on a few pre-defined example questions.", flush=True)
        for instruction in [
            "What are the even numbers between 1 and 13?",
            "Please briefly introduce the animal alpaca.",
            "What is the meaning of life?",
        ]:
            print("Instruction:", instruction.strip())
            print("Response:", evaluate(instruction).split("\n### ")[0].strip())
            print()

        for instruction, input in zip(["Please write a sentence using the given word.",
                                       "Can you repeat the following phrase 3 times?"
                                      ],
                                      ["vicuna",
                                       "Scotland!"
                                      ]):
            print("Instruction:", instruction.strip())
            print("Input:", input.strip())
            print("Response:", evaluate(instruction, input).split("\n### ")[0].strip())
            print()


if __name__ == "__main__":
    fire.Fire(main)
