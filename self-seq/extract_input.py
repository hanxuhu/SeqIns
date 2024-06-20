import json
import os.path
import random

from tqdm import tqdm, trange
import argparse
from token_store import API_KEYs
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import vllm
import cohere
import time
from agent import HfAgent, VllmAgent, GptAgent, CohereAgent
from extract_input_template import *
import refine_template

def get_prompt(p, is_chat=False):
    few_shot_example = FEW_SHOTS_EXAMPLE
    prompt_prefix = PROMPT_PREFIX

    e = few_shot_example.copy()
    random.shuffle(e) # shuffle the few shot examples to prevent position bias
    # prompt = '\n\n'.join([prompt_prefix + '\n' + example for example in e])
    messages = []
    # content = prompt_prefix
    for i, few_shot in enumerate(e):
        if i == 0:
            messages += [{'role': 'user', 'content': prompt_prefix + '\n\n' + PROMPT_TEMPLATE.format(few_shot[0])}, {'role': 'assistant', 'content': few_shot[1]}]
        else:
            messages += [{'role': 'user', 'content': PROMPT_TEMPLATE.format(few_shot[0])}, {'role': 'assistant', 'content': few_shot[1]}]
    # prompt = prompt_prefix + '\n\n' + '\n\n'.join(e)

    if 'conversations' in p: # cases for lima
        instruction, output = p['conversations'][0], p['conversations'][1]
        # prompt += '\n\n' + prompt_prefix + '\n' + PROMPT_TEMPLATE.format(instruction)
        prompt = '\n\n' + PROMPT_TEMPLATE.format(instruction)
        input = ''
    elif 'question' in p: # cases for flancot
        instruction = p['question']
        prompt = '\n\n' + PROMPT_TEMPLATE.format(instruction)
        input = ''
    else: # cases for alpaca like data (with input)
        instruction = p['instruction']
        prompt = '\n\n' + PROMPT_TEMPLATE.format(instruction)
        input = ''

    messages += [{'role': 'user', 'content': prompt}]
    return {**p, 'messages': messages}

def extract_input(agent, generation_kwargs, prompts, batch_size=1):
    if isinstance(agent, GptAgent):
        batch_size = 1
    elif isinstance(agent, VllmAgent):
        batch_size = len(prompts)
    generations = []
    for i in trange(0, len(prompts), batch_size):
        if isinstance(agent, GptAgent):
            batch_prompts = prompts[i]
        else:
            batch_prompts = prompts[i:i + batch_size]
            batch_prompts = [tokenizer.apply_chat_template(p['messages'], add_generation_prompt=True, tokenize=False) for p in batch_prompts]
        outputs = agent.generate(batch_prompts, **generation_kwargs)
        if isinstance(agent, GptAgent):
            outputs = [outputs]
            batch_prompts = [batch_prompts]

        for idx, (prompt, output) in enumerate(zip(batch_prompts, outputs)):
            instruction, input = extract_instruction_input(prompt, output)
            generations.append({
                'idx': i + idx,
                "ori_instruction": prompts[i + idx]['instruction'],
                'instruction': instruction,
                'input': input,
                'system_prompt': prompts[i + idx]['system_prompt'],
                'output': prompts[i + idx]['output'],
            })
    return generations
    
def extract_instruction_input(prompt, output):
    output = output.strip()
    instruction = ''
    input = ''
    if 'instruction:' in output.lower(): # parse from instruction: to input:
        instruction = output[output.lower().index('instruction:') + len('instruction:'):].strip()
        # get the minimum of the index of 'input:' and '###'
        if 'input:' in instruction.lower():
            instruction = instruction[:instruction.lower().index('input:')].strip("# \n")
        elif '###' in instruction:
            instruction = instruction[:instruction.index('###')].strip()
        if "explanation:" in instruction.lower():
            instruction = instruction[:instruction.lower().index('explanation:')].strip("\n ")
    if 'input:' in output.lower():
        input = output[output.lower().index('input:') + len('input:'):].strip()
        if '###' in input:
            input = input[:input.index('###')].strip()
    return instruction, input

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--input_file', type=str, default='data/alpaca.jsonl')
    args.add_argument('--output_file', type=str, default=None)
    args.add_argument('--sample', type=int, default=-1)
    args.add_argument('--query', type=str, default='gpt-3.5-turbo')
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--batch_size', type=int, default=1)
    args.add_argument('--load_8bit', action='store_true')
    args.add_argument('--load_4bit', action='store_true')
    args.add_argument('--use_vllm', action='store_true')
    args.add_argument('--use_instruct', action='store_true')

    args = args.parse_args()
    assert not (args.load_8bit and args.load_4bit)
    input_file = args.input_file
    if args.output_file is None:
        output_file = input_file.replace('.jsonl', f'-{os.path.basename(args.query)}.jsonl')
    else:
        output_file = args.output_file

    input_data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            input_data.append(json.loads(line))

    if args.sample > 0:
        random.seed(args.seed)
        input_data = random.sample(input_data, args.sample)

    prompts = [get_prompt(p, is_chat=args.use_instruct) for p in input_data]

    if 'gpt' in args.query:
        agent = GptAgent(api_key=random.choice(API_KEYs), model_name=args.query)

        generations = extract_input(agent, generation_kwargs={}, prompts=prompts, batch_size=args.batch_size)
    elif args.query in ['command-r']:
        Agent = CohereAgent(api_key=random.choice(API_KEYs))
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.query)
        if args.use_instruct:
            tokenize_prompts = [tokenizer.apply_chat_template(p['messages'], add_generation_prompt=True, tokenize=False) for p in prompts]
            stop = None
            stop_id_sequences = None
        else:
            tokenize_prompts = [p['prompt'] for p in prompts]
            stop = ["###", "###\n", "###\n\n"]
            stop_id_sequences = [tokenizer.encode(s, add_special_tokens=False) for s in stop]

        generation_kwargs = {
            "temperature": 0,
            "top_p": 1,
            "top_k": 50,
            "max_new_tokens": 1024,
        }
        if args.use_vllm:
            vllm_kwargs = {
            "tokenizer_mode": "auto",
            "tensor_parallel_size": torch.cuda.device_count(),
            "gpu_memory_utilization": 0.95,
        }
            generation_kwargs["stop"] = stop
            # replace max_new_tokens with max_tokens
            generation_kwargs["max_tokens"] = generation_kwargs.pop("max_new_tokens")
            agent = VllmAgent(args.query, vllm_kwargs, generation_kwargs)
            generation_kwargs = {}
        else:
            model_kwargs = {
            "load_in_8bit": args.load_8bit,
            "load_in_4bit": args.load_4bit,
            "torch_dtype": torch.bfloat16 if 'gptq' not in args.query.lower() else torch.float16,
            "attn_implementation": 'flash_attention_2',
            "device_map": "auto",
            }
            agent = HfAgent(args.query, model_kwargs, generation_kwargs)
            generation_kwargs = {'stop_id_sequences': stop_id_sequences}

        generations = extract_input(agent, generation_kwargs, prompts=prompts, batch_size=args.batch_size)
        output_file = output_file.replace('.jsonl', '-extracted-input.jsonl')
        with open(output_file, 'w', encoding='utf-8') as file:
            for g in generations:
                file.write(json.dumps(g) + '\n')