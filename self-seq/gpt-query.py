import json
import os.path
import random

from openai import OpenAI
from tqdm import tqdm, trange
from template import *
from instruct_template import *
import argparse
from token_store import API_KEYs
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import vllm
import cohere
import time
from agent import HfAgent, VllmAgent, GptAgent, CohereAgent
import refine_template

def get_prompt(p, is_chat=False):
    few_shot_example = FEW_SHOTS_EXAMPLE if not is_chat else FEW_SHOTS_EXAMPLE_CHAT
    prompt_prefix = PROMPT_PREFIX if not is_chat else PROMPT_PREFIX_CHAT

    e = few_shot_example.copy()
    random.shuffle(e) # shuffle the few shot examples to prevent position bias
    prompt = prompt_prefix + '\n\n' + '\n\n'.join(e)

    if 'conversations' in p: # cases for lima
        instruction, output = p['conversations'][0], p['conversations'][1]
        prompt += '\n\n' + PROMPT_TEMPLATE.format(instruction, '')
        input = ''
    elif 'question' in p: # cases for flancot
        instruction = p['question']
        prompt += '\n\n' + PROMPT_TEMPLATE.format(instruction, '')
        input = ''
    else: # cases for alpaca like data (with input)
        instruction = p['instruction']
        input = ''
        if p['input'] != '':
            # input = INPUT_TEMPLATE.format(p['input'])
            input = p['input']
            if ('position' in p) and (p['position'] == 'right'):
                instruction = f"{instruction} {input}"
            else:
                instruction = f"{input} {instruction}"
        prompt += '\n\n' + PROMPT_TEMPLATE.format(instruction, '')

    if 'system_prompt' in p:
        system_prompt = p['system_prompt']
    else:
        system_prompt = ''

    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]
    return {'prompt': prompt, 'instruction': instruction, 'input': input, 'messages': messages, 'system_prompt': system_prompt}

def get_gen_instruction_prompt(p):
    if (p['option'] is None) or (p['option'] == 'D'):
        return {
            **p,
            'new_instruction': p['instruction'],
        }
    
    if p['option'] == 'A':
        prompt_prefix = PROMPT_PREFIX_A
        few_shot_examples = FEW_SHOTS_EXAMPLE_A
        prompt_template = PROMPT_TEMPLATE_A
    elif p['option'] == 'B':
        prompt_prefix = PROMPT_PREFIX_B
        few_shot_examples = FEW_SHOTS_EXAMPLE_B
        prompt_template = PROMPT_TEMPLATE_B
    elif p['option'] == 'C':
        prompt_prefix = PROMPT_PREFIX_C
        few_shot_examples = FEW_SHOTS_EXAMPLE_C
        prompt_template = PROMPT_TEMPLATE_C

    e = few_shot_examples.copy()
    random.shuffle(e)
    prompt = prompt_prefix + '\n\n' + '\n\n'.join(e)
    if p['input'] != '':
        if ('position' in p) and (p['position'] == 'right'):
            instruction = f"{p['input']} {p['instruction']}"
        else:
            instruction = f"{p['instruction']} {p['input']}"

    prompt += '\n\n' + prompt_template.format(instruction, '')
    # prompt += '\n\n' + prompt_template.format(p['instruction'])
    messages = [{'role': 'user', 'content': prompt}]
    return {**p, 'messages': messages}

def get_refine_prompt(p, is_chat=False):
    if p['extracted_instruction'] is None:
        return p

    original_instruction = p['instruction']
    new_instruction = p['extracted_instruction']

    prompt_prefix = refine_template.PROMPT_PREFIX
    few_shot_example = refine_template.FEW_SHOTS_EXAMPLE
    prompt_template = refine_template.PROMPT_TEMPLATE

    e = few_shot_example.copy()
    random.shuffle(e)
    prompt = prompt_prefix + '\n\n' + '\n\n'.join(e)
    prompt += '\n\n' + prompt_template.format(p['instruction'], p['extracted_instruction'])
    messages = [{'role': 'user', 'content': prompt}]
    return {**p, 'messages': messages}

def extract_classification(o):
    # check if 'option: a', 'option: b', 'option: c', 'option: d' in the completion
    # if so, extract the option and the explanation
    # if not, return None
    if ('option a' in o.lower()) or ('option: a' in o.lower()):
        return 'A'
    elif ('option b' in o.lower()) or ('option: b' in o.lower()):
        return 'B'
    elif ('option c' in o.lower()) or ('option: c' in o.lower()):
        return 'C'
    elif ('option d' in o.lower()) or ('option: d' in o.lower()):
        return 'D'
    else:
        if 'A.' in o:
            return 'A'
        elif 'B.' in o:
            return 'B'
        elif 'C.' in o:
            return 'C'
        elif 'D.' in o:
            return 'D'
        else:
            if 'prefix task' in o.lower():
                return 'B'
            elif 'suffix task' in o.lower():
                return 'C'
            elif 'decompose' in o.lower():
                return 'A'
            else:
                return 'D'

def extract_instruction(o):
    # check if the completion contains 'new instruction' or 'new task'
    # if so, extract the new instruction
    # if not, return None
    if '#new instruction#' in o.lower():
        # get the last appearance of the new instruction
        instruction = o[o.lower().rindex('#new instruction#') + len('#new instruction#'):]
    elif 'new instruction' in o.lower():
        instruction = o[o.lower().rindex('new instruction') + len('new instruction'):]
    else:
        return None
    
    if "###" in instruction:
        instruction = instruction[:instruction.index("###")]
    return instruction.strip(":# \n")
    
def extracted_refined_instruction(o):
    if "no" in " ".join(o.split()[:10]).lower():
        if "#new instruction#" in o.lower():
            o = o[o.lower().index('#new instruction#') + len('#new instruction#'):]
        elif "new instruction" in o.lower():
            o = o[o.lower().index('new instruction') + len('new instruction'):]

        if "###" in o.lower():
            o = o[:o.lower().index("###")]
        return o.strip(":# \n")
    else:
        return None
        
def classification(agent, generation_kwargs, prompts, batch_size=1):
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
            generations.append({
                'idx': i + idx if 'idx' not in prompts[i + idx] else prompts[i + idx]['idx'],
                "input": prompts[i + idx]['input'],
                'system_prompt': prompts[i + idx]['system_prompt'] if 'system_prompt' in prompts[i + idx] else None,
                "instruction": prompts[i + idx]['instruction'],
                'completions': output,
                'option': extract_classification(output)
            })
            if 'final_instruction' in prompts[i + idx]:
                generations[-1]['final_instruction'] = prompts[i + idx]['final_instruction']
            if 'final_instruction_response' in prompts[i + idx]:
                generations[-1]['final_instruction_response'] = prompts[i + idx]['final_instruction_response']
    return generations

def generation(agent, generation_kwargs, prompts, batch_size=1):
    if isinstance(agent, GptAgent):
        batch_size = 1
    elif isinstance(agent, VllmAgent):
        batch_size = len(prompts)
    get_gen_instruction_prompts = [get_gen_instruction_prompt(p) for p in prompts]
    gen_instruction = [p for p in get_gen_instruction_prompts if 'messages' in p]
    new_generations = [p for p in get_gen_instruction_prompts if 'messages' not in p]

    for i in trange(0, len(gen_instruction), batch_size):
        if isinstance(agent, GptAgent):
            batch_prompts = gen_instruction[i]
        else:
            batch_prompts = gen_instruction[i:i + batch_size]
            batch_prompts = [tokenizer.apply_chat_template(p['messages'], add_generation_prompt=True, tokenize=False) for p in batch_prompts]
        outputs = agent.generate(batch_prompts, **generation_kwargs)

        if isinstance(agent, GptAgent):
            outputs = [outputs]
            batch_prompts = [batch_prompts]

        for idx, (prompt, output) in enumerate(zip(batch_prompts, outputs)):
            # remove messages from the prompt
            prompt = gen_instruction[i + idx]
            prompt.pop('messages')
            new_generations.append({
                **prompt,
                'new_instruction': output,
                'extracted_instruction': extract_instruction(output)
            })
    return new_generations

def refinement(agent, prompts, batch_size=1, generation_kwargs={}):
    if isinstance(agent, GptAgent):
        batch_size = 1
    elif isinstance(agent, VllmAgent):
        batch_size = len(prompts)
    refining_generations = [p for p in prompts if ('extracted_instruction' in p) and (p['extracted_instruction'] is not None)]
    refined_generations = [p for p in prompts if ('extracted_instruction' not in p) or (p['extracted_instruction'] is None)]
    
    refineing_prompts = [get_refine_prompt(p) for p in refining_generations]
    for i in trange(0, len(refineing_prompts), batch_size):
        if isinstance(agent, GptAgent):
            batch_prompts = refineing_prompts[i]
        else:
            batch_prompts = [p for p in refineing_prompts[i:i + batch_size]]
            batch_prompts = [tokenizer.apply_chat_template(p['messages'], add_generation_prompt=True, tokenize=False) for p in batch_prompts]

        outputs = agent.generate(batch_prompts, **generation_kwargs)
        if isinstance(agent, GptAgent):
            outputs = [outputs]
            batch_prompts = [batch_prompts]
        for idx, (prompt, output) in enumerate(zip(batch_prompts, outputs)):
            # remove messages from the prompt
            prompt = refineing_prompts[i + idx]
            prompt.pop('messages')
            refined_generations.append({
                **prompt,
                'refine_instruction': output,
                'extracted_refined_instruction': extracted_refined_instruction(output)
            })
    return refined_generations

def generate_response(agent, prompts, batch_size=1, generation_kwargs={}):
    if isinstance(agent, GptAgent):
        batch_size = 1
    elif isinstance(agent, VllmAgent):
        batch_size = len(prompts)
    instruction_prompts = []
    for p in prompts:
        if "extracted_instruction" in p and p["extracted_instruction"] is not None:
            instruction = p["extracted_instruction"]
        else:
            instruction = p["instruction"]
        instruction = instruction.strip("\"")
        if p['input'] != '':
            instruction = f"{instruction} {p['input']}" # p['input'] has prefix "Input:"

        if 'system_prompt' in p:
            instruction_prompts.append([{ 'role': 'system', 'content': p['system_prompt'] }, { 'role': 'user', 'content': instruction }])
        else:
            instruction_prompts.append([{ 'role': 'user', 'content': instruction }])
    if not isinstance(agent, GptAgent):
        instruction_prompts = [tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False) for p in instruction_prompts]
    else:
        instruction_prompts = [{'messages': p} for p in instruction_prompts]

    reponses = []
    for i in trange(0, len(instruction_prompts), batch_size):
        if isinstance(agent, GptAgent):
            batch_prompts = instruction_prompts[i]
        else:
            batch_prompts = [p for p in instruction_prompts[i:i + batch_size]]
        outputs = agent.generate(batch_prompts, **generation_kwargs)

        if isinstance(agent, GptAgent):
            outputs = [outputs]
            batch_prompts = [batch_prompts]

        for idx, (prompt, output) in enumerate(zip(batch_prompts, outputs)):
            # remove messages from the prompt
            reponses.append({
                **prompts[i + idx],
                'final_instruction': prompt,
                'final_instruction_response': output,
            })
    return reponses

def generate_refined_response(agent, prompts, batch_size=1, generation_kwargs={}):
    if isinstance(agent, GptAgent):
        batch_size = 1
    elif isinstance(agent, VllmAgent):
        batch_size = len(prompts)
    extracted_refined_generations = [p for p in prompts if ('extracted_refined_instruction' in p) and (p['extracted_refined_instruction'] is not None)]
    remaining_generations = [p for p in refined_generations if ('extracted_refined_instruction' not in p) or (p['extracted_refined_instruction'] is None)]

    refined_prompts = []
    for p in extracted_refined_generations:
        instruction = p['extracted_refined_instruction']
        if 'system_prompt' in p:
            refined_prompts.append([{ 'role': 'system', 'content': p['system_prompt'] }, { 'role': 'user', 'content': instruction }])
        else:
            refined_prompts.append([{ 'role': 'user', 'content': instruction }])
    if not isinstance(agent, GptAgent):
        refined_prompts = [tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False) for p in refined_prompts]
    else:
        refined_prompts = [{'messages': p} for p in refined_prompts]

    for i in trange(0, len(refined_prompts), batch_size):
        if isinstance(agent, GptAgent):
            batch_prompts = refined_prompts[i]
        else:
            batch_prompts = [p for p in refined_prompts[i:i + batch_size]]
        outputs = agent.generate(batch_prompts, **generation_kwargs)

        if isinstance(agent, GptAgent):
            outputs = [outputs]
            batch_prompts = [batch_prompts]

        for idx, (prompt, output) in enumerate(zip(batch_prompts, outputs)):
            # remove messages from the prompt
            remaining_generations.append({
                **extracted_refined_generations[i + idx],
                'final_refined_instruction_reponse': output,
            })

    return remaining_generations

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
    args.add_argument('--do_refine', action='store_true')
    args.add_argument('--ignore_cache', action='store_true')
    args.add_argument('--add_system_prompt', action='store_true')
    args.add_argument('--no_refinement', action='store_true')
    args.add_argument('--regen_response', action='store_true')
    args.add_argument('--direct_response', action='store_true')
    args.add_argument('--iteration', action='store_true')
    args.add_argument('--temperature', type=float, default=1.0)
    args.add_argument('--top_p', type=float, default=0.9)
    args.add_argument('--top_k', type=int, default=50)
    args.add_argument('--max_new_tokens', type=int, default=2048)

    args = args.parse_args()
    assert not (args.load_8bit and args.load_4bit)
    input_file = args.input_file
    if args.output_file is None:
        output_file = input_file.replace('.jsonl', f'-{os.path.basename(args.query)}.jsonl')
    else:
        output_file = args.output_file
    json_data = []

    # if os.path.exists(output_file):
    #     with open(output_file, 'r', encoding='utf-8') as initial_file:
    #         for line in initial_file:
    #             json_data.append(json.loads(line))
    # else:
    #     with open(output_file, 'w', encoding='utf-8') as initial_file:
    #         json_data = []

    input_data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            input_data.append(json.loads(line))

    if args.sample > 0:
        random.seed(args.seed)
        input_data = random.sample(input_data, args.sample)

    prompts = [get_prompt(p, is_chat=args.use_instruct) for p in input_data]
    if args.iteration:
       for i in range(len(prompts)):
           prompts[i]['final_instruction_response'] = input_data[i]['output']
           prompts[i]['final_instruction'] = input_data[i]['instruction']
           prompts[i]['option'] = input_data[i]['option']
           prompts[i]['idx'] = i

    if (args.add_system_prompt) and ('system_prompt' in prompts[0]):
        system_prompt_map = {i : p['system_prompt'] for i, p in enumerate(prompts)}
    else:
        system_prompt_map = None

    if 'gpt' in args.query:
        agent = GptAgent(api_key=random.choice(API_KEYs), model_name=args.query)

        # step 1: classification
        if (not args.ignore_cache) and (os.path.exists(output_file)):
            print(f'Using cached generations from {output_file}')
            generations = []
            with open(output_file, 'r', encoding='utf-8') as json_file:
                for line in json_file:
                    generations.append(json.loads(line))
        else:
            generations = classification(
                agent=agent,
                generation_kwargs={},
                prompts=prompts,
                batch_size=1
            )
            with open(output_file, 'w', encoding='utf-8') as json_file:
                for g in generations:
                    json_file.write(json.dumps(g, ensure_ascii=False) + '\n')
                
        # Step 2: Add sequential instruction generation
        output_file = output_file.replace('.jsonl', '-generate_instruct.jsonl')
        if (not args.ignore_cache) and (os.path.exists(output_file)):
            print(f'Using cached generations from {output_file}')
            new_generations = []
            with open(output_file, 'r', encoding='utf-8') as json_file:
                for line in json_file:
                    new_generations.append(json.loads(line))
        else:
            new_generations = generation(
                agent=agent,
                generation_kwargs={},
                prompts=generations,
                batch_size=1
            )
            with open(output_file, 'w', encoding='utf-8') as json_file:
                for g in new_generations:
                    json_file.write(json.dumps(g, ensure_ascii=False) + '\n')

        # Step 3 (Optional): Add refinement
        output_file = output_file.replace('.jsonl', '-refine.jsonl')
        if not args.no_refinement:
            if (not args.ignore_cache) and (os.path.exists(output_file)):
                print(f'Using cached generations from {output_file}')
                refined_generations = []
                with open(output_file, 'r', encoding='utf-8') as json_file:
                    for line in json_file:
                        refined_generations.append(json.loads(line))
            else:
                refined_generations = refinement(
                    agent=agent,
                    prompts=new_generations,
                    batch_size=1,
                    generation_kwargs={}
                )
                with open(output_file, 'w', encoding='utf-8') as json_file:
                    for g in refined_generations:
                        json_file.write(json.dumps(g, ensure_ascii=False) + '\n')
        else:
            refined_generations = new_generations

        if args.add_system_prompt:
            assert system_prompt_map is not None
            for i, p in enumerate(refined_generations):
                refined_generations[i]['system_prompt'] = system_prompt_map[p['idx']]

        # Step 4: Return the final output
        output_file = output_file.replace('.jsonl', '-response.jsonl')
        if (not args.ignore_cache) and (os.path.exists(output_file)):
            print(f'Using cached generations from {output_file}')
            refined_generations = []
            with open(output_file, 'r', encoding='utf-8') as json_file:
                for line in json_file:
                    refined_generations.append(json.loads(line))
        else:
            refined_generations = generate_response(
                agent=agent,
                prompts=refined_generations,
                batch_size=1,
                generation_kwargs={},
            )
            with open(output_file, 'w', encoding='utf-8') as json_file:
                for g in refined_generations:
                    json_file.write(json.dumps(g, ensure_ascii=False) + '\n')

        # Step 5 (Optional): Get reponse for refined instruction
        output_file = output_file.replace('.jsonl', '-final.jsonl')
        remaining_generations = generate_refined_response(
            agent=agent,
            prompts=refined_generations,
            batch_size=1,
            generation_kwargs={},
        )
        with open(output_file, 'w', encoding='utf-8') as json_file:
            for g in remaining_generations:
                json_file.write(json.dumps(g, ensure_ascii=False) + '\n')

    elif args.query in ['command-r']:
        Agent = CohereAgent(api_key=random.choice(API_KEYs))

        with open(output_file, 'a', encoding='utf-8') as json_file:
            for index in tqdm(range(len(json_data), len(input_data))):
                prompt = prompts[index]
                cohere_answer = Agent.generate(prompt)
                json_data.append({
                    'idx': index,
                    'input': prompt['input'],
                    'prompt': prompt['prompt'],
                    'completions': cohere_answer,
                })
                json_file.write(json.dumps(json_data[-1], ensure_ascii=False) + '\n')
                print(f'↑ has been stored.')
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
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
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

        if args.direct_response:
            output_file = output_file.replace('.jsonl', '-direct_response.jsonl')
            if (not args.ignore_cache) and (os.path.exists(output_file)):
                print(f'Using cached generations from {output_file}')
                refined_generations = []
                with open(output_file, 'r', encoding='utf-8') as json_file:
                    for line in json_file:
                        refined_generations.append(json.loads(line))
            else:
                refined_generations = generate_response(
                    agent=agent,
                    prompts=prompts,
                    batch_size=args.batch_size,
                    generation_kwargs=generation_kwargs,
                )
                # pop the messages and prompt
                for p in refined_generations:
                    p.pop('messages')
                    p.pop('prompt')
                    
                with open(output_file, 'w', encoding='utf-8') as json_file:
                    for g in refined_generations:
                        json_file.write(json.dumps(g, ensure_ascii=False) + '\n')
            # end the process
            import sys
            sys.exit(1)
        # Step 1: classification
        if args.iteration:
            generations_no_change = [g for g in prompts if g['option']=='D']
            prompts = [g for g in prompts if g['option']!='D']
        if (not args.ignore_cache) and (os.path.exists(output_file)):
            print(f'Using cached generations from {output_file}')
            generations = []
            with open(output_file, 'r', encoding='utf-8') as json_file:
                for line in json_file:
                    generations.append(json.loads(line))
        else:
            generations = classification(
                agent=agent,
                generation_kwargs=generation_kwargs,
                prompts=prompts,
                batch_size=args.batch_size,
            )
            if args.iteration:
                for g in generations:
                    g.pop('final_instruction', None)
                    g.pop('final_instruction_response', None)
            with open(output_file, 'w', encoding='utf-8') as json_file:
                for g in generations:
                    json_file.write(json.dumps(g, ensure_ascii=False) + '\n')

        # Step 2: Add sequential instruction generation
        output_file = output_file.replace('.jsonl', '-generate_instruct.jsonl')
        if (not args.ignore_cache) and (os.path.exists(output_file)):
            print(f'Using cached generations from {output_file}')
            new_generations = []
            with open(output_file, 'r', encoding='utf-8') as json_file:
                for line in json_file:
                    new_generations.append(json.loads(line))
        else:
            if args.iteration:
                new_generations = generation(
                  agent=agent,
                  generation_kwargs=generation_kwargs,
                  prompts=generations,
                  batch_size=args.batch_size,
               )
            else:
                new_generations = generation(
                  agent=agent,
                  generation_kwargs=generation_kwargs,
                  prompts=generations,
                  batch_size=args.batch_size,
               )
            with open(output_file, 'w', encoding='utf-8') as json_file:
                for g in new_generations:
                    json_file.write(json.dumps(g, ensure_ascii=False) + '\n')

        # Step 3 (Optional): Add refinement
        output_file = output_file.replace('.jsonl', '-refine.jsonl')
        if not args.no_refinement:
            if (not args.ignore_cache) and (os.path.exists(output_file)):
                print(f'Using cached generations from {output_file}')
                refined_generations = []
                with open(output_file, 'r', encoding='utf-8') as json_file:
                    for line in json_file:
                        refined_generations.append(json.loads(line))
            else:
                refined_generations = refinement(
                    agent=agent,
                    prompts=new_generations,
                    batch_size=args.batch_size,
                    generation_kwargs=generation_kwargs,
                )
                with open(output_file, 'w', encoding='utf-8') as json_file:
                    for g in refined_generations:
                        json_file.write(json.dumps(g, ensure_ascii=False) + '\n')
        else:
            refined_generations = new_generations

        if args.add_system_prompt:
            assert system_prompt_map is not None
            for i, p in enumerate(refined_generations):
                refined_generations[i]['system_prompt'] = system_prompt_map[p['idx']]

        # Step 4: Return the final output
        output_file = output_file.replace('.jsonl', '-response.jsonl')
        if (not args.ignore_cache) and (not args.regen_response) and (os.path.exists(output_file)):
            print(f'Using cached generations from {output_file}')
            refined_generations = []
            with open(output_file, 'r', encoding='utf-8') as json_file:
                for line in json_file:
                    refined_generations.append(json.loads(line))
        else:
            refined_generations = generate_response(
                agent=agent,
                prompts=refined_generations,
                batch_size=args.batch_size,
                generation_kwargs=generation_kwargs,
            )
            with open(output_file, 'w', encoding='utf-8') as json_file:
                for g in refined_generations:
                    json_file.write(json.dumps(g, ensure_ascii=False) + '\n')

        # Step 5 (Optional): Get reponse for refined instruction
        output_file = output_file.replace('.jsonl', '-final.jsonl')
        if not args.no_refinement:
            remaining_generations = generate_refined_response(
                agent=agent,
                prompts=refined_generations,
                batch_size=args.batch_size,
                generation_kwargs=generation_kwargs,
            )
        else:
            remaining_generations = refined_generations
        print(remaining_generations[1].keys())
        print(len(remaining_generations))
        if args.iteration:
            print(generations_no_change[1].keys())
            # pop 'prompt' and 'messages' from the final instruction
            for g in generations_no_change:
                g.pop('prompt', None)
                g.pop('messages', None)
            for g in remaining_generations:
                g.pop('completions', None)
            remaining_generations = remaining_generations + generations_no_change
        with open(output_file, 'w', encoding='utf-8') as json_file:
            for g in remaining_generations:
                json_file.write(json.dumps(g, ensure_ascii=False) + '\n')
