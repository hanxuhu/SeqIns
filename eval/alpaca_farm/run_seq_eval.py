import os
import json
import argparse
import logging
import random
import torch
import datasets
import vllm
# from alpaca_eval import evaluate as alpaca_farm_evaluate
from eval.utils import generate_completions, dynamic_import_function, load_hf_lm, load_hf_tokenizer
from eval.alpaca_farm.prompter import Prompter
from transformers import AutoTokenizer
from functools import partial

def main(args):
    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    logging.info("loading data and model...")
    alpaca_eval_data = datasets.load_dataset("json", data_files=args.prompt_path)["train"]
    # rename the 'seq_instruction' as 'instruction' to be consistent with the prompter
    # only keep instruction and ori_instruction columns
    # alpaca_eval_data = alpaca_eval_data.map(lambda x: {"instruction": x["seq_instruction"]}, remove_columns=[t for t in alpaca_eval_data.column_names if t not in ['ori_instruction', 'instruction']])

    prompts = []
    # chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
    # prompter = Prompter("alpaca")
    # for example in alpaca_eval_data:
    #     prompt = example["instruction"]
    #     prompt = prompter.generate_prompt(prompt)

    #     prompts.append(prompt)
    for example in alpaca_eval_data:
        prompts.append(example["instruction"])

    model_name = os.path.basename(os.path.normpath(args.model_name_or_path)) if args.model_name_or_path is not None else args.openai_engine

    if args.ignore_cache:
        model_results = []
    else:
        if os.path.exists(os.path.join(args.save_dir, f"{model_name}-seq-eval-greedy-long-output.json")):
            print(f"Loading cached results from {args.save_dir}/{model_name}-seq-eval-greedy-long-output.json")

            with open(os.path.join(args.save_dir, f"{model_name}-seq-eval-greedy-long-output.json"), "r") as fin:
                model_results = [json.loads(line) for line in fin]
                model_results = [{k: v.strip('\n') if isinstance(v, str) else v for k, v in example.items()} for example in model_results]
            if len(model_results) == len(prompts):
                print("Loaded cached results.")
                return
        
    prompts = prompts[:args.sample] if args.sample > 0 else prompts

    if args.model_name_or_path is not None:
        # we always load the tokenizer for vllm or hf models
        tokenizer = load_hf_tokenizer(
                model_name_or_path=args.model_name_or_path,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                use_fast_tokenizer=not args.use_slow_tokenizer,
            )

        if args.use_vllm:
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path is not None else args.model_name_or_path,
                tensor_parallel_size=torch.cuda.device_count(),
            )
            
            sampling_params = vllm.SamplingParams(
                temperature=0,  # greedy decoding
                max_tokens=args.max_new_tokens,
                stop=[args.stop_id_sequences] if not args.use_chat_format else None,
            )
            # apply chat formatting
            if args.use_chat_format:
                formatted_prompts = []
                if args.chat_formatting_function == "mistral":
                    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
                    chat_formatting_function = partial(tokenizer.apply_chat_template, tokenize=False, add_generation_prompt=True)
                elif args.chat_formatting_function == "tulu":
                    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
                    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
                    chat_formatting_function = partial(tokenizer.apply_chat_template, tokenize=False, add_generation_prompt=True)
                else:
                    chat_formatting_function = partial(chat_formatting_function, add_bos=False)
                for prompt in prompts:
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = chat_formatting_function(messages)
                    formatted_prompts.append(formatted_prompt)
                prompts = formatted_prompts
                    
            outputs = model.generate(prompts, sampling_params)
            outputs = [it.outputs[0].text.strip('\n') for it in outputs]
        else:
            model = load_hf_lm(
                model_name_or_path=args.model_name_or_path,
                load_in_8bit=args.load_in_8bit,
                device_map="auto",
                gptq_model=args.gptq,
            )
            # modify tokenizer if required
            from transformers import GPTNeoXForCausalLM, OPTForCausalLM
            if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
                tokenizer.model_max_length = model.config.max_position_embeddings
                print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))

            # apply chat formatting
            if args.use_chat_format:
                formatted_prompts = []
                for prompt in prompts:
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = chat_formatting_function(messages)
                    formatted_prompts.append(formatted_prompt)
                prompts = formatted_prompts
            if args.stop_id_sequences is not None:
                stop_id_sequences = tokenizer.encode(args.stop_id_sequences, add_special_tokens=False)[1:]
            else:
                stop_id_sequences = None
                
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                stop_id_sequences=[stop_id_sequences] if stop_id_sequences is not None else None,
                do_sample=False,
                temperature=0,
                batch_size=args.eval_batch_size if args.eval_batch_size else 1,
                use_cache=True,
                no_repeat_ngram_size=10,
            )
    else:
        openai_query_cache_path = os.path.join(args.save_dir, "openai_query_cache.jsonl")
        openai_func = query_openai_model if args.openai_engine == "text-davinci-003" else query_openai_chat_model
        results = openai_func(
            engine=args.openai_engine,
            instances=[{"id": str(i), "prompt": prompt} for i, prompt in enumerate(prompts)],
            batch_size=args.eval_batch_size if args.eval_batch_size else 10,
            output_path=openai_query_cache_path,
            max_tokens=args.max_new_tokens,
            temperature=0,
            reuse_existing_outputs=True,
        )
        outputs = [result["output"] for result in results]

    model_results = []
    with open(os.path.join(args.save_dir, f"{model_name}-seq-eval-greedy-long-output.json"), "w") as fout:
        for example, output in zip(alpaca_eval_data, outputs):
            example["output"] = output
            example["generator"] = f"{model_name}-seq-eval-greedy-long"
            fout.write(json.dumps(example) + "\n")
            model_results.append(example)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference_path",
        type=str,
        default=None,
        help="Path to the reference outputs. "
             "Alpaca_eval leaderboard use text-davinci-003 to generate the reference outputs, "
             "but they limit the max_tokens to 300, which is a bit unfair for text-davinci-003. "
             "Here we keep this default setup to make numbers comparable to their leaderboard. "
             "But you can also use the regenerated reference outputs with max_tokens=2048 "
             "hosted at https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token.",
    )
    parser.add_argument(
        '--prompt_path',
        type=str,
        default=None,
        help="Path to the prompt file. If not specified, we will use the default prompt file from the alpaca_eval dataset."
    )
    parser.add_argument(
        "--save_dir",
        type=str, 
        default="results/alpaca_farm")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine",
        type=str,
        default=None,
        help="If specified, we will use the OpenAI API to generate the predictions.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8192,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model.",
    )
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="If given, we will use vLLM to generate the predictions - much faster.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--stop_id_sequences",
        type=str,
        default=None,
        help="The stop id sequences for the model.",
    )
    parser.add_argument(
        "--ignore_cache",
        action="store_true",
        help="If given, we will ignore the cache and regenerate the outputs.",
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
