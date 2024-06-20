import argparse
import os
import re
import json
import random
import torch
import vllm
import evaluate
import glob
import tqdm
from eval.utils import (
    generate_completions,
    load_hf_lm_and_tokenizer,
    query_openai_chat_model,
    dynamic_import_function,
)
from eval.math.utils import *

exact_match = evaluate.load("exact_match")


def main(args):
    """
    All data is {
    "problem": "What is 2 + 2?",
    "level": "easy",
    "type" : Algebra,
    "solution": "4"
    }
    """
    random.seed(42)

    print("Loading data...")
    all_tasks = {}
    tasks = os.listdir(os.path.join(args.data_dir, 'test'))
    for task in tqdm.tqdm(tasks, desc="Loading tasks"):
        task_data = []
        task_files = glob.glob(os.path.join(args.data_dir, 'test', task, "*.json"))
        for task_file in task_files:
            with open(task_file) as fin:
                data = json.load(fin)
                task_data.append(data)
        if args.max_num_examples_per_task and len(task_data) >= args.max_num_examples_per_task:
            task_data = random.sample(task_data, args.max_num_examples_per_task)
        task_data = [{**example, "target": remove_boxed(last_boxed_only_string(example["solution"]))} for example in task_data]

        all_tasks[task] = task_data

    with open(args.template_file) as fin:
        template = json.load(fin)

    prompt_template = template["prompt_example"] if args.no_cot else template["prompt_example_cot"]
    prompt_question = template["prompt_question"] if args.no_cot else template["prompt_question_cot"]
        
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    all_prompts = {}
    cot_prompt_files = glob.glob(os.path.join(args.data_dir, "cot-prompt", "*.json"))
    for cot_prompt_file in cot_prompt_files:
        with open(cot_prompt_file) as fin:
            data = json.load(fin)
        
        if args.n_shot:
            if len(data) > args.n_shot:
                data = random.sample(data, args.n_shot)
        prompt = [
            prompt_template.format(
                instruction=example["problem"],
                response=example["short-answer"] if args.no_cot else example["cot-answer"]
            )
            for example in data
        ]
        all_prompts[os.path.basename(cot_prompt_file).replace(".json", "")] = "\n\n".join(prompt)

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        if args.use_vllm:
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
                tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
                tensor_parallel_size=torch.cuda.device_count(),
            )
        else:
            model, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path, 
                tokenizer_name_or_path=args.tokenizer_name_or_path, 
                load_in_8bit=args.load_in_8bit, 
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                gptq_model=args.gptq,
                use_fast_tokenizer=not args.use_slow_tokenizer,
            )
            new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1] # get the last token because the tokenizer may add space tokens at the start.
    else:
        raise NotImplementedError("OpenAI API is not supported for math evaluation.")

    performance = {}
    for task_name in tqdm.tqdm(all_tasks.keys(), desc="Evaluating tasks"):
        task_examples = all_tasks[task_name]
        if args.model_name_or_path:
            prompt_prefix = all_prompts[task_name]
            
            if args.use_chat_format:
                raise NotImplementedError("Evaluation with chat format is not supported for models that are not vllm.")
            else:
                prompts = [prompt_prefix.strip() + "\n\n" + prompt_question.format(instruction=example["problem"]) + "\n\n" for example in task_examples]
                # print(prompts[0])

            if args.use_vllm:
                sampling_params = vllm.SamplingParams(
                    temperature=0,
                    max_tokens=512,
                    # stop=["\n\n"] if not args.use_chat_format else None,  # we only use stop token for non-chat format (usually applied to vanilla pretrained language models). For chat format, we will rely on the model knows when to stop.
                )
                # We need to remap the outputs to the prompts because vllm might not return outputs for some prompts (e.g., if the prompt is too long)
                generations = model.generate(prompts, sampling_params)
                prompt_to_output = {
                    g.prompt: g.outputs[0].text for g in generations
                }
                outputs = [prompt_to_output[prompt] if prompt in prompt_to_output else "" for prompt in prompts]
            # generate with hf model
            else:
                stop_sequence = tokenizer.encode("\n\n", add_special_tokens=False)[-2:] # get the last token because the tokenizer may add space tokens at the start.
                outputs = generate_completions(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_new_tokens=512,
                    temperature=0,
                    batch_size=args.eval_batch_size if args.eval_batch_size else 1,
                    stop_id_sequences=[[stop_sequence]] if not args.use_chat_format else None,  # we only use stop token for non-chat format (usually applied to vanilla pretrained language models). For chat format, we will rely on the model knows when to stop.
                )

        predictions = []
        for example, output in zip(task_examples, outputs):
            print(f"Generated Output: {output}")
            # print(f'Target: {example["target"]}')

            example["raw_output"] = output
            
            # extract the first answer after `the answer is` and before the next period.
            # if there is no such answer, we will just use the raw output.
            extracted_answer = remove_boxed(last_boxed_only_string(output))
            if extracted_answer:
                example["prediction"] = extracted_answer
            else:
                example["prediction"] = output.strip()
            predictions.append(example["prediction"])
        
        os.makedirs(os.path.join(args.save_dir, "predictions"), exist_ok=True)

        with open(os.path.join(args.save_dir, "predictions", f"{task_name}.jsonl"), "w") as fout:
            for example in task_examples:
                fout.write(json.dumps(example) + "\n")

        performance[task_name] = sum([process_results(example)['acc'] for example in task_examples]) / len(task_examples)

        print(f"Task {task_name} - EM: {performance[task_name]}")
    
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        performance["average_exact_match"] = sum(performance.values()) / len(performance)
        print(f"Average EM: {performance['average_exact_match']}")
        json.dump(performance, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/eval/math"
    )
    
    parser.add_argument(
        "--max_num_examples_per_task", 
        type=int, 
        default=None, 
        help="maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="results/math"
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default=None, 
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        default=None, 
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine", 
        type=str, 
        default=None, help="if specified, we will use the OpenAI API to generate the predictions."
    )
    parser.add_argument(
        "--n_shot", 
        type=int, 
        default=3, 
        help="max number of examples to use for demonstration."
    )
    parser.add_argument(
        "--no_cot", 
        action="store_true", 
        help="If given, we're evaluating a model without chain-of-thought."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit", 
        action="store_true", 
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--gptq", 
        action="store_true", 
        help="If given, we're evaluating a 4-bit quantized GPTQ model."
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true", 
        help="If given, we will use the vllm library, which will likely increase the inference throughput."
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
        "--template_file", 
        type=str, 
        default="templates/eval_template.json",
        help="The template file to use for generating prompts."
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
