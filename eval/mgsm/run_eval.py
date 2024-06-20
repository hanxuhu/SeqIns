import argparse
import os
import re
import json
import random
import torch
import vllm
import evaluate
from eval.utils import (
    generate_completions,
    load_hf_lm_and_tokenizer,
    dynamic_import_function,
)
from transformers import AutoTokenizer
from functools import partial
# from eval.gsm.examplars import EXAMPLARS as GSM_EXAMPLARS
LANGS = ['en', 'es', 'fr', 'de', 'ru', 'zh', 'ja', 'th', 'sw', 'bn', 'te']
# LANGS=['en', 'fr', 'zh', 'ja']


exact_match = evaluate.load("exact_match")
INSTRUCTION_PREFIX = {
    "no-cot": "",
    "cot": "First, let's think step-by-step about the question. Then, answer the question, ",
    "cot-en": "First, let's think step-by-step about the question in English. Then, answer the question, ",
    "trans-cot": "First, let's translate the question to English. Then, think step-by-step about the question. Finally, answer the question, "
}

def main(args):
    # if args.mode in ["cot", "cot-en", "trans-cot"]:
    #     LANGS = ["en", "es", "de", "ru", "th", "sw", "bn", "te"]
    # else:
    #     LANGS = ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn", "te"]

    random.seed(42)
    if args.mode in ["cot-en", "trans-cot"]:
        en_examples = []
        with open(os.path.join(args.data_dir, "examples/en.jsonl")) as fin:
            for line in fin:
                en_examples.append(json.loads(line))

    print("Loading data...")
    examples = {}
    test_data = {}
    for lang in LANGS:
        examples[lang] = []
        test_data[lang] = []
        with open(os.path.join(args.data_dir, f"examples/{lang}.jsonl")) as fin:
            for line in fin:
                examples[lang].append(json.loads(line))
        with open(os.path.join(args.data_dir, f"test/{lang}.jsonl")) as fin:
            for line in fin:
                test_data[lang].append(json.loads(line))
        # some numbers are in the `x,xxx` format, and we want to remove the comma
        # for i in range(len(test_data[lang])):
            # test_data[lang][i]["answer"] = re.sub(r"(\d),(\d)", r"\1\2", test_data[lang][i]["answer"])

        if args.max_num_examples and len(test_data[lang]) > args.max_num_examples:
            test_data[lang] = random.sample(test_data[lang], args.max_num_examples)
        
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    with open(args.template_file) as fin:
        template = json.load(fin)

    demonstrations = {}
    # if args.mode in ["no-cot"]:
    prompt_template = template["prompt_example"]
    prompt_question = template["prompt_question"]
    # else:
    #     prompt_template = template["prompt_example_cot"]
    #     prompt_question = template["prompt_question_cot"]

    prompt_prefix = {}
    if args.n_shot:
        for lang in LANGS:
            demonstrations = []
            GSM_EXAMPLARS = examples[lang]
            en_examples = examples["en"]
            if len(GSM_EXAMPLARS) > args.n_shot:
                GSM_EXAMPLARS = random.sample(GSM_EXAMPLARS, args.n_shot)
            for i, example in enumerate(GSM_EXAMPLARS):
                if args.mode in ["no-cot"]:
                    instruction = INSTRUCTION_PREFIX[args.mode] + "Question: " + example["question"]
                    response = str(example["answer_number"])
                elif args.mode in ["en-only"]:
                    instruction = INSTRUCTION_PREFIX[args.mode] + "Question: " + example["question"]
                    response = str(example["answer_number"])
                elif args.mode in ["cot"]:
                    instruction = INSTRUCTION_PREFIX[args.mode] + "Question: " + example["question"]
                    response = example["answer"]
                elif args.mode in ["cot-en"]:
                    instruction = INSTRUCTION_PREFIX[args.mode] + "Question: " + example["question"]
                    # split in The Answer is: <answer>
                    response = en_examples[i]["answer"].rsplit("The answer is", 1)[0].strip() + "\n" + "The Answer is: " + str(example["answer_number"])
                    # response = en_examples[i]["answer"]
                elif args.mode in ["trans-cot"]:
                    instruction = INSTRUCTION_PREFIX[args.mode] + "Question: " + example["question"]
                    response = "Translation the question into English:\n" + en_examples[i]["question"] + "\n" + en_examples[i]["answer"].rsplit("The answer is", 1)[0].strip() + "\n" + "The Answer is: " + str(example["answer_number"])
                    
                if args.use_chat_format:
                    demonstrations.append(
                        instruction + "\nAnswer: " + response
                    )
                else:
                    demonstrations.append(
                        prompt_template.format(
                            instruction=instruction,
                            response=response
                        )
                    )
            prompt_prefix[lang] = "\n\n".join(demonstrations) + "\n\n"
    else:
        prompt_prefix = {lang: "" for lang in LANGS}

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        if args.use_vllm:
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
                tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
                tensor_parallel_size=torch.cuda.device_count(),
            )
            sampling_params = vllm.SamplingParams(
                temperature=0,
                max_tokens=1024,
                stop=["\n\n"] if not args.use_chat_format else None, # we only use stop token for non-chat format (usually applied to vanilla pretrained language models). For chat format, we will rely on the model knows when to stop.
            )
        else:
            model, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path, 
                tokenizer_name_or_path=args.tokenizer_name_or_path, 
                load_in_8bit=args.load_in_8bit, 
                device_map="auto",
                gptq_model=args.gptq,
                use_fast_tokenizer=not args.use_slow_tokenizer,
            )

    metrics, lang_predictions = {}, {}
    for lang in LANGS:
        examples = test_data[lang]
        if args.use_chat_format:
            prompts = []
            if args.chat_formatting_function == "mistral":
                tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
                chat_formatting_function = partial(tokenizer.apply_chat_template, tokenize=False, add_generation_prompt=True)
            elif args.chat_formatting_function == "tulu":
                tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
                tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
                chat_formatting_function = partial(tokenizer.apply_chat_template, tokenize=False, add_generation_prompt=True)
            else:
                chat_formatting_function = dynamic_import_function(args.chat_formatting_function)
                chat_formatting_function = partial(chat_formatting_function, add_bos=False)
            for example in examples:
                messages = [{"role": "user", "content": prompt_prefix[lang] + INSTRUCTION_PREFIX[args.mode] + "Question: " + example["question"]}]
                prompt = chat_formatting_function(messages)
                prompt += " Answer:" if prompt[-1] in ["\n", " "] else " Answer:"
                # prompt += "Answer:" if prompt[-1] in ["\n", " "] else " Answer:"
                prompts.append(prompt)
        else:
            # prompts = [prompt_prefix + "Question: " + example["question"].strip() + "\nAnswer:" for example in test_data]
            prompts = [prompt_prefix[lang] + prompt_question.format(instruction=INSTRUCTION_PREFIX[args.mode] + example["question"]) for example in examples]

        if args.use_vllm:
            # We need to remap the outputs to the prompts because vllm might not return outputs for some prompts (e.g., if the prompt is too long)
            generations = model.generate(prompts, sampling_params)
            prompt_to_output = {
                g.prompt: g.outputs[0].text for g in generations
            }
            outputs = [prompt_to_output[prompt] if prompt in prompt_to_output else "" for prompt in prompts]
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
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=512,
                batch_size=args.eval_batch_size,
                # stop_id_sequences=[[new_line_token]] if not args.use_chat_format else None,  # we only use stop token for non-chat format (usually applied to vanilla pretrained language models). For chat format, we will rely on the model knows when to stop.
                do_sample=False,
            )
        # else:
        #     instances = [{"id": prompt, "prompt": prompt} for _, prompt in enumerate(prompts)]
        #     results = query_openai_chat_model(
        #         engine=args.openai_engine,
        #         instances=instances,
        #         batch_size=args.eval_batch_size if args.eval_batch_size else 10,
        #         output_path=os.path.join(args.save_dir, f"openai_results.jsonl"),
        #     )
        #     outputs = [result["output"] for result in results]

        predictions = []
        # translated = 0
        # cot = 0
        # answer = 0
        for output in outputs:
            # replace numbers like `x,xxx` with `xxxx`
            output = re.sub(r"(\d),(\d)", r"\1\2", output)
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
            if numbers:
                predictions.append(numbers[-1])
            else:
                predictions.append(output)
    
        print("Calculating accuracy...")
        targets = [str(example["answer_number"]) for example in examples]

        em_score = exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]
        print(f"Exact match for {lang}: {em_score}")

        predictions = [{
            "question": example["question"],
            "answer": example["answer_number"],
            "model_output": output,
            "prediction": pred
        } for example, output, pred in zip(examples, outputs, predictions)]

        lang_predictions[lang] = predictions
        metrics[lang] = {
            "exact_match": em_score
        }

        with open(os.path.join(args.save_dir, f"predictions_{lang}.jsonl"), "w") as fout:
            for prediction in predictions:
                fout.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump(metrics, fout, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/eval/mgsm"
    )
    parser.add_argument(
        "--max_num_examples", 
        type=int, 
        default=None, 
        help="maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="results/mgsm"
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
        default=8, 
        help="max number of examples to use for demonstration."
    )
    parser.add_argument(
        "--mode", 
        type=str,
        default="cot",
        choices=["no-cot", "cot", "cot-en", "trans-cot"]
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
