import argparse
import os
import json
import random
import glob
import tqdm
from eval.math.utils import *

def main(args):
    random.seed(42)

    print("Loading data...")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    all_tasks = {}
    tasks = os.listdir(args.data_dir)
    for task in tqdm.tqdm(tasks, desc="Loading tasks"):
        task_data = []
        task_files = glob.glob(os.path.join(args.data_dir, task, "*.json"))
        for task_file in task_files:
            with open(task_file) as fin:
                data = json.load(fin)
                task_data.append(data)
        if args.max_few_shot_examples_per_task and len(task_data) >= args.max_few_shot_examples_per_task:
            task_data = random.sample(task_data, args.max_few_shot_examples_per_task)
        task_data = [{**example, "short-answer": remove_boxed(last_boxed_only_string(example["solution"]))} for example in task_data]

        all_tasks[task] = task_data

    for task, data in all_tasks.items():
        print(f"Task: {task}")
        print(f"Number of examples: {len(all_tasks[task])}")

        task_few_shots = [{
            "problem": example["problem"],
            "cot-answer": example["solution"],
            "short-answer": example["short-answer"]
        } for example in data]

        with open(os.path.join(args.save_dir, f"{task}.json"), "w") as fout:
            json.dump(task_few_shots, fout, indent=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/eval/math/train"
    )
    parser.add_argument(
        "--max_few_shot_examples_per_task", 
        type=int, 
        default=10, 
        help="maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="data/eval/math/cot-prompt"
    )
    parser.add_argument(
        "--template_file", 
        type=str, 
        default="templates/eval_template.json",
        help="The template file to use for generating prompts."
    )
    args = parser.parse_args()

    main(args)
