from datasets import load_dataset
import random
import json
import os
from argparse import ArgumentParser
from utils import Prompter

INSTRUCTION="First, translate the input into English. Then, answer the translated question based on the translated context. Your answer should be directly extracted from the context, and it should be a single entity, name, or number, not a sentence."
INSTRUCTION_NO_TRANSLATE="Answer the question based on the context. Your answer should be directly extracted from the context, and it should be a single entity, name, or number, not a sentence."
PROMPTER=Prompter("alpaca")
LABEL_TEMPLATE="Translate context into English:\n{context_en}\nTranslated Question: {question_en}\nAnswer: {answer_en}"
LABEL_TEMPLATE_NO_TRANSLATE="Answer: {answer_en}"

FEW_SHOT_PATH='data/xquad/examples'
# def make_few_shot_examples(dataset, num_examples):
#     examples = []
#     dataset_shuffled = dataset.shuffle()
#     for i in range(num_examples):
#         examples.append({
#             'context': dataset_shuffled[i]['context'],
#             'question': dataset_shuffled[i]['question'],
#             'answers': dataset_shuffled[i]['answers'],
#         })
#     return examples

# def main():
#     random.seed(42)
#     dataset = load_dataset('rajpurkar/squad_v2', split='train')
#     # filter if 'answers' is empty
#     dataset = dataset.filter(lambda x: len(x['answers']['text']) > 0)
#     few_shot_examples = make_few_shot_examples(dataset, 4)
#     os.makedirs('data/test/few_shot', exist_ok=True)
#     with open('data/test/few_shot/xquad_few_shot_en.jsonl', 'w', encoding='utf-8') as file:
#         for example in few_shot_examples:
#             json.dump(example, file)
#             file.write('\n')

def get_fewshots(examples_tar, examples_en):
    few_shot_examples = []
    for i in range(len(examples_tar)):
        instruction = INSTRUCTION
        input = 'Context: ' + examples_tar[i]['context'] + '\n' + 'Question: ' + examples_tar[i]['question']
        label = LABEL_TEMPLATE.format(context_en=examples_en[i]['context'], question_en=examples_en[i]['question'], answer_en=examples_en[i]['answers']['text'][0])

        few_shot_examples.append(PROMPTER.generate_prompt(instruction, input, label))
    return '\n\n'.join(few_shot_examples)

def get_fewshots_no_translate(examples_tar):
    few_shot_examples = []
    for i in range(len(examples_tar)):
        instruction = INSTRUCTION_NO_TRANSLATE
        input = 'Context: ' + examples_tar[i]['context'] + '\n' + 'Question: ' + examples_tar[i]['question']
        label = LABEL_TEMPLATE_NO_TRANSLATE.format(answer_en=examples_tar[i]['answers']['text'][0])

        few_shot_examples.append(PROMPTER.generate_prompt(instruction, input, label))
    return '\n\n'.join(few_shot_examples)

def main():
    parser = ArgumentParser(
                    description='What the program does',)

    parser.add_argument('--target', type=str, default="en",
                        help='target language')
    parser.add_argument('--dataset', type=str, default="xquad",
                        help='name of the dataset')
    parser.add_argument('--typename', type=str, default="zeroshot",
                        help='type name', choices=['zeroshot', 'fewshot', 'fewshot_en', 'fewshot_multi', 'fewshot_translated', 'fewshot_answer_only'])
    args = parser.parse_args()

    if args.dataset == 'xquad':
        dataset = load_dataset('xquad', f'xquad.{args.target}')['validation']
    else:
        raise NotImplementedError
    en_dataset = load_dataset('xquad', 'xquad.en')['validation']

    data_list = []
    if args.typename == 'zeroshot':
        for i in range(len(dataset)):
            item = dataset[i]
            data = {}
            data['instruction'] = INSTRUCTION
            data['input'] = 'Context: ' + item['context'] + '\n' + 'Question: ' + item['question']
            data['target'] = item['answers']['text'][0]
            data_list.append(data)
    elif args.typename == 'fewshot':
        en_fewshot_path = os.path.join(FEW_SHOT_PATH, f'{args.dataset}_few_shot_en.jsonl')
        tar_fewshot_path = os.path.join(FEW_SHOT_PATH, f'{args.dataset}_few_shot_{args.target}.jsonl')
        with open(en_fewshot_path, 'r', encoding='utf-8') as file_en, open(tar_fewshot_path, 'r', encoding='utf-8') as file_tar:
            en_fewshots = []
            tar_fewshots = []
            for line in file_en:
                en_fewshots.append(json.loads(line))
            for line in file_tar:
                tar_fewshots.append(json.loads(line))
        
        fewshots = get_fewshots(tar_fewshots, en_fewshots)
        for i in range(len(dataset)):
            data = {}
            data['instruction'] = fewshots + '\n\n' + INSTRUCTION
            data['input'] = 'Context: ' + dataset[i]['context'] + '\n' + 'Question: ' + dataset[i]['question']
            data['target'] = dataset[i]['answers']['text'][0]
            data_list.append(data)
    elif args.typename == 'fewshot_en':
        en_fewshot_path = os.path.join(FEW_SHOT_PATH, f'{args.dataset}_few_shot_en.jsonl')
        with open(en_fewshot_path, 'r', encoding='utf-8') as file_en:
            en_fewshots = []
            for line in file_en:
                en_fewshots.append(json.loads(line))
        fewshots = get_fewshots(en_fewshots, en_fewshots)
        for i in range(len(dataset)):
            data = {}
            data['instruction'] = fewshots + '\n\n' + INSTRUCTION
            data['input'] = 'Context: ' + dataset[i]['context'] + '\n' + 'Question: ' + dataset[i]['question']
            data['target'] = dataset[i]['answers']['text'][0]
            data_list.append(data)
    elif args.typename == 'fewshot_multi':
        multi_fewshot_path = os.path.join(FEW_SHOT_PATH, f'{args.dataset}_few_shot_multi.jsonl')
        en_fewshot_path = os.path.join(FEW_SHOT_PATH, f'{args.dataset}_few_shot_en.jsonl')
        with open(multi_fewshot_path, 'r', encoding='utf-8') as file_multi, open(en_fewshot_path, 'r', encoding='utf-8') as file_en:
            multi_fewshots = []
            en_fewshots = []
            for line in file_multi:
                multi_fewshots.append(json.loads(line))
            for line in file_en:
                en_fewshots.append(json.loads(line))
        fewshots = get_fewshots(multi_fewshots, en_fewshots)
        for i in range(len(dataset)):
            data = {}
            data['instruction'] = fewshots + '\n\n' + INSTRUCTION
            data['input'] = 'Context: ' + dataset[i]['context'] + '\n' + 'Question: ' + dataset[i]['question']
            data['target'] = dataset[i]['answers']['text'][0]
            data_list.append(data)
    elif args.typename == 'fewshot_translated':
        en_fewshot_path = os.path.join(FEW_SHOT_PATH, f'{args.dataset}_few_shot_en.jsonl')
        tar_fewshot_path = os.path.join(FEW_SHOT_PATH, f'{args.dataset}_few_shot_{args.target}.jsonl')
        with open(en_fewshot_path, 'r', encoding='utf-8') as file_en, open(tar_fewshot_path, 'r', encoding='utf-8') as file_tar:
            en_fewshots = []
            tar_fewshots = []
            for line in file_en:
                en_fewshots.append(json.loads(line))
            for line in file_tar:
                tar_fewshots.append(json.loads(line))
        fewshots = get_fewshots(tar_fewshots, en_fewshots)
        
        for i in range(len(dataset)):
            data = {}
            data['instruction'] = fewshots + '\n\n' + INSTRUCTION
            data['input'] = 'Context: ' + dataset[i]['context'] + '\n' + 'Question: ' + dataset[i]['question']
            data['target'] = dataset[i]['answers']['text'][0]
            data['translated_input'] = LABEL_TEMPLATE.format(context_en=tar_fewshots[i]['context'], question_en=tar_fewshots[i]['question'], answer_en="")
            data_list.append(data)
    elif args.typename == 'fewshot_answer_only':
        tar_fewshot_path = os.path.join(FEW_SHOT_PATH, f'{args.dataset}_few_shot_{args.target}.jsonl')
        with open(tar_fewshot_path, 'r', encoding='utf-8') as file_tar:
            tar_fewshots = []
            for line in file_tar:
                tar_fewshots.append(json.loads(line))
        fewshots = get_fewshots_no_translate(tar_fewshots)
        for i in range(len(dataset)):
            data = {}
            data['instruction'] = fewshots + '\n\n' + INSTRUCTION_NO_TRANSLATE
            data['input'] = 'Context: ' + dataset[i]['context'] + '\n' + 'Question: ' + dataset[i]['question']
            data['target'] = dataset[i]['answers']['text'][0]
            data_list.append(data)

    os.makedirs(f'data/{args.dataset}/{args.typename}', exist_ok=True)
    with open(f'data/{args.dataset}/{args.typename}/{args.dataset}_{args.target}.jsonl', 'w', encoding='utf-8') as file:
        for data in data_list:
            json.dump(data, file, ensure_ascii=False)
            file.write('\n')

if __name__ == '__main__':
    main()