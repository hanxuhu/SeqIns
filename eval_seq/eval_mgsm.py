import json
import numpy as np
from argparse import ArgumentParser
from tqdm import trange
import os
from rouge import Rouge
# import evaluate

def main(args):
    # squad_metric = evaluate.load('squad')
    references = []
    #load the jsonl file
    with open(args.ref_file, 'r', encoding='utf-8') as file:
        for line in file:
            obj = json.loads(line)
            references.append(obj)
    #load json file
    preds = []
    with open(args.test_file, 'r', encoding='utf-8') as file:
        for line in file:
            obj = json.loads(line)
            preds.append(obj)
    #extract 'output' from each item of json
    assert len(preds) == len(references), f'{len(preds)} != {len(references)}'
    rouge = Rouge()

    metric = {}
    translated = 0
    target_lang = args.test_file.replace('.jsonl', '').rsplit('_', 1)[1]
    missing_words_line = []

    answers = []
    predictions = []
    for i in trange(len(preds), desc=f'Evaluating {target_lang}'):
        pred = preds[i]
        pred_output = pred['model_output'].lower()
        if args.is_mistral:
            pred_output = pred_output.split('\n\n')[0]
        answer = references[i]
        question = answer['question'].lower()

        try:
            is_translated = rouge.get_scores(pred_output, question)[0]['rouge-l']['f'] > 0.5
        except ValueError:
            is_translated = False
        # print('Translated:', is_translated)
        if is_translated:
            # print(i)
            translated += 1
        elif 'translate the question' in pred_output:
            translated += 1

    metric['translated'] = translated / len(preds)

    output_file = os.path.join(args.test_file.rsplit('/', 1)[0], 'following.json')
    target_lang = args.test_file.replace('.jsonl', '').rsplit('_', 1)[1]
    data = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as file:
            data = json.load(file)
    data[target_lang] = metric
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_file', type=str, default=None) # target language file
    parser.add_argument('--ref_file', type=str, default=None) # ref (usually en) language file
    parser.add_argument('--is_mistral', action='store_true')
    args = parser.parse_args()
    main(args)
