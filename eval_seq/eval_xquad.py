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

    words = ['result:', 'answer:', 'answer is', 'result is']
    # words = ['answer:']
    metric = {}
    r = 0
    # ac_en = 0
    ac_tar = 0
    ac_with_follow = 0
    translated = 0
    rouge_scores = []
    output_file = os.path.join(args.test_file.rsplit('/', 1)[0], 'metric.json')
    target_lang = args.test_file.replace('.jsonl', '').rsplit('_', 1)[1]
    missing_words_line = []

    answers = []
    predictions = []
    inputs = []
    for i in trange(len(preds), desc=f'Evaluating {target_lang}'):
        pred = preds[i]
        predictions.append({'id': i, 'prediction_text': pred['output']})
        pred_output = pred['output'].lower()
        if args.is_mistral:
            pred_output = pred_output.split('\n\n')[0]
        answer = references[i]
        answers.append({'id': i, 'answers': {'text': [answer['target']]}})
        # target_answer = answer['target'].lower()
        target_lan_answer = answer['target'].lower()

        # try:
        #     is_translated = rouge.get_scores(answer['input'].lower(), pred_output)[0]['rouge-1']['f'] > 0.4
        # except ValueError:
        #     is_translated = False
        # # print('Translated:', is_translated)
        # if is_translated:
        #     translated += 1
        word = ''
        for word in words:
            position = pred_output.find(word)
            if position != -1:
                r+=1
                break
        if position == -1:
            missing_words_line.append(i)
            if (target_lan_answer in pred_output) or (pred_output in target_lan_answer):
                ac_tar += 1
            continue
        else:
            pred = pred_output[position+len(word):].strip()
            # if '\n\n' in pred:
            #     pred = pred.split('\n\n')[0]
            # if 'context:' in pred:
            #     pred = pred.split('context:')[0]
            if (target_lan_answer in pred) or (pred in target_lan_answer):
                ac_with_follow += 1
                ac_tar += 1

    metric['Follow'] = r / len(preds)
    metric['ac'] = ac_tar / len(preds)
    metric['ac_with_follow'] = ac_with_follow / len(preds)
    metric['translated'] = translated / len(preds)
    # squad_results = squad_metric(answers, predictions)
    # metric['squad'] = squad_results

    output_file = os.path.join(args.test_file.rsplit('/', 1)[0], 'metric.json')
    output_missing_line = os.path.join(args.test_file.rsplit('/', 1)[0], 'missing_words_line.json')
    target_lang = args.test_file.replace('.jsonl', '').rsplit('_', 1)[1]
    data = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as file:
            data = json.load(file)
    data[target_lang] = metric
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)
    missing_lines = {}
    if os.path.exists(output_missing_line):
        with open(output_missing_line, 'r') as file:
            missing_lines = json.load(file)
    missing_lines[target_lang] = missing_words_line
    with open(output_missing_line, 'w') as file:
        json.dump(missing_lines, file, indent=4)
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_file', type=str, default=None) # target language file
    parser.add_argument('--ref_file', type=str, default=None) # ref (usually en) language file
    parser.add_argument('--is_mistral', action='store_true')
    args = parser.parse_args()
    main(args)
