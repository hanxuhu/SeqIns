import json
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from argparse import ArgumentParser
from tqdm import trange

def main(args):
    #load json file
    preds = []
    with open(args.test_file, 'r') as file:
        for line in file:
            obj = json.loads(line)
            preds.append(obj['output'])
    #extract 'output' from each item of json
    references = []
    #load the jsonl file
    if args.ref_file is None:
        with open(args.test_file, 'r') as file:
            for line in file:
                obj = json.loads(line)
                references.append(obj['input'])
    else:
        with open(args.ref_file, 'r') as file:
            for line in file:
                obj = json.loads(line)
                references.append(obj['output'])
    if len(preds) != len(references):
        references = references[:len(preds)]
    #labels = []
    '''
    for i in range(len(lines)):
        labels.append(lines[i]['input'])
    '''
    #compute the BLEU score
    # smoothie = SmoothingFunction().method4
    # score = 0
    # for i in trange(len(preds), desc='BLEU'):
    #     score += sentence_bleu(references[i], preds[i], smoothing_function=smoothie)
    # print(f'BLEU: {score/len(preds)}')

    from rouge import Rouge
    rouge = Rouge()
    score = 0
    for i in trange(len(preds), desc='ROUGE'):
        score += rouge.get_scores(preds[i], references[i])[0]['rouge-l']['f']
    print(f'ROUGE: {score/len(preds)}')
    from bert_score import score
    P, R, F1 = score(preds, references, lang="en", verbose=True)
    print(f'BERT_SCORE: {F1.mean()}')
#print(score/len(preds))
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--ref_file', type=str, default=None)
    args = parser.parse_args()
    main(args)
