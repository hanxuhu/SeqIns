from datasets import load_dataset
from tqdm import tqdm
import json
#dataset_zh = load_dataset('xnli', "zh")
#dataset_en = load_dataset('xnli', "en")
import argparse

parser = argparse.ArgumentParser(
                    description='What the program does',)

parser.add_argument('--source', type=str, default="en",
                    help='source language')
parser.add_argument('--target', type=str, default="en",
                    help='target language')
parser.add_argument('--dataset', type=str, default="xquad",
                    help='name of the dataset')

args = parser.parse_args()
import json

# Opening file
dataset_en = load_dataset("xquad",'xquad.en')
dataset = load_dataset("xquad",'xquad.{}'.format(args.target))    
    # Output: {'name': 'John', 'age': 30, 'city': 'New York'}
    

def format_instruction(sample, sample_en):
	return f"""### Instruction:
  The Input below contains one premise and hypothesis, you should firstly translate them into English then classify the relationship between them.
  ### Input:
  Premise: {sample['premise']}
  Hypothesis: {sample['hypothesis']}
  ### Response:
  The translation in English is:
  Premise: {sample_en['premise']}
  Hypothesis: {sample_en['hypothesis']}
  So the relationship between the premise and hypothesis is: {class_mapping[sample['label']]}
"""
#dataset_zh = load_dataset(args.dataset, args.target)
#dataset_zh_train = dataset_zh['train']
#dataset_en = load_dataset(args.dataset, args.source)
#dataset_en_train = dataset_en['train']
#print(dataset_en_train[1])
dataset_en = dataset_en['validation']

data_list = []

for i in range(len(dataset['validation'])):
  item = dataset['validation'][i]
  data = {}
  data['instruction'] = "Answer the question from the given passage. Your answer should be directly extracted from the passage, and it should be a single entity, name, or number, not a sentence."#"Answer the question in the input based on the given context"
  data['input'] = 'Passage: ' + item['context'] + '\n' + 'Question: ' + item['question'] + 'Note: Your answer should be directly extracted from the passage and be a single entity, name, or number. Provide the answer in quotations.' + '\n' + '### Response: Based on the passage, the answer to the question is : '
   #input_en = 'context: ' + dataset_en[i]['context'] + '\n' + 'question: ' + dataset_en[i]['question']
  data['output'] = item['answers']['text'][0]  #'Translation of input:' + input_en + '\n' + "result: "+ dataset_en[i]['answers']['text'][0]
  data_list.append(data)

with open('../multilingual-alpaca/xquad_{}.json'.format(args.target), 'w', encoding='utf-8') as file:
      json.dump(data_list,file, ensure_ascii=False)
