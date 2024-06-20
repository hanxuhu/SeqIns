from datasets import load_dataset
from tqdm import tqdm
import json
#dataset_zh = load_dataset('xnli', "zh")
#dataset_en = load_dataset('xnli', "en")
import argparse
import random
parser = argparse.ArgumentParser(
                    description='What the program does',)

parser.add_argument('--source', type=str, default="en",
                    help='source language')
parser.add_argument('--target', type=str, default="es",
                    help='target language')
parser.add_argument('--dataset', type=str, default="xnli",
                    help='name of the dataset')

args = parser.parse_args()
import json

# Opening file
with open('../multilingual-alpaca/alpaca_data_cleaned.en.json', 'r') as file:
    data_en = json.load(file)
with open('../multilingual-alpaca/alpaca_paraphrase_only.json', 'r') as file:
    data_paraphrase = json.load(file)
#with opn('../multilingual-alpaca/alpaca_data_cleaned.{}.json'.format(args.target), 'r') as file:
#    data_de = json.load(file)
with open('../multilingual-alpaca/alpaca_repeat_only.json', 'r') as file:
    data_repeat = json.load(file)
    
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
dataset_final = []
print(len(data_en))
dataset_para = []
#print(len(data_de))
prompt_bank =  [ 'First repeat below question, then ', 'First repeat below input then ', 'First repeat below input, then ', 'Please first repeat below input, and then ', 'Please first repeat below input, and then ', 'Please first restate the following input, then', 'Kindly reiterate the given input initially, and ', 'Please kindly repeat the given input initially, and ' ]
for i in tqdm(range(len(data_paraphrase))):
    item = data_paraphrase[i]
    #item_en = dataset_en_train[i]
    data = {}
    if "Result:" not in item['output']:
         data["input"] =  ""#data_de[i]["instruction"] 
         data["instruction"] = item["instruction"] #"First translate the below instruction into English, then respond the instruction.\nInstruction: " + data_de[i]["instruction"]
         data["output"] = item['output'] #"Translation of instruction: " + item['instruction'] + '\n' + "Response: " + item['output']
    else:
         data_item = data_paraphrase[i]
         dataset_para.append(data_item)
         prompt_bank =  [ 'First repeat below question, then ', 'First repeat below input then ']
         #if item['input'] != "":
         #data["instruction"] = item["instruction"]
         data["instruction"] = random.choice(prompt_bank) + item["instruction"].lower()[6:]  #"First repeat below input, then " + item["instruction"].lower()
         data["input"] = item["input"]
         #print('-----input', item['input'])
         #print(item['output'])
         data["output"] = "1. Repeat:\n" + item['input'] + '\n' + '2. '+  item['output'].split("Result")[0] + '\n'+ "Result: " + item['output'].split("Result:")[1]
         # print(data)
         # data["output"] = item['output']
         dataset_final.append(data)
dataset_final1 = []
for i in tqdm(range(len(data_repeat))):
       item = data_repeat[i]
       dataset_para.append(data_item)
       prompt_bank =  [ 'First paraphrase below question, then ', 'First paraphrase below input then ']
       #if item['input'] != "":
       #data["instruction"] = item["instruction"]
       data["instruction"] = random.choice(prompt_bank) + item["instruction"].lower()[6:]  #"First repeat below input, then " + item["instruction"].lower()
       data["input"] = item["input"]
       data["output"] = "1. Paraphrase:\n" + item['input'] + '\n'+ "2. Repeat:\n" + item['input'] + '\n'  + "Result: " + item['output'].split("Result: ")[1]
       #print(data)
       #data["output"] = item['output']
       dataset_final1.append(data)
random.shuffle(data_repeat)
random.shuffle(data_paraphrase)
random.shuffle(dataset_final)
random.shuffle(dataset_final1)
data_repeat = data_repeat[:4000]
data_paraphrase = data_paraphrase[:4000]
dataset_final1 = dataset_final1[:4000]

dataset_final = dataset_final1 +  dataset_final + data_paraphrase + data_repeat  + data_en[:32000] 
print(len(dataset_final))
random.shuffle(dataset_final)
with open('../multilingual-alpaca/alpaca_multi_merge_3step1.json'.format(args.target), 'w', encoding='utf-8') as file:
      json.dump(dataset_final,file, ensure_ascii=False)
