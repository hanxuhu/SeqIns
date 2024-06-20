from datasets import load_dataset
from tqdm import tqdm
import json
#dataset_zh = load_dataset('xnli', "zh")
#dataset_en = load_dataset('xnli', "en")
import argparse
import random
from time import sleep
parser = argparse.ArgumentParser(
                    description='What the program does',)


args = parser.parse_args()
import json
API_BANK = [''] #add your api keys here
# Opening file
#import openai
openai.api_key = API_BANK[0]
with open('../alpaca_data_cleaned.en.json', 'r') as file:
    data_en = json.load(file)


#with open('../multilingual-alpaca/alpaca_data_cleaned.{}.json'.format(args.target), 'r') as file:
#    data_de = json.load(file)

    
    # Output: {'name': 'John', 'age': 30, 'city': 'New York'}
    
#data_en = data_en[5500:]
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
#print(len(data_de))
k = 1000
m = 0 
j = 1

data_en = data_en[12000:]
for i in tqdm(range(len(data_en))):
    item = data_en[i]
    if i % 2 ==0:
       m+=1 
       m = m % len(API_BANK)
       openai.api_key = API_BANK[m]
    #item_en = dataset_en_train[i]
    data = {}
    if item['input'] == "":
       data["input"] =  ""#data_de[i]["instruction"] 
       data["instruction"] = item["instruction"] #"First translate the below instruction into English, then respond the instruction.\nInstruction: " + data_de[i]["instruction"]
       data["output"] = item['output'] #"Translation of instruction: " + item['instruction'] + '\n' + "Response: " + item['output']
    elif k>=0:
       while 1:
         try:
            response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                            {"role": "system", "content": "Paraphase the input sentence given by user."},
                            {"role": "user", "content": "paraphrase this sentence:" + item["input"]},
                        ],
                    temperature=0,
                    max_tokens=256,
                )
            res = response['choices'][0]['message']['content']
            data["instruction"] = "First paraphrase below input, then " + item["instruction"].lower()
            data["input"] = item['input'] #ponse['choices'][0]['message']['content'] #item["input"]
            data["output"] = "Raraphrase:\n" + res + '\n'  + "Result: " + item['output']
            sleep(2)
            break
         except:
            sleep(5)
            print('error')
            m+=1
            m = m  % len(API_BANK)
            openai.api_key = API_BANK[m]
       #print(data)
       #data["output"] = item['output']
       dataset_final.append(data)
       k -=1

dataset_final = dataset_final + data_en # + data_de[:16000]
#print(len(dataset_final))
print(len(data_para))
#random.shuffle(dataset_final)
dataset_final = data_en + data_para
with open('../../data/alpaca/alpaca_paraphrase.json', 'w', encoding='utf-8') as file:
      json.dump(dataset_final,file, ensure_ascii=False)
