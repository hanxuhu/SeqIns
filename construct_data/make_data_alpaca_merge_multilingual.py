from datasets import load_dataset
from tqdm import tqdm
import json
#dataset_zh = load_dataset('xnli', "zh")
#dataset_enI = load_dataset('xnli', "en")
import argparse
import random
parser = argparse.ArgumentParser(
                    description='What the program does',)


args = parser.parse_args()
import json

# Opening file
with open('../data/alpaca/alpaca_data_cleaned.en.json', 'r') as file:
    data_en = json.load(file)

with open('../data/alpaca/alpaca_data_cleaned.de.json', 'r') as file:
    data_de = json.load(file)

with open('../data/alpaca/alpaca_data_cleaned.zh.json', 'r') as file:
    data_zh = json.load(file)# Output: {'name': 'John', 'age': 30, 'city': 'New York'}

with open('../data/alpaca/alpaca_data_cleaned.es.json', 'r') as file:
    data_es = json.load(file)

with open('../data/alpaca/alpaca_data_cleaned.ru.json', 'r') as file:
    data_ru = json.load(file)

#dataset_zh = load_dataset(args.dataset, args.target)
#dataset_zh_train = dataset_zh['train']
#dataset_en = load_dataset(args.dataset, args.source)
#dataset_en_train = dataset_en['train']
#print(dataset_en_train[1])
dataset_de = []
dataset_es = []
dataset_ru = []
dataset_zh = []
print(len(data_en))
print(len(data_de))
idx = []
for i in tqdm(range(len(data_de))):
    item = data_en[i]
    #item_en = dataset_en_train[i]
    data = {}
    if item['input'] == "":
        data["input"] =  ""#data_de[i]["instruction"] 
        data["instruction"] = item["instruction"] #"First translate the below instruction into English, then respond the instruction.\nInstruction: " + data_de[i]["instruction"]
        data["output"] = item['output'] #"Translation of instruction: " + item['instruction'] + '\n' + "Response: " + item['output']
        dataset_de.append({})
    else:
        #if item['input'] != "":
        #data["instruction"] = item["instruction"]
        data["instruction"] = "First translate below input into English, then " + data_en[i]["instruction"].lower()
        data["input"] = data_de[i]["input"]
        data["output"] = "Translation into English: \n" + data_en[i]['input'] + '\n'  + "Result: " + data_en[i]['output']
        #print(data)
        #data["output"] = item['output']
        idx.append(i)
        dataset_de.append(data)

for i in tqdm(range(len(data_zh))):
    item = data_zh[i]
    #item_en = dataset_en_train[i]
    data = {}
    if item['input'] == "":
        data["input"] =  ""#data_de[i]["instruction"]
        data["instruction"] = item["instruction"] #"First translate the below instruction into English, then respond the instruction.\nInstruction: " + data_de[i]["instruction"]
        data["output"] = item['output'] #"Translation of instruction: " + item['instruction'] + '\n' + "Response: " + item['output']
        dataset_zh.append({})
    else:
        #if item['input'] != "":
        #data["instruction"] = item["instruction"]
        data["instruction"] = "First translate below input into English, then " + data_en[i]["instruction"].lower()
        data["input"] = data_zh[i]["input"]
        data["output"] = "Translation into English: \n" + data_en[i]['input'] + '\n'  + "Result: " + data_en[i]['output']
        #print(data)
        #data["output"] = item['output']
        dataset_zh.append(data)

for i in tqdm(range(len(data_ru))):
    item = data_ru[i]
    #item_en = dataset_en_train[i]
    data = {}
    if item['input'] == "":
        data["input"] =  ""#data_de[i]["instruction"]
        data["instruction"] = data_en[i]["instruction"] #"First translate the below instruction into English, then respond the instruction.\nInstruction: " + data_de[i]["instruction"]
        data["output"] = item['output'] #"Translation of instruction: " + item['instruction'] + '\n' + "Response: " + item['output']
        dataset_ru.append({})
    else:
        #if item['input'] != "":
        #data["instruction"] = item["instruction"]
        data["instruction"] = "First translate below input into English, then " + data_en[i]["instruction"].lower()
        data["input"] = data_ru[i]["input"]
        data["output"] = "Translation into English: \n" + data_en[i]['input'] + '\n'  + "Result: " + data_en[i]['output']
        #print(data)
        #data["output"] = item['output']
        dataset_ru.append(data)

for i in tqdm(range(len(data_es))):
    item = data_es[i]
    #item_en = dataset_en_train[i]
    data = {}
    if item['input'] == "":
        data["input"] =  ""#data_de[i]["instruction"]
        data["instruction"] = data_en[i]["instruction"] #"First translate the below instruction into English, then respond the instruction.\nInstruction: " + data_de[i]["instruction"]
        data["output"] = item['output'] #"Translation of instruction: " + item['instruction'] + '\n' + "Response: " + item['output']
        dataset_es.append({})
    else:
        #if item['input'] != "":
        #data["instruction"] = item["instruction"]
        data["instruction"] = "First translate below input into English, then " + data_en[i]["instruction"].lower()
        data["input"] = data_es[i]["input"]
        data["output"] = "Translation into English: \n" + data_en[i]['input'] + '\n'  + "Result: " + data_en[i]['output']
        #print(data)
        #data["output"] = item['output']
        #data["output"] = item['output']
        dataset_es.append(data)

# shuffle the idx
random.shuffle(idx)
print(len(idx))

dataset_ru = [dataset_ru[i] for i in idx[:4750]]
dataset_es = [dataset_es[i] for i in idx[4750:9500]]
dataset_zh = [dataset_zh[i] for i in idx[9500:14250]]
dataset_de = [dataset_de[i] for i in idx[14250:19000]]

print(len(data_en))
left_idx = list(set(range(len(data_en))) - set(idx[:19000]))
data_en = [data_en[i] for i in left_idx]

# random.shuffle(data_de)
# random.shuffle(data_en)
# random.shuffle(data_zh)
# random.shuffle(data_es)
# random.shuffle(data_ru)
# random.shuffle(dataset_de)
# random.shuffle(dataset_zh)
# random.shuffle(dataset_es)
# random.shuffle(dataset_ru)

# data_final = dataset_ru[:4750] + dataset_es[:4750] + dataset_zh[:4750] + dataset_de[:4750] +data_en #+ data_zh[:6400] + data_es[:6400] + data_ru[:6400]
data_final = dataset_ru + dataset_es + dataset_zh + dataset_de + data_en
random.shuffle(data_final)
print(len(data_final))
with open('../data/alpaca/alpaca_trans_5lang.jsonl', 'w', encoding='utf-8') as file:
    for i, data in enumerate(data_final):
        data['system_prompt'] = ""
        data['idx'] = i

        file.write(json.dumps(data) + '\n')

with open('../data/alpaca/alpaca_data_cleaned.en.json', 'r') as file:
    data_en = json.load(file)

with open('../data/alpaca/alpaca_cleaned.jsonl', 'w', encoding='utf-8') as file:
    for i, data in enumerate(data_en):
        data['system_prompt'] = ""
        data['idx'] = i

        file.write(json.dumps(data) + '\n')