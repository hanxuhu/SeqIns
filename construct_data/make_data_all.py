from datasets import load_dataset
from tqdm import tqdm
import json
# dataset_zh = load_dataset('xnli', "zh")
# dataset_en = load_dataset('xnli', "en")
import argparse

parser = argparse.ArgumentParser(
                    description='What the program does',)

parser.add_argument('--target', type=str, default="en",
                    help='target language')
parser.add_argument('--dataset', type=str, default="xquad",
                    help='name of the dataset')
parser.add_argument('--typename', type=str, default="baseline",
                    help='type name')
parser.add_argument('--chat_template', type=str, default="tulu")

args = parser.parse_args()
import json

# Opening file

if args.dataset=='alpaca':
   with open('../multilingual_alpaca_dataset/alpaca_data_cleaned.en.json', 'r') as file:
     dataset = json.load(file)
elif 'xquad' == args.dataset :
   dataset = load_dataset('xquad', 'xquad'+'.'+args.target )['validation']
elif 'commonsense_qa' == args.dataset:
   dataset = load_dataset('tau/commonsense_qa', split='validation')
#dataset = load_dataset("xquad",'xquad.{}'.format(args.target))
    # Output: {'name': 'John', 'age': 30, 'city': 'New York'}

dataset_en = load_dataset('xquad', 'xquad.en')['validation']

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

data_list = []
if args.dataset == 'commonsense_qa' and args.typename == 'base':
   for i in range(len(dataset)):
     item = dataset[i]
     data = {}
     data['instruction'] = "Select the correct choice to answer the question in the input"
     if len(item['choices']['label']) == 4:
        data['input'] = 'Question: ' + item['question'] + '\n' + 'Choices: ' + item['choices']['label'][0] + ':'+ item['choices']['text'][0] + ', ' + item['choices']['label'][1] + ':'+ item['choices']['text'][1] + ', ' + item['choices']['label'][2] + ':'+ item['choices']['text'][2] + ', ' + item['choices']['label'][3] + ':'+ item['choices']['text'][3]
     if len(item['choices']['label']) == 5:
        data['input'] = 'Question: ' + item['question'] + '\n' + 'Choices: ' + item['choices']['label'][0] + ':'+ item['choices']['text'][0] + ', ' + item['choices']['label'][1] + ':'+ item['choices']['text'][1] + ', ' + item['choices']['label'][2] + ':'+ item['choices']['text'][2] + ', ' + item['choices']['label'][3] + ':'+ item['choices']['text'][3] + ', ' + item['choices']['label'][4] + ':'+ item['choices']['text'][4]
     #input_en = 'context: ' + dataset_en[i]['context'] + '\n' + 'question: ' + dataset_en[i]['question']
     data['output'] = item['answerKey']  #'Translation of input:' + input_en + '\n' + "result: "+ dataset_en[i]['answers']['text'][0]
     data_list.append(data)
   with open('../data/test/commonsense_qa.json'.format(args.target), 'w', encoding='utf-8') as file:
      json.dump(data_list,file, ensure_ascii=False)
if args.dataset == 'commonsense_qa' and args.typename == 'repeat':
   for i in range(len(dataset)):
      item = dataset[i]
      data = {}
      # data['instruction'] = "First paraphrase the input, then select the correct choice to answer the question in the input"
      data['instruction'] = "Please repeat the input, then select the correct choice to answer the question in the input."
      if len(item['choices']['label']) == 4:
         data['input'] = 'Question: ' + item['question'] + '\n' + 'Choices: ' + item['choices']['label'][0] + ':'+ item['choices']['text'][0] + ', ' + item['choices']['label'][1] + ':'+ item['choices']['text'][1] + ', ' + item['choices']['label'][2] + ':'+ item['choices']['text'][2] + ', ' + item['choices']['label'][3] + ':'+ item['choices']['text'][3]
      if len(item['choices']['label']) == 5:
         data['input'] = 'Question: ' + item['question'] + '\n' + 'Choices: ' + item['choices']['label'][0] + ':'+ item['choices']['text'][0] + ', ' + item['choices']['label'][1] + ':'+ item['choices']['text'][1] + ', ' + item['choices']['label'][2] + ':'+ item['choices']['text'][2] + ', ' + item['choices']['label'][3] + ':'+ item['choices']['text'][3] + ', ' + item['choices']['label'][4] + ':'+ item['choices']['text'][4]
  #input_en = 'context: ' + dataset_en[i]['context'] + '\n' + 'question: ' + dataset_en[i]['question']
      data['output'] = item['answerKey']  #'Translation of input:' + input_en + '\n' + "result: "+ dataset_en[i]['answers']['text'][0]
      data_list.append(data)
   with open('data/test/commonsense_qa_repeat.json'.format(args.target), 'w', encoding='utf-8') as file:
      json.dump(data_list,file, ensure_ascii=False)

if args.dataset == 'xquad' and args.typename == 'base':
   for i in range(len(dataset)):
       item = dataset[i]
       data = {}
       data['instruction'] = "First, translate the input into English. Then, answer the translated question based on the translated context. Your answer should be directly extracted from the context, and it should be a single entity, name, or number, not a sentence."#"Answer the question in the input based on the given context"
       data['input'] = 'Context: ' + item['context'] + '\n' + 'Question: ' + item['question']    #input_en = 'context: ' + dataset_en[i]['context'] + '\n' + 'question: ' + dataset_en[i]['question']
       data['output'] = item['answers']['text'][0]  #'Translation of input:' + input_en + '\n' + "result: "+ dataset_en[i]['answers']['text'][0]
       data_list.append(data)

   with open('../data/test/xquad_base_{}.json'.format(args.target), 'w', encoding='utf-8') as file:
        json.dump(data_list,file, ensure_ascii=False)

if args.dataset == 'xquad' and args.typename == 'trans':
   for i in range(len(dataset)):
       item = dataset[i]
       data = {}
       data['instruction'] = "First, translate the input into English. Then, answer the translated question based on the translated context. Your answer should be directly extracted from the context, and it should be a single entity, name, or number, not a sentence."
       data['input'] = 'Context: ' + item['context'] + '\n' + 'Question: ' + item['question']
      #  input_en = 'context: ' + dataset_en[i]['context'] + '\n' + 'question: ' + dataset_en[i]['question']
       data['output'] =  dataset_en[i]['answers']['text'][0]  #'Translation of input:' + input_en + '\n' + "result: "+ dataset_en[i]['answers']['text'][0
       data_list.append(data)
   with open('../data/test/xquad_trans_{}.json'.format(args.target), 'w', encoding='utf-8') as file:
        json.dump(data_list,file, ensure_ascii=False)


