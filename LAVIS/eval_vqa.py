'''
input is a jsonl file with the format like:
{"text": ["response: blue"], "label": ["blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue"]}
{"text": ["response: purple"], "label": ["blue", "blue", "purple", "purple", "blue", "purple", "purple", "purple", "purple", "blue"]}
...

so the input is a list of dict, each dict has two keys: "text" and "label"
write a function to read the jsonl file and return false/true for whether the text is in the label for each dict
'''
from collections import Counter
import json
import random
def commonSubstring(a, b):
    for i in range(len(a)-2):
        if a[i:i+3] in b:
            return True
    return False

def evaluate(
    data_path: str,
    ):
    '''
    data_path: path to the jsonl file
    output_path: path to the output file
    ''' 
    acc = 0
    if 'base' in data_path:
      # read the jsonl file
      with open(data_path, 'r') as f:
          data = f.readlines()
      data = data[:300]
      # write the output file
      for line in data:
          line = json.loads(line)
          #text = line['text'][0]
          text = line['text'][0].split(' ')
              
          label = line['label']
          
          print(text[-1])
          print(label)
          label_count = Counter(label)
          label_most = label_count.most_common(1)[0][0]
          #print(label_most)
          #for item in label:
          if text[-1] == label or text[-1] in label:
              acc += 1
              print(1)
          
              #break  
              #else:
              #acc += 0
      acc = acc/len(data)
      print('baseline')
    else:
      # read the jsonl file
      with open(data_path, 'r') as f:
          data = f.readlines()
      # wriate the output file
      i = 0
      #random.shuffle(data)
      data = data[:300]
      for line in data:
          line = json.loads(line)
          
          if len(line['text'][0].lower().split('answer'))>1:
             text = line['text'][0].lower().split('answer')[1].strip()
          else:
             text = 'none'
          label = line['label']
          print(text)
          label_count = Counter(label)
          label_most = label_count.most_common(1)[0][0]
          print(label)
          #print(label_most)
          if text == label or label in text:
            acc += 1
          else:
            acc += 0
          i += 1
          print(acc)
      acc = acc/len(data)
      #print(acc)
    return acc
       
acc = evaluate('outputs_baseline_gqa.jsonl')
print('acc', acc)
