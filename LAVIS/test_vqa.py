# -*- coding: utf-8 -*- 
from collections import Counter
import jsonlines
import torch
from PIL import Image
import random
# setup device to use
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load sample image
with open('../coco/annotations/vqa_val.json', 'r') as file:
    # Read the contents of the fileI
    data_all = json.load(file)

#th open(data_path, 'r') as f:
'''
def evaluate(
    data_path: str,
    ):
    
    data_path: path to the jsonl file
    output_path: path to the output file
    
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
'''
data_final = []

with open('../coco/annotations/gqa/testdev_balanced_questions_en.json', 'r') as f:
       data_all = json.load(f)
for key in data_all.keys():
       question = data_all[key]['question']
       label = data_all[key]['answer']
       imageId = data_all[key]['imageId']
       url = "../coco/images/gqa/images/" + imageId + ".jpg"
       item = { "url":url, "question":question, 'answer':label }
       data_final.append(item)
'''
for data in data_all:
  #data = json.loads(line) 
  #print(data)
  #data = data_all[key]
  #dir_left = "../marvl-image/zh/images/"+  data["concept"] + "/" + data["left_img"]
  #dir_right = "../marvl-image/zh/images/"+  data["concept"] + "/" + data["right_img"]
  #caption = data["caption"]
  item = { "url":'../coco/images/' + data["image"] , "question":data["question"], 'answer':data['answer'] }
  data_final.append(item)



raw_image_all = []
for item in data_final:
  #raw_image0 = Image.open(item["dir_left"]).convert("RGB")
  #raw_image1 = Image.open(item["dir_right"]).convert("RGB")
  #raw_image0 = Image.open("ex0_0.jpg").convert("RGB")
  #raw_image1 = Image.open("ex0_1.jpg").convert("RGB")
  #new_width = 500
  #new_height = 500
  #raw_image0 = raw_image0.resize((new_width, new_height))
  #raw_image1 = raw_image1.resize((new_width, new_height))
  raw_image = Image.open(item["url"]).convert("RGB")#get_concat_h_resize(raw_image0, raw_image1).save('../concat.jpg')
  #print("concat")
  raw_image_all.append(raw_image)
'''
from lavis.models import load_model_and_preprocess
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#print("raw image", raw_image)
#print("raw image1", raw_image1)
#print("raw image0", raw_image0)
# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct",model_type="vicuna7b", is_eval=True, device=device)
# preprocess the image
# vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
res_all = []
random.shuffle(data_final)
print(len(data_final))
for i in range(len(data_final)):
    raw_image = Image.open(data_final[i]['url']).convert("RGB") #data_final[i]['url']
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    res = model.generate({"image": image,"prompt":'Please first describe the input image then answer the question: '+ data_final[i]["question"] })#+  data_final[i]["question"]})#"Based on the image, is this statement true or false? 'The left image contains twice the number of cats as the right image, and at least two cats in total are standing.'"})#"Based on the image, is this statement true or false? 'The left image contains twice the number of dogs as the right image, and at least two dogs in total are standing.' Answer: "})#data_final[i]["caption"]})
    print(res)
    if len(res.lower().split('answer'))>1:
          text = res.lower().split('answer')[1].strip()
    else:
          text = 'none'
    label = data_final[i]['label']
    print(text)
          #label_count = Counter(label)
          #label_most = #label_count.most_common(1)[0][0]
    print(label)
          #print(label_most)
    if text == label or label in text:
            acc += 1
    print(acc/i)
     #print(data_final[i]['answer'])
    res_all.append(res)
with jsonlines.open('outputs_describe_vqa_random.jsonl', mode='w') as writer:
   for i in range(len(res_all)):    
      writer.write({"text":res_all[i], 'label': data_final[i]['answer'] })
# ['a large fountain spewing water into the air']
