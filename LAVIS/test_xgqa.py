# -*- coding: utf-8 -*- 
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
print(len(data_final))
acc = 0
random.shuffle(data_final)
for i in range(int(len(data_final))):
    raw_image = Image.open(data_final[i]['url']).convert("RGB") #data_final[i]['url']
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    res = model.generate({"image": image,"prompt":'Please answer the question: ' +  data_final[i]["question"]})#"Based on the image, is this statement true or false? 'The left image contains twice the number of cats as the right image, and at least two cats in total are standing.'"})#"Based on the image, is this statement true or false? 'The left image contains twice the number of dogs as the right image, and at least two dogs in total are standing.' Answer: "})#data_final[i]["caption"]})
    print(res)
    res = res[0]
    #print(res)
    '''   
    if len(res.lower().split('answer'))>1:
          text = res.lower().split('answer')[1].strip()
    else:
          text = 'null'
    '''
    #text = res.lower()
    label = data_final[i]['answer']
    #print(text)
          #label_count = Counter(label)
          #label_most = #label_count.most_common(1)[0][0]
    if text=='' or text== ' ':
       text = 'NULL'       #print(label_most)
    if text == label: #or label in text or text in label:
            acc += 1
    res_all.append(res)
with jsonlines.open('outputs_base_gqa_notrain.jsonl', mode='w') as writer:
   for i in range(len(res_all)):    
      writer.write({"text":res_all[i], 'label': data_final[i]['answer'] })
# ['a large fountain spewing water into the air']
