# -*- coding: utf-8 -*- 
import jsonlines
import torch
from PIL import Image
import random
# setup device to use
import json
import argparse

parser = argparse.ArgumentParser(description="Inference")

parser.add_argument("--data_path", required=True, help="path to configuration file.")
parser.add_argument("--save_path", required=True, help="path to configuration file.")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load sample image
with open(args.data_path, 'r') as file:
    # Read the contents of the fileI
    data_all = json.load(file)
with open('../coco/annotations/merge_vqa_train.json', 'r') as file:
    data_cap = json.load(file)
#th open(data_path, 'r') as f:
data_final = []
for data in data_all:
  #data = json.loads(line) 
  #print(data)
  #data = data_all[key]
  #dir_left = "../marvl-image/zh/images/"+  data["concept"] + "/" + data["left_img"]
  #dir_right = "../marvl-image/zh/images/"+  data["concept"] + "/" + data["right_img"]
  #caption = data["caption"]
  item = { "url":'../coco/images/' + data["image"] , "question":data["question"], 'answer':data['answer'] }
  data_final.append(item)


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
random.shuffle(data_cap)
random.shuffle(data_final)
for i in range(len(data_final)):
    raw_image = Image.open(data_final[i]['url']).convert("RGB") #data_final[i]['url']
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    random_num = random.randint(1, len(data_cap))
    res = model.generate({"image": image,"prompt":'Frist describe the image then answer the question: '+ data_final[i]["question"] +'\n'+ data_cap[random_num]['text_output'].split('Answer:')[0] })#+  data_final[i]["question"]})#"Based on the image, is this statement true or false? 'The left image contains twice the number of cats as the right image, and at least two cats in total are standing.'"})#"Based on the image, is this statement true or false? 'The left image contains twice the number of dogs as the right image, and at least two dogs in total are standing.' Answer: "})#data_final[i]["caption"]})
    print(res)
    print(data_final[i]['answer'])
    res_all.append(res)
#save to 'outputs_neg_vqa_random.jsonl'
with jsonlines.open(args.save_path, mode='w') as writer:
   for i in range(len(res_all)):    
      writer.write({"text":res_all[i], 'label': data_final[i]['answer'] })
# ['a large fountain spewing water into the air']
