import json
import random
with open('../coco/annotations/coco_karpathy_train.json', 'r') as file:
    cap_data = json.load(file)

with open('../coco/annotations/vqa_train.json', 'r') as file:
    vqa_data = json.load(file)

vqa_data_set = {}
for item in vqa_data:
   vqa_data_set[item['image']] = item
cap_data_set = {}
for item in cap_data:
   cap_data_set[item['image']] = item
res_data = []
for image in vqa_data_set.keys():
   if image in cap_data_set.keys():
     print(1)
     cap_data = cap_data_set[image]
     vqa_data = vqa_data_set[image]
     item = {}
     item['image'] = image
     item['caption'] = 'First describe this image then answer the question: {}\nDescription: {}\nAnswer: {}'.format(vqa_data['question'], cap_data['caption'], vqa_data['answer'][0])
     item['text_input'] = 'First describe this image then answer the question: {}'.format(vqa_data['question'])
     item['text_output'] = 'Description: {}\nAnswer: {}'.format(cap_data['caption'], vqa_data['answer'][0])
     item['image_id'] = cap_data['image_id']
     res_data.append(item)
base_data = []
#answer_prompt = 'Answer the question directly: {}'
for image in vqa_data_set.keys():
   if image in cap_data_set.keys():
     print(1)
     cap_data = cap_data_set[image]
     vqa_data = vqa_data_set[image]
     item = {}
     item['image'] = image
     item['caption'] = 'First describe this image then answer the question: {}\nDescription: {}\nAnswer: {}'.format(vqa_data['question'], cap_data['caption'], vqa_data['answer'][0])
     prompts = ['Answer the question directly: {}'.format(vqa_data['question']), 'Please answer the following question: {}'.format(vqa_data['question']), 'Answer this question: {}'.format(vqa_data['question']), 'You should answer the following question: {}'.format(vqa_data['question'])]
     prompt = random.choice(prompts)
     item['text_input'] = prompt#'Answer the question directly: {}'.format(vqa_data['question'])
     item['text_output'] = 'Answer: {}'.format(cap_data['caption'], vqa_data['answer'][0])
     item['image_id'] = cap_data['image_id']
     base_data.append(item)
print(len(base_data))
final_data = base_data + res_data
random.shuffle(final_data)
with open('../coco/annotations/merge_vqa_train.json', 'w') as file:
    json.dump(final_data, file)
with open('../coco/annotations/describe_vqa_train.json', 'w') as file:
    json.dump(res_data, file)
with open('../coco/annotations/base_vqa_train.json', 'w') as file:
    json.dump(base_data, file)
