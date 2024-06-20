wget -O ../coco/annotations/vqa_train.json https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/vqav2/vqa_train.json
wget -O ../coco/annotations/vqa_val.json https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/vqav2/vqa_val.json
wget -O ../coco/annotations/vqa_test.json https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/vqav2/vqa_test.json
wget -O ../coco/annotations/coco_karpathy_train.json https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json
python make_instruct_data.py
