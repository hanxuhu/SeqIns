#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export CUDA_VISIBLE_DEVICES=$1

BASE_MODEL=bigscience/bloom-7b1 #baffo32/decapoda-research-llama-7B-hf #decapoda-research/llama-7b-hf
OUTPUT=lora-bloom-7b1
OUTPUT_DIR=../models/alpaca-5lang-trans-lora-bloom-7b1
#  WORLD_SIZE=4 torchrun --nproc_per_node=4 --master_port 12343  finetune.py \
python finetune.py \
    --base_model ${BASE_MODEL} \
    --data_path ../../data/alpaca/alpaca_trans_5lang.json \
    --output_dir ../${OUTPUT_DIR} \
    --batch_size 64 \
    --micro_batch_size 16 \
    --num_epochs 5 \
    --learning_rate 3e-4 \
    --cutoff_len 512 \
    --val_set_size 500 \
    --lora_r 8 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --lora_target_modules '[query_key_value]' \
    --train_on_inputs \
    --group_by_length
