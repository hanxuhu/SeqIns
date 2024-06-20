#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export MODEL_PATH=$1
export FINAL_SAVE=$2
# multilingual

MODEL=llama-7b
TESTFILE=self-seq/data/lima_500-replaced.jsonl
python3 eval_seq/generate_batch.py \
  --base_model ${MODEL_PATH} \
  --length 1024 \
  --test_file ${TESTFILE} \
  --save_file ${FINAL_SAVE} \
  --batch_size 32 \
  --samples 200 \