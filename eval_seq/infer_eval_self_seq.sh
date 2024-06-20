#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export MODEL_PATH=$1

mkdir -p data/testset

# multilingual
MODEL_NAME=llama-7b
TASK=xquad
TRAIN_TYPE=base
for TESTLANG in en de es ru zh
do
  TESTFILE=data/test/xquad_${TRAIN_TYPE}_${TESTLANG}.json
  python3 eval_seq/generate_batch.py \
    --base_model ${MODEL_PATH} \
    --length 512 \
    --test_file ${TESTFILE} \
    --batch_size 128 \
    --save_file data/testset/${MODEL_NAME}_${TRAIN_TYPE}_${TASK}_${TESTLANG}.jsonl \
    --load_8bit False
done

TRAIN_TYPE=trans
for TESTLANG in ar vi th tr hi
do
  TESTFILE=data/test/xquad_${TRAIN_TYPE}_${TESTLANG}.json
  python3 eval_seq/generate_batch.py \
    --base_model ${MODEL_PATH} \
    --length 512 \
    --test_file ${TESTFILE} \
    --batch_size 128 \
    --save_file data/testset/${MODEL_NAME}_${TRAIN_TYPE}_${TASK}_${TESTLANG}.jsonl \
    --load_8bit False
done
