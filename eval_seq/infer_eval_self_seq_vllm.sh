#!/bin/bash

export MODEL_PATH=$1

mkdir -p data/testset

# multilingual
MODEL_NAME=llama-7b
TASK=xquad
TRAIN_TYPE=base
MODEL_NAME=$(basename $MODEL_PATH)
mkdir -p eval_results/${MODEL_NAME}

for TESTLANG in en de es ru zh
# for TESTLANG in en zh
do
  TESTFILE=data/test/xquad_${TRAIN_TYPE}_${TESTLANG}.json
  python3 eval_seq/generate_batch.py \
    --base_model ${MODEL_PATH} \
    --length 512 \
    --test_file ${TESTFILE} \
    --batch_size 128 \
    --save_file eval_results/${MODEL_NAME}/${MODEL_NAME}_${TRAIN_TYPE}_${TASK}_${TESTLANG}.jsonl \
    --load_8bit False \
    --use_vllm True \
    --is_chat True \
    --chat_template tulu
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
    --save_file eval_results/${MODEL_NAME}/${MODEL_NAME}_${TRAIN_TYPE}_${TASK}_${TESTLANG}.jsonl \
    --load_8bit False \
    --use_vllm True \
    --is_chat True \
    --chat_template tulu
done
