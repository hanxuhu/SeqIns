#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
# export MODEL_PATH=$1

mkdir -p data/testset

# multilingual
MODEL_NAME=llama-7b
TASK=xquad
TRAIN_TYPE=base
PROMPT_TYPES=(fewshot fewshot_en)
for MODEL_PATH in meta-llama/Meta-Llama-3-8B /mnt/nfs/public/hf/models/meta-llama/Llama-2-7b-hf
do
for PROMPT_TYPE in ${PROMPT_TYPES[@]}
do
  MODEL_NAME=$(basename $MODEL_PATH)
  MODEL_NAME=${MODEL_NAME}_${PROMPT_TYPE}

  echo MODEL_NAME: ${MODEL_NAME}

  mkdir -p eval_results/${MODEL_NAME}

  for TESTLANG in en es ar el hi th de ru zh tr vi
  do
    TESTFILE=data/xquad/${PROMPT_TYPE}/xquad_${TESTLANG}.jsonl
    python3 eval_seq/generate_batch.py \
      --base_model ${MODEL_PATH} \
      --length 1024 \
      --test_file ${TESTFILE} \
      --batch_size 64 \
      --save_file eval_results/${MODEL_NAME}/${MODEL_NAME}_${TRAIN_TYPE}_${TASK}_${TESTLANG}.jsonl \
      --load_8bit False \
      --use_vllm True
  done
done