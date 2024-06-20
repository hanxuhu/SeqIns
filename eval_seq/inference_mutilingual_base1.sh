#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export CUDA_VISIBLE_DEVICES=$1
export LANG=$2
# multilingual

for LORALANG in enzh
do
  for TESTLANG in fr
  do
    TESTFILE=../multilingual-alpaca/xquad_${LANG}.json

   for BASE in bloom-7b1
   do
      LORA=../models/alpaca-mistral-5lang-mistral-7b/checkpoint-600-mistral-7b
      if [[ -f ${LORA}/adapter_model.safetensors && ! -f ../multilingual-alpaca/test-${TESTLANG}_decoded_by_${LORALANG}-lora-${BASE}.jsonl ]]
      then
        echo "==========${LORA} exists. Doing inference now for ${TESTFILE}"
        python generate_batch.py \
          --length 300 \
          --lora_weights ${LORA} \
          --test_file ${TESTFILE} \
          --save_file ../multilingual-alpaca/mistral-7b-700_5lang_xquad_${LANG}.jsonl
      fi
    done
  done
done

