#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export CUDA_VISIBLE_DEVICES=$1
# multilingual
MODEL_NAME=llama-7b
TRAIN_TYPE=alpaca_trans
TASK=xquad
for TRAIN_TYPE in base trans
do
  for TESTLANG in de ru es zh el ar vi th tr hi 
  do
    TESTFILE=../../data/test/xquad_${TRAIN_TYPE}_${TESTLANG}.json

    for BASE in ${MODEL_NAME}
    do
      LORA=../models/${TRAIN_TYPE}-5lang-${BASE}
      if [[ -f ${LORA}/adapter_model.safetensors && ! -f ../multilingual-alpaca/test-${TESTLANG}_decoded_by_${LORALANG}-lora-${BASE}.jsonl ]]
      then
        echo "==========${LORA} exists. Doing inference now for ${TESTFILE}"
        python generate_eval.py \
          --lora_weights ${LORA} \
          --test_file ${TESTFILE} \
          --task ${TASK} \
          --save_file ../../data/testset/${MODEL_NAME}_${TRAIN_TYPE}_${TASK}_${TESTLANG}.jsonl
      fi
    done
  done
done

