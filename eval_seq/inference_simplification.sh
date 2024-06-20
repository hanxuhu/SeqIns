#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export CUDA_VISIBLE_DEVICES=$1
export STEP=$2
export LORA_PATH=$3
export FINAL_SAVE=$4
# multilingual

for LORALANG in all
do
  for TESTLANG in en
  do
    TESTFILE=../../data/commonsense_qa.json

   for BASE in bloom-7b1
   do
      LORA=${PATH}
      if [[ -f ${LORA_PATH}/adapter_model.safetensors && ! -f ../multilingual-alpaca/test-${TESTLANG}_decoded_by_${LORALANG}-lora-${BASE}.jsonl ]]
      then
        echo "==========${LORA} exists. Doing inference now for ${TESTFILE}"
        python generate_batch.py \
          --lora_weights ${LORA_PATH} \
          --length 300 \
          --test_file ${TESTFILE} \
          --save_file ${FINAL_SAVE}
      fi
    done
  done
done

