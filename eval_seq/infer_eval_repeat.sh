#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
# export MODEL_PATH=$1

mkdir -p data/testset

# multilingual
MODEL_NAME=llama-7b
TASK=csqa
# for MODEL_PATH in /mnt/nfs/public/hf/models/meta-llama/Llama-2-7b-hf /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-8B simonycl/self-seq-Llama-2-7b-hf-new simonycl/self-seq-7b-baseline simonycl/self-seq-combined-Llama-2-7b-epoch-1 /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B /mnt/nfs/public/hf/models/meta-llama/Llama-2-70b-hf
# for MODEL_PATH in /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B
# for MODEL_PATH in /mnt/data/Seq_IT/output/self-seq-Llama-2-7b-hf-new-without-ab
for MODEL_PATH in simonycl/self-seq-Llama-2-7b-hf-tulu
do
  MODEL_NAME=$(basename $MODEL_PATH)
  # MODEL_NAME=${MODEL_NAME}_${PROMPT_TYPE}

  echo MODEL_NAME: ${MODEL_NAME}

  mkdir -p eval_results/csqa_repeat

  TESTFILE=data/test/commonsense_qa_repeat.json
  python3 eval_seq/generate_batch.py \
    --base_model ${MODEL_PATH} \
    --length 128 \
    --test_file ${TESTFILE} \
    --batch_size 64 \
    --samples 100 \
    --save_file eval_results/csqa_repeat/${MODEL_NAME}.json \
    --load_8bit False \
    --use_vllm True
done

# self-seq-7B-1-3-new self-seq-alpaca-cleaned_wizardlm_replaced
