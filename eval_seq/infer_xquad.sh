#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export MODEL_PATH=$1

export PYTHONPATH="$PWD:$PYTHONPATH"
mkdir -p data/testset

# multilingual
MODEL_NAME=llama-7b
TASK=xquad
TRAIN_TYPE=base
# for MODEL_PATH in simonycl/self-seq-Meta-Llama-3-8B-alpaca_it_llmam_70b simonycl/self-seq-Meta-Llama-3-8B-alpaca_sit_llama_70b simonycl/self-seq-Meta-Llama-3-8B-alpaca_llmam_70b-iter-2 simonycl/self-seq-Meta-Llama-3-8B-flancot_full_it_llama_70b simonycl/self-seq-Meta-Llama-3-8B-flancot_full_sit_llama_70b simonycl/self-seq-Meta-Llama-3-8B-flancot_full_sit_llama_70b_iter_2 simonycl/self-seq-Meta-Llama-3-8B-alpaca_sit_llama_70b-iter3 simonycl/self-seq-Meta-Llama-3-8B-alpaca_sit_llama_70b-iter4
# for MODEL_PATH in simonycl/self-seq-Mistral-7B-v0.1-alpaca_it simonycl/self-seq-Mistral-7B-v0.1-alpaca_sit
for MODEL_PATH in simonycl/self-seq-Mistral-7B-v0.1-alpaca_it_gen simonycl/self-seq-Mistral-7B-v0.1-alpaca_sit_gen
do	
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
	    --chat_template alpaca
	done
	
	TRAIN_TYPE=trans
	for TESTLANG in ar vi th tr hi el
	# for TESTLANG in el
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
	    --chat_template alpaca

    done
done
