CHECKPOINT_PATH=$1
MODEL_NAME=$2

export IS_ALPACA_EVAL_2=False

for CHECKPOINT_PATH in simonycl/self-seq-Meta-Llama-3-8B-alpaca_it_llmam_70b simonycl/self-seq-Meta-Llama-3-8B-alpaca_sit_llama_70b simonycl/self-seq-Meta-Llama-3-8B-alpaca_llmam_70b_iter simonycl/self-seq-Meta-Llama-3-8B-alpaca_llmam_70b-iter-2 simonycl/self-seq-Meta-Llama-3-8B-wizardlm simonycl/self-seq-Meta-Llama-3-8B-flancot_full_it_llama_70b simonycl/self-seq-Meta-Llama-3-8B-flancot_full_sit_llama_70b_iter_2
do
	MODEL_NAME=$(basename CHECKPOINT_PATH)	
# # MGSM 8 shot
	python3 -m eval.mgsm.run_eval \
	    --data_dir data/eval/mgsm/ \
	    --save_dir results/mgsm/${MODEL_NAME}-cot-en-0shot \
	    --model $CHECKPOINT_PATH \
	    --tokenizer $CHECKPOINT_PATH \
	    --n_shot 0 \
	    --mode cot-en \
	    --use_vllm \
	    --use_chat_format \
	    --chat_formatting_function tulu
	
	 # MGSM 8 shot
	python3 -m eval.mgsm.run_eval \
	    --data_dir data/eval/mgsm/ \
	    --save_dir results/mgsm/${MODEL_NAME}-trans-cot-0shot \
	    --model $CHECKPOINT_PATH \
	    --tokenizer $CHECKPOINT_PATH \
	    --n_shot 0 \
	    --mode trans-cot \
	    --use_vllm \
	    --use_chat_format \
	    --chat_formatting_function tulu
done
