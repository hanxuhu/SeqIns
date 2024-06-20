export CUDA_VISIBLE_DEVICES=0

for CHECKPOINT_PATH in mistralai/Mistral-7B-Instruct-v0.2 simonycl/self-seq-Mistral-7B-Instruct-v0.2 simonycl/Mistral-7B-Instruct-v0.2-alpaca-baseline
do
    MODEL_NAME=$(basename $CHECKPOINT_PATH)
    # AlpacaEval
    python3 -m eval.alpaca_farm.run_eval \
        --model_name_or_path $CHECKPOINT_PATH \
        --save_dir results/alpaca_farm/${MODEL_NAME} \
        --eval_batch_size 20 \
        --max_new_tokens 1024 \
        --use_vllm \
        --use_chat_format \
        --chat_formatting_function mistral
done
