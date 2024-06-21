CHECKPOINT_PATH=$1
MODEL_NAME=$2

export IS_ALPACA_EVAL_2=False

# GSM8K 8 shot
python3 -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --save_dir results/gsm/${MODEL_NAME}-cot-8shot \
    --model $CHECKPOINT_PATH \
    --tokenizer $CHECKPOINT_PATH \
    --n_shot 8 \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function tulu

# # MGSM 8 shot
python3 -m eval.mgsm.run_eval \
    --data_dir data/eval/mgsm/ \
    --save_dir results/mgsm/${MODEL_NAME}-cot-en-8shot \
    --model $CHECKPOINT_PATH \
    --tokenizer $CHECKPOINT_PATH \
    --n_shot 8 \
    --mode cot-en \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function tulu

# # MGSM 8 shot
python3 -m eval.mgsm.run_eval \
    --data_dir data/eval/mgsm/ \
    --save_dir results/mgsm/${MODEL_NAME}-trans-cot-8shot \
    --model $CHECKPOINT_PATH \
    --tokenizer $CHECKPOINT_PATH \
    --n_shot 8 \
    --mode trans-cot \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function tulu

# CodeEval
python3 -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 10 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --save_dir results/codex_humaneval/{$MODEL_NAME}_temp_0_8 \
    --model $CHECKPOINT_PATH \
    --tokenizer $CHECKPOINT_PATH \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function tulu
    
# # AlpacaEval
python3 -m eval.alpaca_farm.run_eval \
    --model_name_or_path $CHECKPOINT_PATH \
    --save_dir results/alpaca_farm/${MODEL_NAME} \
    --eval_batch_size 20 \
    --max_new_tokens 2048 \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function tulu
