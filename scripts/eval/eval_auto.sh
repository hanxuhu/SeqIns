export CUDA_VISIBLE_DEVICES=0
CHECKPOINT_PATH=$1
MODEL_NAME=$2

export IS_ALPACA_EVAL_2=False

# # # TyDiQA
# python3 -m eval.tydiqa.run_eval \
#     --data_dir data/eval/tydiqa/ \
#     --n_shot 1 \
#     --max_num_examples_per_lang 100 \
#     --max_context_length 512 \
#     --save_dir results/tydiqa/${MODEL_NAME} \
#     --model $CHECKPOINT_PATH \
#     --tokenizer $CHECKPOINT_PATH \
#     --eval_batch_size 4 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# GSM8K 8 shot
python3 -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --save_dir results/gsm/${MODEL_NAME}-cot-8shot \
    --model $CHECKPOINT_PATH \
    --tokenizer $CHECKPOINT_PATH \
    --n_shot 8 \
    --use_vllm

python3 -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --save_dir results/gsm/${MODEL_NAME}-cot-0shot \
    --model $CHECKPOINT_PATH \
    --tokenizer $CHECKPOINT_PATH \
    --n_shot 0 \
    --use_vllm

# python3 -m eval.mgsm.run_eval \
#     --data_dir data/eval/mgsm/ \
#     --save_dir results/mgsm/${MODEL_NAME}-cot-8shot \
#     --model $CHECKPOINT_PATH \
#     --tokenizer $CHECKPOINT_PATH \
#     --n_shot 8 \
#     --mode no-cot \
#     --use_vllm


# # MGSM 8 shot
# python3 -m eval.mgsm.run_eval \
#     --data_dir data/eval/mgsm/ \
#     --save_dir results/mgsm/${MODEL_NAME}-cot-8shot \
#     --model $CHECKPOINT_PATH \
#     --tokenizer $CHECKPOINT_PATH \
#     --n_shot 8 \
#     --mode cot \
#     --use_vllm

# # MGSM 8 shot
# python3 -m eval.mgsm.run_eval \
#     --data_dir data/eval/mgsm/ \
#     --save_dir results/mgsm/${MODEL_NAME}-cot-en-8shot \
#     --model $CHECKPOINT_PATH \
#     --tokenizer $CHECKPOINT_PATH \
#     --n_shot 8 \
#     --mode cot-en \
#     --use_vllm

# # MGSM 8 shot
# python3 -m eval.mgsm.run_eval \
#     --data_dir data/eval/mgsm/ \
#     --save_dir results/mgsm/${MODEL_NAME}-trans-cot-8shot \
#     --model $CHECKPOINT_PATH \
#     --tokenizer $CHECKPOINT_PATH \
#     --n_shot 8 \
#     --mode trans-cot \
#     --use_vllm

# # BBH
python3 -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/bbh/${MODEL_NAME}-cot \
    --model $CHECKPOINT_PATH \
    --tokenizer $CHECKPOINT_PATH \
    --max_num_examples_per_task 40 \
    --use_vllm

# MMLU 5 shot
python3 -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/${MODEL_NAME}-5shot \
    --model_name_or_path $CHECKPOINT_PATH \
    --tokenizer_name_or_path $CHECKPOINT_PATH \
    --eval_batch_size 4 \

# CodeEval
python3 -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 10 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --save_dir results/codex_humaneval/{$MODEL_NAME}_temp_0_8 \
    --model $CHECKPOINT_PATH \
    --tokenizer $CHECKPOINT_PATH \
    --use_vllm

# AlpacaEval
python3 -m eval.alpaca_farm.run_eval \
    --model_name_or_path $CHECKPOINT_PATH \
    --save_dir results/alpaca_farm/${MODEL_NAME} \
    --eval_batch_size 20 \
    --max_new_tokens 1024 \
    --use_vllm
