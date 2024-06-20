# export CUDA_VISIBLE_DEVICES=0,1
MODEL=$1

MODEL_SIZE=7B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=2
EVAL_BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=64
MODEL_NAME_OR_PATH=meta-llama/Llama-2-7b-hf


export PYTHONPATH="$PWD:$PYTHONPATH"

MODEL_NAME=$(basename "$MODEL")

mkdir -p eval_results/$MODEL_NAME
# check if exist data/eval/gsm or data/eval/codex_humaneval or data/eval/alpaca_farm
# if not, download the data
if [ ! -d "data/eval/gsm" ] || [ ! -d "data/eval/codex_humaneval" ] || [ ! -d "data/eval/alpaca_farm" ]; then
    bash scripts/prepare_eval_data.sh
fi

# Run evaluation on ARC, GSM8K, HellaSwag, TruthfulQA, and MATH
# bash lm-evaluation-harness/eval_model.sh $MODEL self-seq-$MODEL_NAME > eval_results/$MODEL_NAME/self-seq-$MODEL_NAME.log

# Evaluation script for GSM, MGSM, CodeX-HumanEval, Alpaca-eval
bash scripts/eval/eval_auto_mistral.sh $MODEL self-seq-$MODEL_NAME > eval_results/$MODEL_NAME/self-seq-$MODEL_NAME.log

# Run evaluation on MMLU, ARC, GSM8K
bash lm-evaluation-harness/eval_model.sh $MODEL > eval_results/$MODEL_NAME/self-seq-$MODEL_NAME-hf.log

# Run evaluation on XQuAD
bash eval_seq/infer_eval_self_seq_vllm.sh $MODEL > eval_results/$MODEL_NAME/self-seq-$MODEL_NAME-xquad.log