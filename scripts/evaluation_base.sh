# export CUDA_VISIBLE_DEVICES=0,1
MODEL=$1

HF_TOKEN=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

MODEL_SIZE=7B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=2
EVAL_BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=64
MODEL_NAME_OR_PATH=meta-llama/Llama-2-7b-hf


export PYTHONPATH="$PWD:$PYTHONPATH"

# for MODEL in simonycl/sparseIT-Llama-2-7b-hf-multi-task simonycl/data_selection_Llama-2-7b-hf-multi_task-mask-mlp-by-dataset
# for MODEL in simonycl/sparseIT-Llama-2-7b-hf-multi-task
# for MODEL in simonycl/sparseIT_Llama-2-7b-hf-stanford-alpaca
# for MODEL in /mnt/data/sparseIT/output/sparseIT_Llama-2-7b-hf-stanford-alpaca-mask-by-cluster
# for MODEL in /mnt/data/sparseIT/output/sparseIT_Llama-2-7b-hf-multi-task-data-no-mlp
# for MODEL in simonycl/self-seq-7b-1-3-new
# for MODEL in meta-llama/Llama-2-7b-hf
# for MODEL in simonycl/self-seq-7b-baseline
# do

    # split by '/' and select the last two elements and join them with '-'
    # MODEL_NAME=$(echo $MODEL | tr '/' '-' | cut -d '-' -f 2-)

MODEL_NAME=$(basename "$MODEL")

mkdir -p eval_results/$MODEL_NAME

# Run evaluation on ARC, GSM8K, HellaSwag, TruthfulQA, and MATH
# bash lm-evaluation-harness/eval_model.sh $MODEL self-seq-$MODEL_NAME > eval_results/$MODEL_NAME/self-seq-$MODEL_NAME.log

# Evaluation script for MMLU, TydiQA and CodeX-HumanEval
bash scripts/eval/eval_auto_tulu_base.sh $MODEL self-seq-$MODEL_NAME > eval_results/$MODEL_NAME/self-seq-$MODEL_NAME-alpaca.log

mkdir -p data/$MODEL_NAME

# bash eval_seq/inference_lima.sh $MODEL data/$MODEL_NAME/self-seq-$MODEL_NAME.jsonl > eval_results/$MODEL_NAME/inference_lima-$MODEL_NAME.log

# python3 eval_seq/eval_rouge.py \
#     --test_file data/$MODEL_NAME/self-seq-$MODEL_NAME.jsonl \
#     --ref_file self-seq/data/lima_500-replaced.jsonl > eval_results/$MODEL_NAME/rouge-$MODEL_NAME.log
# done

# nohup bash scripts/evaluation_lora.sh > ./logs/evaluation_lora_Llama-2-7b-hf-sharegpt-KMenasRandomDeita-64-005-lora-epoch_4.log 2>&1 &

# nohup bash scripts/evaluation_lora.sh > ./logs/evaluation_lora_Llama-2-7b-hf-sharegpt-KCenterGreedyDeita-005-lora-epoch_4.log 2>&1 &
