INPUT_FILE=self-seq/data/alpaca/alpaca_cleaned.jsonl

python3 self-seq/gpt-query.py \
    --input_file $INPUT_FILE \
    --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
    --batch_size 4 \
    --use_instruct \
    --ignore_cache \
    --use_vllm \
    --no_refinement