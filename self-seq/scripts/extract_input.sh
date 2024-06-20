
python3 self-seq/extract_input.py \
    --input_file self-seq/data/flancot/final_15k_data_origin.jsonl \
    --sample 250 \
    --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
    --batch_size 1 \
    --use_vllm \
    --use_instruct
