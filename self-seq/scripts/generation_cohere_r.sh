python3 self-seq/gpt-query.py \
    --input_file self-seq/data/flancot/flancot_filtered_15k.jsonl \
    --sample 100 \
    --query /mnt/nfs/public/hf/models/CohereForAI/c4ai-command-r-v01 \
    --batch_size 8 \
    --use_instruct \
    --add_system_prompt


python3 self-seq/gpt-query.py \
    --input_file self-seq/data/alpaca/alpaca-cleaned.jsonl \
    --sample 100 \
    --query /mnt/nfs/public/hf/models/CohereForAI/c4ai-command-r-v01 \
    --batch_size 8 \
    --use_instruct