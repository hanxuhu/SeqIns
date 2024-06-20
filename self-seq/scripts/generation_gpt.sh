
python3 self-seq/gpt-query.py \
    --input_file self-seq/data/flancot/flancot_filtered_15k.jsonl \
    --sample 100 \
    --query gpt-3.5-turbo \
    --add_system_prompt


python3 self-seq/gpt-query.py \
    --input_file self-seq/data/alpaca/alpaca-cleaned.jsonl \
    --sample 100 \
    --query gpt-3.5-turbo