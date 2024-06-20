INPUT_FILE=self-seq/data/flancot/final_15k_data_origin.jsonl

python3 self-seq/gpt-query.py \
    --input_file $INPUT_FILE \
    --query /mnt/data/models/c4ai-command-r-plus-GPTQ \
    --batch_size 4 \
    --use_instruct \
    --direct_response \
    --use_vllm \
    --no_refinement