INPUT_FILE=self-seq/data/alpaca_final/alpaca_final_1

# python3 self-seq/extract_input.py \
#     --input_file $INPUT_FILE.jsonl \
#     --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
#     --output_file $INPUT_FILE-extracted-input.jsonl \
#     --batch_size 1 \
#     --use_vllm \
#     --use_instruct

python3 self-seq/gpt-query.py \
    --input_file $INPUT_FILE.jsonl \
    --query /mnt/data/models/c4ai-command-r-plus-GPTQ \
    --output_file $INPUT_FILE-r_plus_iteration_0.jsonl \
    --batch_size 4 \
    --use_instruct \
    --ignore_cache \
    --use_vllm \
    --no_refinement \
    --temperature 1.0 \
    --top_p 0.9 \
    --max_new_tokens 2048

python3 self-seq/data/process_multi_it.py \
	--file_path $INPUT_FILE-r_plus_iteration_0-generate_instruct-refine-response-final.jsonl \
	--output_file $INPUT_FILE-r_plus_iteration_1.jsonl

python3 self-seq/gpt-query.py \
    --input_file $INPUT_FILE.jsonl \
    --query /mnt/data/models/c4ai-command-r-plus-GPTQ \
    --output_file $INPUT_FILE-r_plus_direct_response.jsonl \
    --batch_size 4 \
    --use_instruct \
    --direct_response \
    --use_vllm \
    --no_refinement \
    --iteration \
    --temperature 1.0 \
    --top_p 0.9 \
    --max_new_tokens 2048


python3 self-seq/data/process_multi_it.py \
	--file_path $INPUT_FILE-r_plus_direct_response-generate_instruct-refine-response-final.jsonl \
	--output_file $INPUT_FILE-r_plus_direct_response.jsonl

python3 self-seq/gpt-query.py \
    --input_file $INPUT_FILE-r_plus_iteration_1-iter.jsonl \
	--query /mnt/data/models/c4ai-command-r-plus-GPTQ \
    --output_file $INPUT_FILE-r_plus_iteration_2.jsonl \
    --batch_size 4 \
    --use_instruct \
    --ignore_cache \
    --use_vllm \
    --no_refinement \
    --iteration \
    --temperature 1.0 \
    --top_p 0.9 \
    --max_new_tokens 2048

python3 self-seq/data/process_multi_it.py \
        --file_path $INPUT_FILE-r_plus_iteration_2-generate_instruct-refine-response-final.jsonl \
        --output_file $INPUT_FILE-r_plus_iteration_2.jsonl

python3 self-seq/gpt-query.py \
    --input_file $INPUT_FILE-iteration_2-iter.jsonl \
	--query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
    --output_file $INPUT_FILE-iteration_3.jsonl \
    --batch_size 4 \
    --use_instruct \
    --ignore_cache \
    --use_vllm \
    --no_refinement \
    --iteration \
    --temperature 1.0 \
    --top_p 0.9 \
    --max_new_tokens 2048

python3 self-seq/data/process_multi_it.py \
        --file_path $INPUT_FILE-iteration_3-generate_instruct-refine-response-final.jsonl \
        --output_file $INPUT_FILE-iteration_3.jsonl
