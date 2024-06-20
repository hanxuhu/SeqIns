INPUT_FILE=self-seq/data/flancot_full/flancot_100k_split3

# python3 self-seq/extract_input.py \
#     --input_file $INPUT_FILE.jsonl \
#     --query /mnt/data/models/c4ai-command-r-plus-GPTQ \
#     --output_file $INPUT_FILE-extracted-input.jsonl \
#     --batch_size 1 \
#     --use_vllm \
#     --use_instruct
# 
# python3 self-seq/gpt-query.py \
#     --input_file $INPUT_FILE-extracted-input.jsonl \
#     --query /mnt/data/models/c4ai-command-r-plus-GPTQ \
#     --output_file $INPUT_FILE-iteration_0.jsonl \
#     --batch_size 4 \
#     --use_instruct \
#     --ignore_cache \
#     --use_vllm \
#     --no_refinement
# 
# python3 self-seq/data/process_multi_it.py \
# 	--file_path $INPUT_FILE-iteration_0-generate_instruct-refine-response-final.jsonl \
# 	--output_file $INPUT_FILE-iteration_1.jsonl

python3 self-seq/gpt-query.py \
    --input_file $INPUT_FILE.jsonl \
    --query /mnt/data/models/c4ai-command-r-plus-GPTQ \
    --output_file $INPUT_FILE-direct_response.jsonl \
    --batch_size 4 \
    --use_instruct \
    --direct_response \
    --use_vllm \
    --no_refinement

python3 self-seq/data/process_multi_it.py \
	--file_path $INPUT_FILE-direct_response-direct_response.jsonl \
	--output_file $INPUT_FILE-direct_response.jsonl

# python3 self-seq/gpt-query.py \
#     --input_file $INPUT_FILE-iteration_1-iter.jsonl \
# 	--query /mnt/data/models/c4ai-command-r-plus-GPTQ \
#     --output_file $INPUT_FILE-iteration_2.jsonl \
#     --batch_size 4 \
#     --use_instruct \
#     --ignore_cache \
#     --use_vllm \
#     --no_refinement \
#     --iteration
# 
# python3 self-seq/data/process_multi_it.py \
#         --file_path $INPUT_FILE-iteration_2-generate_instruct-refine-response-final.jsonl \
#         --output_file $INPUT_FILE-iteration_3.jsonl
