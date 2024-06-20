mkdir -p self-data/data/flancot_full/

wget https://huggingface.co/simonycl/temp_file/resolve/main/sit/flancot/flancot_100k.jsonl -O self-seq/data/flancot_full/flancot_100k.jsonl
INPUT_FILE=self-seq/data/flancot_full/flancot_100k

python3 self-seq/extract_input.py \
    --input_file $INPUT_FILE.jsonl \
    --query meta-llama/Meta-Llama-3-70B-Instruct \
    --output_file $INPUT_FILE-extracted-input.jsonl \
    --batch_size 1 \
    --use_vllm \
    --use_instruct

python3 self-seq/check_input_position.py \
        --input_file $INPUT_FILE-extracted-input-extracted-input.jsonl \
        --output_file $INPUT_FILE-extracted_input.jsonl

python3 self-seq/data/process_multi_it.py \
        --file_path $INPUT_FILE-extracted_input.jsonl \
        --output_file $INPUT_FILE-iter_0.jsonl

python3 self-seq/gpt-query.py \
    --input_file $INPUT_FILE-iter_0-iter.jsonl \
    --query meta-llama/Meta-Llama-3-70B-Instruct \
    --output_file $INPUT_FILE-iteration_0.jsonl \
    --batch_size 4 \
    --use_instruct \
    --ignore_cache \
    --use_vllm \
    --no_refinement

python3 self-seq/data/process_multi_it.py \
        --file_path $INPUT_FILE-iteration_0-generate_instruct-refine-response-final.jsonl \
        --output_file $INPUT_FILE-iteration_1.jsonl

# python3 self-seq/extract_input.py \
#     --input_file $INPUT_FILE \
#     --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
#     --output_file self-seq/data/flancot_extract_refine/final_15k_data_origin.jsonl \
#     --batch_size 1 \
#     --use_vllm \
#     --use_instruct

# python3 self-seq/gpt-query.py \
#     --input_file self-seq/data/flancot_extract_refine/final_15k_data_origin-extracted-input.jsonl \
#     --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
#     --output_file self-seq/data/flancot_extract_refine/flancot_15k_Meta-Llama-3-70B-Instruct_iter_0.jsonl \
#     --batch_size 4 \
#     --use_instruct \
#     --ignore_cache \
#     --use_vllm \
#     --no_refinement \
#     --direct_response 
# 
# python3 self-seq/gpt-query.py \
#     --input_file self-seq/data/flancot_extract_refine/final_15k_data_origin-extracted-input.jsonl \
#     --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
#     --output_file self-seq/data/flancot_extract_refine/flancot_15k_Meta-Llama-3-70B-Instruct_iter_0.jsonl \
#     --batch_size 4 \
#     --use_instruct \
#     --ignore_cache \
#     --use_vllm \
#     --no_refinement

python3 self-seq/data/process_multi_it.py \
	--file_path self-seq/data/flancot_extract_refine/flancot_15k_Meta-Llama-3-70B-Instruct_iter_0-generate_instruct-refine-response-final.jsonl \
	--output_file self-seq/data/flancot_extract_refine/flancot_llama70b_iteration_1.jsonl

# python3 self-seq/gpt-query.py \
#     --input_file self-seq/data/flancot_extract_refine/flancot_llama70b_iteration_1-iter.jsonl \
# 	--query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
#     --output_file self-seq/data/flancot_extract_refine/flancot_15k_Meta-Llama-3-70B-Instruct_iter_1.jsonl \
#     --batch_size 4 \
#     --use_instruct \
#     --use_vllm \
#     --no_refinement \
#     --iteration

python3 self-seq/data/process_multi_it.py \
        --file_path self-seq/data/flancot_extract_refine/flancot_15k_Meta-Llama-3-70B-Instruct_iter_1-generate_instruct-refine-response-final.jsonl \
        --output_file self-seq/data/flancot_extract_refine/flancot_llama70b_iteration_2.jsonl
# 
# python3 self-seq/gpt-query.py \
#     --input_file self-seq/data/flancot_extract/flancot_llama70b_iteration_2-iter.jsonl \
# 	--query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
#     --output_file self-seq/data/flancot_extract/flancot_15k_Meta-Llama-3-70B-Instruct_iter_2.jsonl \
#     --batch_size 4 \
#     --use_instruct \
#     --ignore_cache \
#     --use_vllm \
#     --no_refinement \
#     --iteration
# 
# python3 self-seq/data/process_multi_it.py \
#         --file_path self-seq/data/flancot_extract/flancot_15k_Meta-Llama-3-70B-Instruct_iter_2-generate_instruct-refine-response-final.jsonl \
#         --output_file self-seq/data/flancot_extract/flancot_llama70b_iteration_3.jsonl
