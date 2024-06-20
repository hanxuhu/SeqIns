INPUT_FILE=self-seq/data/flancot_full/flancot100k

python3 self-seq/extract_input.py \
    --input_file $INPUT_FILE.jsonl \
    --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
    --output_file $INPUT_FILE-extracted-input.jsonl \
    --batch_size 1 \
    --use_vllm \
    --use_instruct

python3 self-seq/gpt-query.py \
    --input_file $INPUT_FILE-extracted-input.jsonl \
    --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
    --output_file $INPUT_FILE-iteration_0.jsonl \
    --batch_size 4 \
    --use_instruct \
    --ignore_cache \
    --use_vllm \
    --no_refinement

python3 self-seq/data/process_multi_it.py \
	--file_path $INPUT_FILE-iteration_0-generate_instruct-refine-response-final.jsonl \
	--output_file $INPUT_FILE-iteration_1.jsonl

python3 self-seq/gpt-query.py \
    --input_file $INPUT_FILE-iteration_0-iter.jsonl \
    --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
    --output_file $INPUT_FILE-direct_response.jsonl \
    --batch_size 4 \
    --use_instruct \
    --direct_response \
    --use_vllm \
    --no_refinement \
    --iteration


python3 self-seq/data/process_multi_it.py \
	--file_path $INPUT_FILE-direct_response-generate_instruct-refine-response-final.jsonl \
	--output_file $INPUT_FILE-direct_response.jsonl

python3 self-seq/gpt-query.py \
    --input_file $INPUT_FILE-iteration_1-iter.jsonl \
	--query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
    --output_file $INPUT_FILE-iteration_2.jsonl \
    --batch_size 4 \
    --use_instruct \
    --ignore_cache \
    --use_vllm \
    --no_refinement \
    --iteration

python3 self-seq/data/process_multi_it.py \
        --file_path $INPUT_FILE-iteration_2-generate_instruct-refine-response-final.jsonl \
        --output_file $INPUT_FILE-iteration_3.jsonl

python3 self-seq/gpt-query.py \
    --input_file $INPUT_FILE-iteration_3-iter.jsonl \
	--query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
    --output_file $INPUT_FILE-iteration_4.jsonl \
    --batch_size 4 \
    --use_instruct \
    --ignore_cache \
    --use_vllm \
    --no_refinement \
    --iteration

python3 self-seq/data/process_multi_it.py \
        --file_path $INPUT_FILE-iteration_4-generate_instruct-refine-response-final.jsonl \
        --output_file $INPUT_FILE-iteration_5.jsonl
