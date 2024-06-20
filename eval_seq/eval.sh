python3 eval_seq/eval_rouge.py \
    --test_file data/alpaca-llama-2-7b-hf-baseline/seq-it-alpaca-llama-2-7b-hf-baseline.jsonl \
    --ref_file self-seq/data/lima_500-replaced.jsonl

python3 eval_seq/eval_rouge.py \
    --test_file data/alpaca-llama-2-7b-hf-new/seq-it-alpaca-llama-2-7b-hf-new.jsonl \
    --ref_file self-seq/data/lima_500-replaced.jsonl

python3 eval_seq/eval_rouge.py \
    --test_file data/llama-2-7b-hf/seq-it-Llama-2-7b-hf.jsonl \
    --ref_file self-seq/data/lima_500-replaced.jsonl