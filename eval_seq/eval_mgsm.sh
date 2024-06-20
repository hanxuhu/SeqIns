#!/bin/bash
FILES=(self-seq-Meta-Llama-3-8B-alpaca_it_llmam_70b-trans-cot-0shot \
self-seq-Meta-Llama-3-8B-alpaca_it_llmam_70b-trans-cot-8shot \
self-seq-Meta-Llama-3-8B-alpaca_llmam_70b_iter-trans-cot-0shot)
FILES=(self-seq-Meta-Llama-3-8B-alpaca_llmam_70b_iter-trans-cot-8shot)
FILES=(self-seq-Meta-Llama-3-8B-alpaca_llmam_70b-iter-2-trans-cot-0shot \
self-seq-Meta-Llama-3-8B-alpaca_llmam_70b-iter-2-trans-cot-8shot \
self-seq-Meta-Llama-3-8B-flancot_full_it_llama_70b-trans-cot-0shot \
self-seq-Meta-Llama-3-8B-flancot_full_it_llama_70b-trans-cot-8shot \
self-seq-Meta-Llama-3-8B-flancot_full_sit_llama_70b_iter_2-trans-cot-0shot \
self-seq-Meta-Llama-3-8B-flancot_full_sit_llama_70b_iter_2-trans-cot-8shot \
self-seq-Meta-Llama-3-8B-wizardlm-trans-cot-0shot \
self-seq-Meta-Llama-3-8B-wizardlm-trans-cot-8shot)
# FILES=(self-seq-Meta-Llama-3-8B-wizardlm-trans-cot-0shot)
FILES=(self-seq-Meta-Llama-3-8B-alpaca_llmam_70b_iter-trans-cot-0shot \
self-seq-Meta-Llama-3-8B-alpaca_llmam_70b-iter-2-trans-cot-0shot \
self-seq-Meta-Llama-3-8B-flancot_full_it_llama_70b-trans-cot-0shot \
self-seq-Meta-Llama-3-8B-flancot_full_sit_llama_70b_iter_2-trans-cot-0shot \
self-seq-Meta-Llama-3-8B-wizardlm-trans-cot-0shot)

for FILE in ${FILES[@]}
do
    for LANG in bn de es fr ja ru sw te th zh
    # for LANG in zh
    do
        echo "Evaluating ${FILE} on XQuAD ${LANG}"
        python3 eval_seq/eval_mgsm.py \
            --test_file results/mgsm/${FILE}/predictions_${LANG}.jsonl \
            --ref_file data/eval/mgsm/test/en.jsonl
    done
done