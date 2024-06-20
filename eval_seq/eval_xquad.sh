#!/bin/bash
TYPES=(fewshot fewshot_en fewshot_multi)
FILES=(self-seq-7b-1-3-new self-seq-7B-baseline self-seq-alpaca-cleaned_repeat self-seq-alpaca-cleaned_wizardlm_replaced self-seq-wizardlm)
FILES=(Llama-2-7b-hf)
FILES=(self-seq-Mistral-7B-v0.1-alpaca_5lang self-seq-Mistral-7B-v0.1-alpaca_it self-seq-Mistral-7B-v0.1-alpaca_mistral_it self-seq-Meta-Llama-3-8B-flancot_full_sit_llama_70b_iter_2)
FILES=(self-seq-Meta-Llama-3-8B-flancot_llama_70b)
FILES=(self-seq-Mistral-7B-v0.1-alpaca_it self-seq-Mistral-7B-v0.1-alpaca_sit self-seq-Meta-Llama-3-8B-alpaca_it)
FILES=(self-seq-Meta-Llama-3-8B-alpaca_it_llmam_70b)
FILES=(self-seq-Mistral-7B-v0.1-alpaca_it_gen self-seq-Mistral-7B-v0.1-alpaca_sit_gen)
# FILES=(self-seq-Meta-Llama-3-8B-alpaca_llmam_70b-iter-2 self-seq-Meta-Llama-3-8B-flancot_full_it_llama_70b)
# FILES=(self-seq-Meta-Llama-3-8B-flancot_full_sit_llama_70b_iter_2)
FILES=(self-seq-Meta-Llama-3-8B-alpaca_llmam_70b-iter-2 self-seq-Meta-Llama-3-8B-flancot_full_sit_llama_70b_iter_2 self-seq-Meta-Llama-3-8B-flancot_full_it_llama_70b)
FILES=(self-seq-Meta-Llama-3-8B-alpaca_it self-seq-Meta-Llama-3-8B-alpaca_lang5 self-seq-Meta-Llama-3-8B-alpaca_llmam_70b-iter-2)
FILES=(self-seq-Meta-Llama-3-8B-wizardlm)
FILES=(self-seq-Mistral-7B-v0.1-alpaca_sit_gen)
FILES=(self-seq-Meta-Llama-3-8B-sit-alpaca_rplus self-seq-Meta-Llama-3-8B-alpaca_rplus_it)
FILES=(self-seq-Mistral-7B-v0.1-alpaca_it_gen self-seq-Mistral-7B-v0.1-alpaca_sit_gen)

for FILE in ${FILES[@]}
do
    for LANG in de es ru zh
    do
        echo "Evaluating ${FILE} on XQuAD ${LANG}"
        python3 eval_seq/eval_xquad.py \
            --test_file eval_results/${FILE}/${FILE}_base_xquad_${LANG}.jsonl \
            --ref_file data/xquad/fewshot/xquad_en.jsonl
            
    done
    for LANG in ar hi th tr vi el
    do
        echo "Evaluating ${FILE} on XQuAD ${LANG}"
        python3 eval_seq/eval_xquad.py \
            --test_file eval_results/${FILE}/${FILE}_trans_xquad_${LANG}.jsonl \
            --ref_file data/xquad/fewshot_multi/xquad_en.jsonl
            
    done
done