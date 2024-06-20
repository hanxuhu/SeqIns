for INPUT in self-seq-Llama-2-7b-hf-new self-seq-7b-baseline self-seq-7b-1-3-new self-seq-combined-Llama-2-7b-epoch-1 self-seq-alpaca-replaced-wizardlm
do
    for REF in gpt-3.5-turbo-long gpt-3.5-turbo-short
    do
        python3 eval/alpaca_farm/reward.py \
            --input_file results/alpaca_farm/${INPUT}/${INPUT}-greedy-long-output.json \
            --ref_file results/alpaca_farm/${REF}/alpaca_eval_reward.json
    done
done
