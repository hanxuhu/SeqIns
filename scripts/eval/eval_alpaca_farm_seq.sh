
for CHECKPOINT_PATH in simonycl/self-seq-Meta-Llama-3-8B-alpaca_it_llmam_70b simonycl/self-seq-Meta-Llama-3-8B-alpaca_llmam_70b_iter simonycl/self-seq-Meta-Llama-3-8B-alpaca_llmam_70b-iter-2 simonycl/self-seq-Meta-Llama-3-8B-wizardlm simonycl/self-seq-Meta-Llama-3-8B-flancot_full_it_llama_70b simonycl/self-seq-Meta-Llama-3-8B-flancot_full_sit_llama_70b simonycl/self-seq-Meta-Llama-3-8B-flancot_full_sit_llama_70b_iter_2 simonycl/self-seq-Meta-Llama-3-8B-alpaca_sit_llama_70b-iter3
do
    MODEL_NAME=$(basename $CHECKPOINT_PATH)

    # check if sequential_instruction_tuning/SeqAlpacaEval/alpaca_eval_gpt4_baseline.seq.json exists
    if [ -f "sequential_instruction_tuning/SeqAlpacaEval/alpaca_eval_gpt4_baseline.seq.json" ]; then
        echo "AlpacaEval file exists"
    else
        git clone https://github.com/PinzhenChen/sequential_instruction_tuning.git
    fi

    python3 -m eval.alpaca_farm.run_seq_eval \
        --model_name_or_path $CHECKPOINT_PATH \
        --save_dir results/alpaca_farm/seqEval/${MODEL_NAME} \
        --eval_batch_size 20 \
        --max_new_tokens 2048 \
        --use_vllm \
        --prompt_path sequential_instruction_tuning/SeqAlpacaEval/seqEval.json \
        --use_chat_format \
        --chat_formatting_function tulu

    # python3 -m eval.alpaca_farm.run_seq_eval \
    #     --model_name_or_path $CHECKPOINT_PATH \
    #     --save_dir results/alpaca_farm/seq-1/${MODEL_NAME} \
    #     --eval_batch_size 20 \
    #     --max_new_tokens 2048 \
    #     --use_vllm \
    #     --prompt_path sequential_instruction_tuning/SeqAlpacaEval/alpaca_eval_gpt4_baseline.seq.json \
    #     --use_chat_format \
    #     --chat_formatting_function tulu

    # python3 -m eval.alpaca_farm.run_seq_eval \
    #     --model_name_or_path $CHECKPOINT_PATH \
    #     --save_dir results/alpaca_farm/seq-2/${MODEL_NAME} \
    #     --eval_batch_size 20 \
    #     --max_new_tokens 2048 \
    #     --use_vllm \
    #     --prompt_path sequential_instruction_tuning/SeqAlpacaEval/alpaca_eval_gpt4_baseline.seq.seq.json \
    #     --use_chat_format \
    #     --chat_formatting_function tulu

    # python3 -m eval.alpaca_farm.run_seq_eval \
    #     --model_name_or_path $CHECKPOINT_PATH \
    #     --save_dir results/alpaca_farm/seq-3/${MODEL_NAME} \
    #     --eval_batch_size 20 \
    #     --max_new_tokens 2048 \
    #     --use_vllm \
    #     --prompt_path sequential_instruction_tuning/SeqAlpacaEval/alpaca_eval_gpt4_baseline.seq.seq.seq.json \
    #     --use_chat_format \
    #     --chat_formatting_function tulu

    # python3 -m eval.alpaca_farm.run_seq_eval \
    #     --model_name_or_path $CHECKPOINT_PATH \
    #     --save_dir results/alpaca_farm/seq-4/${MODEL_NAME} \
    #     --eval_batch_size 20 \
    #     --max_new_tokens 2048 \
    #     --use_vllm \
    #     --prompt_path sequential_instruction_tuning/SeqAlpacaEval/alpaca_eval_gpt4_baseline.seq.seq.seq.seq.json \
    #     --use_chat_format \
    #     --chat_formatting_function tulu
done