export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
TRAIN_FILE=self-seq/data/alpaca_final/alpaca_15k_llama70b_iteration_4.jsonl
TRAIN_FILE=self-seq/data/alpaca/data_sit_same_instance_output_tokens.jsonl
TRAIN_FILE=ablation/iter/alpaca_llama70b_iteration_2-separated.jsonl
# TRAIN_FILE=self-seq/data/alpaca/alpaca_llama_70b_iter_2.jsonl
# TRAIN_FILE=self-seq/data/alpaca/alpaca_llama70b_iteration_1.jsonl
# check if TRAIN_FILE exists
# if [ ! -f "$TRAIN_FILE" ]; then
#     wget https://huggingface.co/simonycl/temp_file/resolve/main/sit/alpaca_llmam_70b.jsonl -O $TRAIN_FILE
# fi

MODEL_NAME_OR_PATH=/mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-8B
MODEL_NAME=$(basename $MODEL_NAME_OR_PATH)

GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory, 
# but it will trade off speed.
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    self-seq/finetune.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --use_flash_attn \
    --tokenizer_name $MODEL_NAME_OR_PATH \
    --use_slow_tokenizer \
    --train_file $TRAIN_FILE \
    --max_seq_length 2048 \
    --preprocessing_num_workers 64 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --gradient_checkpointing \
    --output_dir output/self-seq-${MODEL_NAME}-alpaca_sit_separated/ \
    --prompt_template tulu \
    --with_tracking \
    --do_eval \
    --eval_steps 100 \
    --eval_file self-seq/data/lima500_withsys.jsonl \
    --report_to wandb \
    --logging_steps 5

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
bash scripts/evaluation.sh output/self-seq-${MODEL_NAME}-alpaca_sit_separated > logs/eval.log
