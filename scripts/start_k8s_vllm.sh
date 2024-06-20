pip3 install accelerate deepspeed peft bitsandbytes tokenizers evaluate

pip install packaging
pip install ninja

pip install wandb

cd ..
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .

pip uninstall flash-attn -y

pip3 install -U flash-attn --no-cache-dir

pip uninstall antlr4-python3-runtime -y

pip install antlr4-python3-runtime==4.11

pip install -U openai

pip install cohere

pip install rouge bert_score

export HF_HOME=/mnt/data/.cache/huggingface
export HF_TRANSFORMERS_CACHE=/mnt/data/.cache/transformers
export HF_DATASETS_CACHE=/mnt/data/.cache/datasets

git config --global credential.helper store

# wget https://huggingface.co/simonycl/temp_file/resolve/main/self-seq/alpaca-cleaned_replaced.jsonl -P self-seq/data