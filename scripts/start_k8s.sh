export VLLM_VERSION=0.2.6
export PYTHON_VERSION=38

pip3 install accelerate deepspeed peft bitsandbytes tokenizers evaluate

pip install packaging
pip install ninja

pip install wandb

pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl

pip uninstall -y torch torchvision torchaudio

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

pip3 install -U flash-attn

pip uninstall -y xformers

pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118

pip uninstall antlr4-python3-runtime -y

pip install antlr4-python3-runtime==4.11

pip install -U openai

pip install rouge bert_score

export HF_HOME=/mnt/data/.cache/huggingface
export HF_TRANSFORMERS_CACHE=/mnt/data/.cache/transformers
export HF_DATASETS_CACHE=/mnt/data/.cache/datasets

git config --global credential.helper store

wget https://huggingface.co/simonycl/temp_file/resolve/main/self-seq/alpaca-cleaned_replaced.jsonl -P self-seq/data