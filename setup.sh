pip install -r requirements.txt

# Uninstall flash-attn if already installed
pip uninstall flash-attn -y

# Install flash-attn separately to ensure it is installed after vllm
pip install -U flash-attn --no-cache-dir