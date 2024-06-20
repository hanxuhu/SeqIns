CHECKPOINT_PATHS=(
    simonycl/self-seq-Meta-Llama-3-8B-flancot_sit_same_instance_output_tokens
    simonycl/self-seq-Meta-Llama-3-8B-flancot_it_same_instance_output_tokens
    simonycl/self-seq-Meta-Llama-3-8B-flancot_sit_same_total_output_tokens
    simonycl/self-seq-Meta-Llama-3-8B-flancot_llama70b_full_it1
    simonycl/self-seq-Meta-Llama-3-8B-sharegpt_seq_llama70b
    simonycl/self-seq-Meta-Llama-3-8B-tulu50k_seq_llama70b
    simonycl/self-seq-Meta-Llama-3-8B-tulu50k_base_llama70b
    simonycl/self-seq-Meta-Llama-3-8B-sharegpt_base_llama70b
    simonycl/self-seq-Meta-Llama-3-8B-sharegpt-llama70b_full_it1
)

echo "Cloning lm-evaluation-harness"
git submodule update --init
cd lm-evaluation-harness
pip install -e .
cd ..

for CHECKPOINT_PATH in ${CHECKPOINT_PATHS[@]}
do
    echo "Running evaluation for $CHECKPOINT_PATH"
    bash scripts/evaluation.sh $CHECKPOINT_PATH
done