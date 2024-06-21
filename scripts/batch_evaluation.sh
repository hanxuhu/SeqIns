CHECKPOINT_PATHS=(
    # Insert paths to checkpoints here
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