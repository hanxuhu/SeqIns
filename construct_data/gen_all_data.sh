mkdir -p ../data/test
mkdir -p ../data/alpaca

for TYPE in fewshot fewshot_en fewshot_multi
do
  for LANG in en es ar el hi th de ru zh tr vi
  do
    echo "Making ${LANG} ${TYPE} examples"
    python3 construct_data/make_few_shot_examples.py --dataset xquad --target ${LANG} --typename ${TYPE}
  done
done

# self-seq-7B-baseline self-seq-wizardlm