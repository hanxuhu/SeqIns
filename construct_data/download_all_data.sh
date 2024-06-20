mkdir -p ../data/test
mkdir -p ../data/alpaca
# for LANG in en es de zh ru el ar vi hi tr th
# do
#   wget -O ../../data/alpaca/alpaca_data_cleaned.${LANG}.json https://raw.githubusercontent.com/hplt-project/monolingual-multilingual-instruction-tuning/main/training-data/alpaca_data_cleaned.${LANG}.json
# done
# mkdir ../../data/test
# python make_data_all.py --dataset commonsense_qa --typename base
# python make_data_all.py --dataset commonsense_qa --typename repeat
# python3 make_data_all.py --dataset xquad --target en --typename base

for LANG in es de zh ru
do
  python3 make_data_all.py --dataset xquad --target ${LANG} --typename base
done

for LANG in el ar el ar vi hi tr th
do 
  python3 make_data_all.py --dataset xquad --target ${LANG} --typename trans
done

# python make_data_alpaca_paraphrase1.py

# python3 make_data_alpaca_merge_multilingual.py
