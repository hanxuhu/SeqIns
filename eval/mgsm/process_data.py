from datasets import load_dataset
import os
import json

LANGS = ['en', 'es', 'fr', 'de', 'ru', 'zh', 'ja', 'th', 'sw', 'bn', 'te']

def main():
    gsm_dataset = {}
    for lang in LANGS:
        gsm_dataset[lang] = load_dataset('juletxara/mgsm', lang)
    
    os.makedirs('data/eval/mgsm/examples', exist_ok=True)
    os.makedirs('data/eval/mgsm/test', exist_ok=True)
    for lang in LANGS:
        with open(f'data/eval/mgsm/examples/{lang}.jsonl', 'w', encoding='utf-8') as file:
            for i in range(len(gsm_dataset[lang]['train'])):
                file.write(json.dumps(gsm_dataset[lang]['train'][i], ensure_ascii=False))
                file.write('\n')
        with open(f'data/eval/mgsm/test/{lang}.jsonl', 'w', encoding='utf-8') as file:
            for i in range(len(gsm_dataset[lang]['test'])):
                file.write(json.dumps(gsm_dataset[lang]['test'][i], ensure_ascii=False))
                file.write('\n')

if __name__ == '__main__':
    main()