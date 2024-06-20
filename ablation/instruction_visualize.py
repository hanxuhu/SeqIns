import benepar, spacy
import argparse
import pandas as pd
import json
import tqdm
import concurrent.futures
import time

nlp = spacy.load('en_core_web_md')
if spacy.__version__.startswith('2'):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})
doc = nlp("The time for action is now. It's never too late to do something.")

def find_root_verb_and_its_dobj(tree_root):
    # first check if the current node and its children satisfy the condition
    if tree_root.pos_ == "VERB":
        for child in tree_root.children:
            if child.dep_ == "dobj" and child.pos_ == "NOUN":
                return tree_root.lemma_, child.lemma_
        return tree_root.lemma_, None
    # if not, check its children
    for child in tree_root.children:
        return find_root_verb_and_its_dobj(child)
    # if no children satisfy the condition, return None
    return None, None

def find_root_verb_and_its_dobj_in_string(s):
    doc = nlp(s)
    first_sent = list(doc.sents)[0]
    return find_root_verb_and_its_dobj(first_sent.root)

def split_instruction(instruction):
    # split the sentence everything by ". Then", "and then", "and", ". And"
    for split in [". Then", "and then", "and", ". And", ". "]:
        if split in instruction:
            return instruction.split(split)
    return [instruction]


def process_instruction(instruction):
    if isinstance(instruction, tuple):
        option = instruction[1]
        instruction = instruction[0]

    splited = split_instruction(instruction)
    if option == "B":
        splited = splited[:1]
    elif option == "C":
        splited = splited[1:]
    processed = []
    
    for s in splited:
        try:
            verb, noun = find_root_verb_and_its_dobj_in_string(s)
            processed.append({
                "verb": verb,
                "noun": noun,
                "instruction": s
            })
        except Exception as e:
            pass
    return processed

def main(args):

    generated_data_path = args.input_path
    cache_path = args.cache_path
    if cache_path is None:
        machine_generated_tasks = []
        with open(generated_data_path, "r") as fin:
            for line in fin:
                machine_generated_tasks.append(json.loads(line))
        if args.sample_size:
            machine_generated_tasks = machine_generated_tasks[:args.sample_size]

        machine_generated_tasks = [task for task in machine_generated_tasks if task["option"] != "D"]
        # filter based on instruction
        instructions = list(set([(task["instruction"], task["option"]) for task in machine_generated_tasks]))
        print(len(instructions))

        raw_phrases = []
        if args.use_gpu:
            for instruction in tqdm.tqdm(instructions):
                raw_phrases.extend(process_instruction(instruction))
        else:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = {executor.submit(process_instruction, instruction): instruction for instruction in instructions}
                for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    result = future.result()
                    if result:
                        raw_phrases.extend(result)
        
        raw_phrases = pd.DataFrame(raw_phrases)
        phrases = pd.DataFrame(raw_phrases).dropna()
        # save df as csv
        phrases.to_csv(f"ablation/{args.input_path.split('/')[-1].split('.')[0]}_phrases.csv", index=False)
    else:
        phrases = pd.read_csv(cache_path)
    top_verbs = phrases[["verb"]].groupby(["verb"]).size().nlargest(20).reset_index()

    df = phrases[phrases["verb"].isin(top_verbs["verb"].tolist())]
    # df = df[~df["noun"].isin(["I", "what"])]
    # df = phrases
    # df[~df["verb"].isin(top_verbs["verb"].tolist())]["verb"] = "other"
    # df[~df["verb"].isin(top_verbs["verb"].tolist())]["noun"] = "other"
    df = df.groupby(["verb", "noun"]).size().reset_index().rename(columns={0: "count"}).sort_values(by=["count"], ascending=False)
    # df = df[df["count"] > 10]
    df = df.groupby("verb").apply(lambda x: x.sort_values("count", ascending=False).head(4)).reset_index(drop=True)
    
    import plotly.graph_objects as go
    import plotly.express as px

    # df["blank"] = "ROOT"
    # df = phrases.groupby(["verb", "noun"]).size().sort_values(ascending=False).head(5).reset_index().rename(columns={0: "count"})

    df = df[df["count"] > args.filter]
    fig = px.sunburst(df, path=['verb', 'noun'], values='count')
    # fig.update_layout(uniformtext=dict(minsize=10, mode='hide'))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        font_family="Times New Roman",
    )
    # save the plot
    fig.write_html(f"ablation/{args.input_path.split('/')[-1].split('.')[0]}_sunburst.html")
    fig.write_image(f"ablation/{args.input_path.split('/')[-1].split('.')[0]}_sunburst.pdf", format="pdf")
    time.sleep(2)

    fig.write_image(f"ablation/{args.input_path.split('/')[-1].split('.')[0]}_sunburst.pdf", format="pdf")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_path", type=str, default="self-seq/data/alpaca_final/alpaca_final.jsonl")
    args.add_argument("--sample_size", type=int, default=None)
    args.add_argument("--filter", type=int, default=100)
    args.add_argument("--use_gpu", type=bool, default=False)
    args.add_argument("--cache_path", type=str, default=None)
    args = args.parse_args()

    main(args)