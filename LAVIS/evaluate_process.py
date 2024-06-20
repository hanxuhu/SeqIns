#description: This file is used to evaluate the accuracy of the model on the VQA dataset.
import sys
import re
import json
import random
import argparse
contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
}
manualMap = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
}
articles = ["a", "an", "the"]

periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile("(\d)(,)(\d)")
punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
]

def evaluate(jsonl_path):
        # if quesIds == None:
        #     quesIds = [quesId for quesId in self.params["question_id"]]
        # gts = {}
        # res = {}
        # for quesId in quesIds:
        #     gts[quesId] = self.vqa.qa[quesId]
        #     res[quesId] = self.vqaRes.qa[quesId]

        # =================================================
        # Compute accuracy
        # =================================================
        # open jsonl path and read the file:
        with open(jsonl_path, 'r') as f:
            data = f.readlines()
        # accQA = []
        # accQuesType = {}
        # accAnsType = {}
        print("computing accuracy")
        #step = 0
        gtAcc = []
        random.shuffle(data)
        
        print(len(data))
        for item in data:
            item = json.loads(item)
            #print(item[0])
            resAns = item["text"].lower()
            if len(resAns.lower().split('answer'))>1:
               text = resAns.lower().split('answer')[1].strip()
            else:
               text = 'none'
            resAns = text
            #resAns = resAns.split('answer')[-1]
            #print(resAns)
            resAns = resAns.replace("\n", " ")
            resAns = resAns.replace("\t", " ")
            resAns = resAns.strip()
            resAns = resAns.split(" ")[-1]
            resAns = processPunctuation(resAns)
            resAns = processDigitArticle(resAns)
            
            Label = item["label"]
            Label = processPunctuation(Label)
            Label = processDigitArticle(Label)
            if resAns == "" or resAns == " ":
               resAns = "NULL"
           
            if resAns == Label or resAns in Label or Label in resAns:
                gtAcc.append(1)
            else:
                gtAcc.append(0)
        Acc = sum(gtAcc) / len(gtAcc)
        print("Acc", Acc)


def processPunctuation(inText):
        outText = inText
        for p in punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

def processDigitArticle(inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = manualMap.setdefault(word, word)
            if word not in articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in contractions:
                outText[wordId] = contractions[word]
        outText = " ".join(outText)
        return outText

parser = argparse.ArgumentParser(description="Evaluation")

parser.add_argument("--datapath", required=True, help="path to configuration file.")

args = parser.parse_args()

evaluate(args.datapath)

# def setAccuracy(self, accQA, accQuesType, accAnsType):
#         self.accuracy["overall"] = round(100 * float(sum(accQA)) / len(accQA), self.n)
#         self.accuracy["perQuestionType"] = {
#             quesType: round(
#                 100 * float(sum(accQuesType[quesType])) / len(accQuesType[quesType]),
#                 self.n,
#             )
#             for quesType in accQuesType
#         }
#         self.accuracy["perAnswerType"] = {
#             ansType: round(
#                 100 * float(sum(accAnsType[ansType])) / len(accAnsType[ansType]), self.n
#             )
#             for ansType in accAnsType
#         }

    # def setEvalQA(self, quesId, acc):
    #     self.evalQA[quesId] = round(100 * acc, self.n)

    # def setEvalQuesType(self, quesId, quesType, acc):
    #     if quesType not in self.evalQuesType:
    #         self.evalQuesType[quesType] = {}
    #     self.evalQuesType[quesType][quesId] = round(100 * acc, self.n)

    # def setEvalAnsType(self, quesId, ansType, acc):
    #     if ansType not in self.evalAnsType:
    #         self.evalAnsType[ansType] = {}
    #     self.evalAnsType[ansType][quesId] = round(100 * acc, self.n)

    # def updateProgress(self, progress):
    #     barLength = 20
    #     status = ""
    #     if isinstance(progress, int):
    #         progress = float(progress)
    #     if not isinstance(progress, float):
    #         progress = 0
    #         status = "error: progress var must be float\r\n"
    #     if progress < 0:
    #         progress = 0
    #         status = "Halt...\r\n"
    #     if progress >= 1:
    #         progress = 1
    #         status = "Done...\r\n"
    #     block = int(round(barLength * progress))
    #     text = "\rFinshed Percent: [{0}] {1}% {2}".format(
    #         "#" * block + "-" * (barLength - block), int(progress * 100), status
    #     )
    #     sys.stdout.write(text)
    #     sys.stdout.flush()
