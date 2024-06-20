# import pandas
from datasets import load_dataset
import random

SYSTEM_PROMPT = "You are an assistant to classify the correct choices for the given instruction. Please follow strictly the format of the few shot examples given and provide explanation."
PROMPT_PREFIX = """Given the original instruction, you should propose a new instruction based on it by doing one of following things:
A. Decompose it into two tasks.
B. Add a prefix task.
C. Add a suffix task.
D. Keep as original version. (Choose this if the original instruction is already sufficient)
You should decide which option is suitable for the input instruction, following the few shot examples and end your answer with “###”.
"""
PROMPT_PREFIX_CHAT = """Given the original instruction, you should propose a new instruction based on it by doing one of following things:
A. Decompose it into two tasks.
B. Add a prefix task.
C. Add a suffix task.
D. Keep as original version. (Choose this if the original instruction is already sufficient)
You should decide which option is suitable for the input instruction.
"""

FEW_SHOTS_EXAMPLE = [
    """The instruction is: Describe the structure of an atom.

Let's think step by step. For the given instruction, a suitable adaptation is to add a suffix task. This would deepen the user's understanding by applying the knowledge in a practical context. New instruction: "Describe the structure of an atom and explain how this structure determines its chemical properties."
This modification (Option C) not only covers the original request to describe the atom's structure but also extends the learning by connecting atomic structure to chemical properties, making the explanation more comprehensive and applicable.
So the option is: C.###""",

    """The instruction is: 'James runs a TV show and there are 5 main characters and 4 minor characters. He pays the minor characters $15,000 each episode. He paid the major characters three times as much. How much does he pay per episode? Let's be accurate as possible.'

Let's think step by step. For the given instruction, a suitable adaptation is to keep as original version. The instruction is already a well-formed, clear mathematical problem that conveys all the necessary information needed to solve it. It integrates a real-world context with a math problem, which can engage users by connecting calculations to practical scenarios.
So the option is: D.###""",

"""The instruction is: 'What happens next in this paragraph? She then rubs a needle on a cotton ball then pushing it onto a pencil and wrapping thread around it. She then holds up a box of a product and then pouring several liquids into a bowl. Choose your answer from: A. adds saucepan and shakes up the product in a grinder. B. pinches the thread to style a cigarette, and then walks away. C. then dips the needle in ink and using the pencil to draw a design on her leg, rubbing it off with a rag in the end. D. begins to style her hair and cuts it several times before parting the ends of it to show the hairstyle she has created.'

Let's think step by step. For the given instruction, a suitable adaptation is to decompose it into two tasks. The existing instruction involves understanding the sequence described and choosing an appropriate continuation. This can be decomposed into first identifying and explaining each action in the sequence, and then predicting the next action based on logical continuation or typical usage of the items mentioned. New instruction: "First, Describe and explain each action performed in the sequence provided. Then, based on the actions described, select the most logical continuation from the provided options."
This modification (Option A) helps to ensure that the user comprehends each part of the sequence fully before making a decision on its continuation, which could enhance understanding and accuracy in selecting the correct option.
So the option is: A.###""",

"""The instruction is: Multi-choice question: 'What is the sentiment of the following tweet? Tweet: @nikkigreen I told you Choose your answer from: negative; positive'

Let's think step by step. For the given instruction, a suitable adaptation is to keep as original version. The instruction is clear and directly asks to perform the multi-choice question. Addition information will merely help answering the questions.
So the option is: D.###""",

"""The instruction is: 'Give three tips for staying healthy.'

Let's think step by step. For the given instruction, a suitable adaptation is to add a prefix task. This would involve asking the user to perform a preliminary activity before providing the tips for staying healthy. New instruction: "Research the most common health issues in your area, and then give three tips for staying healthy based on your findings."
This addition prepares the user by directing them to focus on specific health concerns relevant to their location, making the subsequent tips more targeted and practical.
So the option is: B.###"""
]

FEW_SHOTS_EXAMPLE_CHAT = [
    """The instruction is: Describe the structure of an atom.

Let's think step by step. For the given instruction, a suitable adaptation is to add a suffix task. This would deepen the user's understanding by applying the knowledge in a practical context. New instruction: "Describe the structure of an atom and explain how this structure determines its chemical properties."
This modification (Option C) not only covers the original request to describe the atom's structure but also extends the learning by connecting atomic structure to chemical properties, making the explanation more comprehensive and applicable.
So the option is: C.""",

    """The instruction is: 'James runs a TV show and there are 5 main characters and 4 minor characters. He pays the minor characters $15,000 each episode. He paid the major characters three times as much. How much does he pay per episode? Let's be accurate as possible.'

Let's think step by step. For the given instruction, a suitable adaptation is to keep as original version. The instruction is already a well-formed, clear mathematical problem that conveys all the necessary information needed to solve it. It integrates a real-world context with a math problem, which can engage users by connecting calculations to practical scenarios.
So the option is: D.""",

"""The instruction is: 'What happens next in this paragraph? She then rubs a needle on a cotton ball then pushing it onto a pencil and wrapping thread around it. She then holds up a box of a product and then pouring several liquids into a bowl. Choose your answer from: A. adds saucepan and shakes up the product in a grinder. B. pinches the thread to style a cigarette, and then walks away. C. then dips the needle in ink and using the pencil to draw a design on her leg, rubbing it off with a rag in the end. D. begins to style her hair and cuts it several times before parting the ends of it to show the hairstyle she has created.'

Let's think step by step. For the given instruction, a suitable adaptation is to decompose it into two tasks. The existing instruction involves understanding the sequence described and choosing an appropriate continuation. This can be decomposed into first identifying and explaining each action in the sequence, and then predicting the next action based on logical continuation or typical usage of the items mentioned. New instruction: "First, Describe and explain each action performed in the sequence provided. Then, based on the actions described, select the most logical continuation from the provided options."
This modification (Option A) helps to ensure that the user comprehends each part of the sequence fully before making a decision on its continuation, which could enhance understanding and accuracy in selecting the correct option.
So the option is: A.""",

"""The instruction is: Multi-choice question: 'What is the sentiment of the following tweet? Tweet: @nikkigreen I told you Choose your answer from: negative; positive'

Let's think step by step. For the given instruction, a suitable adaptation is to keep as original version. The instruction is clear and directly asks to perform the multi-choice question. Addition information will merely help answering the questions.
So the option is: D.""",

"""The instruction is: 'Give three tips for staying healthy.'

Let's think step by step. For the given instruction, a suitable adaptation is to add a prefix task. This would involve asking the user to perform a preliminary activity before providing the tips for staying healthy. New instruction: "Research the most common health issues in your area, and then give three tips for staying healthy based on your findings."
This addition prepares the user by directing them to focus on specific health concerns relevant to their location, making the subsequent tips more targeted and practical.
So the option is: B."""
]

PROMPT_TEMPLATE = """
The instruction is: "{}". {}

Let's think step by step."""

INPUT_TEMPLATE = "Input: '{}'"

def get_prompt(p, is_chat=False):
    few_shot_example = FEW_SHOTS_EXAMPLE if not is_chat else FEW_SHOTS_EXAMPLE_CHAT
    prompt_prefix = PROMPT_PREFIX if not is_chat else PROMPT_PREFIX_CHAT

    e = few_shot_example.copy()
    random.shuffle(e) # shuffle the few shot examples to prevent position bias
    prompt = prompt_prefix + '\n\n' + '\n\n'.join(e)

    if 'conversations' in p: # cases for lima
        instruction, output = p['conversations'][0], p['conversations'][1]
        prompt += '\n\n' + PROMPT_TEMPLATE.format(instruction, '')
        input = ''
    elif 'question' in p: # cases for flancot
        instruction = p['question']
        prompt += '\n\n' + PROMPT_TEMPLATE.format(instruction, '')
        input = ''
    else: # cases for alpaca like data (with input)
        instruction = p['instruction']
        input = ''
        if p['input'] != '':
            input = INPUT_TEMPLATE.format(p['input'])
            prompt += '\n\n' + PROMPT_TEMPLATE.format(instruction, input)
        else:
            prompt += '\n\n' + PROMPT_TEMPLATE.format(instruction, '')

    if 'system_prompt' in p:
        system_prompt = p['system_prompt'] # this is used for the sequential instruction genereation process, different from SYSTEM_PROMPT
    else:
        system_prompt = ''

    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]
    return {'prompt': prompt, 'instruction': instruction, 'input': input, 'messages': messages, 'system_prompt': system_prompt}