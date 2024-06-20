import random

PROMPT_PREFIX_A="""Your objective is to decompose the given instruction (#Original Instruction#) into two logical related sequential instructions (#New Instruction#) by either make the original instruction more complex or clear to handle. 
The response to the new instruction should be the same or similar to the original instruction, including the format. The added instruction should have its own explicit response, so something like “reading”, "familiarizing", "repeating", “analyzing” or "understanding" the original instruction is not considered a good choice.
Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:, and should only modify the instruction part and keep all the key details such as options, hypothesis and questions.
Provide your explanation before having the final instruction by thinking step by step.
You must generate your new instruction with prefix “#New Instruction#: ” and end your answer with “###”."""

PROMPT_PREFIX_B="""Your objective is to add prefix task to the given instruction (#Original Instruction#) to form a sequential related instruction (#New Instruction#). 
Adding familiarize, read or understand the original given information is not counted as valid prefix task.
The response to the new instruction should be the same or similar to the original instruction, including the format. The added instruction should have its own explicit response, so something like “reading”, "familiarizing", "repeating", “analyzing” or "understanding" the original instruction is not considered a good choice.
Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:, and should only modify the instruction part and keep all the key details such as options, hypothesis and questions.
Provide your explanation before having the final instruction by thinking step by step.
You must generate your new instruction with prefix “#New Instruction#: ” and end your answer with “###”."""

PROMPT_PREFIX_C="""Your objective is to add suffix task to the given instruction (#Original Instruction#) to form a sequential related instruction (#New Instruction#). 
Adding familiarize, read or understand the original given information is not counted as valid prefix task.
The response to the new instruction should be the same or similar to the original instruction, including the format. The added instruction should have its own explicit response, so something like “reading”, "familiarizing", "repeating", “analyzing” or "understanding" the original instruction is not considered a good choice.
Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:, and should only modify the instruction part and keep all the key details such as options, hypothesis and questions.
Provide your explanation before having the final instruction by thinking step by step.
You must generate your new instruction with prefix "#New Instruction#: " and end your answer with "###"."""

FEW_SHOTS_EXAMPLE_A = [
    """#Original Instruction#:'Describe the structure of an atom.'
You task is to decompose the instruction into two sequential instructions that will eventually lead to the answer the original instructions. Let's think step by step. To effectively describe the structure of an atom, we can break down the explanation into two main tasks or steps. Here’s a logical way to organize it. Firstly, we can explore the basic components of an atom, then explores how the components are organized and how they interact. These two tasks cover the basic description of an atom's structure, from its components to the arrangement and behavior of these components.
#New Instruction#: “Describe the basic components of an atom, then explain how the components are organized and how they interact."###""",
    
    """#Original Instruction#:'What is an example of a tweet?'
You task is to decompose the instruction into two sequential instructions that will eventually lead to the answer the original instructions. Let's think step by step. To effectively describe an example of a tweet, the explanation can be decomposed into several focused steps. Here's how we might logically organize it: 1. Begin by defining what a tweet is, emphasizing its platform origin (Twitter), character limit, and primary function; 2. Next, explore the elements that make up a tweet, such as the text content, media attachments (photos, videos, links), and interactive features (like, retweet, and reply buttons); 3. Discuss the typical contexts in which tweets are used, such as news dissemination, personal updates, or promotional content; 4.  Provide a specific example of a tweet, detailing its content, purpose, and any notable reactions or interactions it may have received.
#New instruction#: “Define what a tweet is, explain its components, discuss its typical uses, and provide a specific example of a tweet."###""",

    """#Original Instruction#: 'The plate needed to be put away where did the mother put it?\nOptions:\n- flea market\n- kitchen cupboard\n- dishwasher\n- restaurant\n- museum\nPlease answer and provide answer explanation.'
You task is to decompose the instruction into two sequential instructions that will eventually lead to the answer the original instructions. Let's think step by step. The instruction are splited into two task, the initial task involves understanding the context provided, and the second task requires selecting the correct location and providing an explanation for the choice.
#New Instruction#: “The plate needed to be put away where did the mother put it?\nOptions:\n- flea market\n- kitchen cupboard\n- dishwasher\n- restaurant\n- museum\n First, read the context provided and identify the appropriate location where the mother put the plate. Then, select the correct option from the given choices and provide an explanation for your selection based on the context.”###"""
]

FEW_SHOTS_EXAMPLE_B = [
    """#Original Instruction#:'Summarize this article in one sentence. The incident happened on Fife Street between 04:30 BST and 05:00 BST on Saturday. The police have appealed for witnesses to the crash. York Road is closed between its junctions with Skegoneill Avenue and Alexandra Park Avenue. Diversions are in place. Summary:'
Now adding a prefix task. Let's think step by step. The first task could be identifying the key components of the article, and the second task would focus on creating the one-sentence summary. This approach would help in better handling the process of summarizing by breaking it down into more manageable steps.
#New Instruction#: “Identify the details of the article, noting the key events and information provided, based on your understanding, summarize the article in one sentence. The incident happened on Fife Street between 04:30 BST and 05:00 BST on Saturday. The police have appealed for witnesses to the crash. York Road is closed between its junctions with Skegoneill Avenue and Alexandra Park Avenue. Diversions are in place.”###""",

    """#Original Instruction#: 'Here is a goal: To hold down a keyboard note without using your hands, How would you accomplish this goal?'
Now adding a prefix task. Let's think step by step. The original instruction asks how to accomplish a goal, but it doesn't specify preparing for the task or considering necessary tools or methods. By adding a prefix task, you can enhance the clarity and feasibility of achieving the goal.
#New Instruction#: “Assess available objects or tools that can be used to depress a keyboard note, and determine how to use the selected object or tool to hold down a keyboard note without using your hands.”###""",

    """#Original Instruction#: 'Generate a movie review with positive sentiment.'
Now adding a prefix task. Let's think step by step. The instruction already directs to produce a specific type of content, namely a movie review that has a positive tone. However, it lacks a step that involves evaluating or considering specific elements of the movie which could be highlighted in the review to reinforce the positive sentiment. Adding a prefix task would provide a clearer framework for constructing a positive review by focusing on specific positive aspects of the movie.
#New Instruction#: “Evaluate the key positive elements of the movie, such as acting, direction, storyline, and cinematography, and based on these elements, generate a movie review with positive sentiment.”###"""
]

FEW_SHOTS_EXAMPLE_C = [
    """#Original Instruction#:'How did Julius Caesar die?'
Now adding a suffix task. Let's think step by step. The newly suffix taskwould be to add a suffix task to gain a deeper understanding of the impact of Caesar's death. This approach would not only informs about the manner of Caesar's demise but also encourages users to explore the subsequent events that shaped history, providing a more holistic understanding of the significance of Caesar's death.
#New Instruction#: “How did Julius Caesar die? and explain the consequences of Julius Caesar's death on the course of history, particularly in Rome.”###""",
"""#Original Instruction#: 'Generate three verbs that mean the same as "to apologize.”'
Now adding a suffix task. Let's think step by step. The original instruction asks how to accomplish a goal, but it doesn't specify preparing for the task or considering necessary tools or methods. The suffix task could focus on the practical use of the newly generated three verbs, improving comprehension and making the task more engaging.
#New Instruction#: “Generate three verbs that mean the same as "to apologize”. Then provide three situations in which one might use these verbs to mean 'to apologize'.”###""",
"""#Original Instruction#: Given the original instruction: 'Premise: \"A tour group is standing on the grass with ruins in the background.\"\nBased on this premise, can we conclude that the hypothesis \"Some women are posing for a picture.\" is true?\nOptions:\n- yes\n- it is not possible to tell\n- no Let's be accurate as possible and think first.'
Now adding a suffix task. Let's think step by step. The instruction directs to pick from either choice of the options. However, it lacks a explanation of the decisions. Adding a suffix task would involves making a logical deduction based on the premise provided and justifying it. This addition not only tests a user's ability to evaluate the hypothesis but also emphasizes the reasoning behind the conclusion, enhancing critical thinking skills.
#New Instruction#: “Premise: \"A tour group is standing on the grass with ruins in the background.\"\nBased on this premise, can we conclude that the hypothesis \"Some women are posing for a picture.\" Based on the premise provided, deduce whether the hypothesis is true, false, or if there isn't enough information to determine the answer. Then, explain your thought process and conclusion.”###"""
]

PROMPT_TEMPLATE_A = """#Original Instruction#: '{}' {}
You task is to decompose the instruction into two sequential instructions that will eventually lead to the answer the original instructions. Let's think step by step. """
PROMPT_TEMPLATE_B = """#Original Instruction#: '{}' {}
Now adding a prefix task. Let's think step by step. """
PROMPT_TEMPLATE_C = """#Original Instruction#: '{}' {}
Now adding a suffix task. Let's think step by step. """

def get_gen_instruction_prompt(p):
    if (p['option'] is None) or (p['option'] == 'D'):
        return {
            **p,
            'new_instruction': p['instruction'],
        }
    
    if p['option'] == 'A':
        prompt_prefix = PROMPT_PREFIX_A
        few_shot_examples = FEW_SHOTS_EXAMPLE_A
        prompt_template = PROMPT_TEMPLATE_A
    elif p['option'] == 'B':
        prompt_prefix = PROMPT_PREFIX_B
        few_shot_examples = FEW_SHOTS_EXAMPLE_B
        prompt_template = PROMPT_TEMPLATE_B
    elif p['option'] == 'C':
        prompt_prefix = PROMPT_PREFIX_C
        few_shot_examples = FEW_SHOTS_EXAMPLE_C
        prompt_template = PROMPT_TEMPLATE_C

    e = few_shot_examples.copy()
    random.shuffle(e)
    prompt = prompt_prefix + '\n\n' + '\n\n'.join(e)
    prompt += '\n\n' + prompt_template.format(p['instruction'])
    messages = [{'role': 'user', 'content': prompt}]
    return {**p, 'messages': messages}