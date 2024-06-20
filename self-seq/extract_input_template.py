# import pandas
import random

# SYSTEM_PROMPT = "You are an assistant to classify the correct choices for the given instruction. Please follow strictly the format of the few shot examples given and provide explanation."
PROMPT_PREFIX = """Please extract the instruction and input from the given prompt. The instruction can be a question or a task to perform. The input will be all other details, including sentences, table, code data to necessary to complete the task in the instruction. The instruction should not be overlap with the input and contain any part of the input, to prevent duplication. There must be an instruction but not necessary an input, the input can be empty string only when needed. Your extraction should neither loss information from or add new information to the prompt. Please answer with the following format: Instruction: “[INSTRUCTION]###\nInput: [INPUT]###”, and no more new contents after the last ###."""

# PROMPT_PREFIX_CHAT = """Given the original instruction, you should propose a new instruction based on it by doing one of following things:
# A. Decompose it into two tasks.
# B. Add a prefix task.
# C. Add a suffix task.
# D. Keep as original version. (Choose this if the original instruction is already sufficient)
# You should decide which option is suitable for the input instruction.
# """

FEW_SHOTS_EXAMPLE = [
    ("""Prompt: "Objective: To stamp your patter on the butter cookie dough\n\nWhich of the following solutions is more sound in terms of naive physics reasoning?" """,
"""Instruction: "Which of the following solutions is more sound in terms of naive physics reasoning?"###
Input: "Objective: To stamp your patter on the butter cookie dough"###"""),

("""Prompt: "Leo: Given the sentence \"This person is about to put paper into a copy machine.\" is it true that \"The copy machine is too full to add more paper at this time.\"?\nMei: OK, So, let's think first...\nMe:" """,
"""Instruction: "Given the sentence "This person is about to put paper into a copy machine." is it true that "The copy machine is too full to add more paper at this time."?"###
Input: ###"""),

("""Prompt: "Write a sentence that about [15 WINNING_TEAM Tasman Motorsports;   1995 IndyCar season;  RND 15; 15 POLE_POSITION André Ribeiro; 15 RACE_NAME New England 200; 15 FASTEST_LAP Teo Fabi]." """,

"""Instruction: "Write a sentence that about the input."###
Input: The input is 15 WINNING_TEAM Tasman Motorsports; 1995 IndyCar season; RND 15; 15 POLE_POSITION André Ribeiro; 15 RACE_NAME New England 200; 15 FASTEST_LAP Teo Fabi ###"""),

("""Prompt: “Otto Stern (17 February 1888 \u2013 17 August 1969) was a German physicist and Nobel laureate in physics. He was the second most nominated person for a Nobel Prize with 82 nominations in the years 1925\u20131945 (most times nominated is Arnold Sommerfeld with 84 nominations), ultimately winning in 1943.\nDoes this next sentence follow, given the preceding text?\nOtto Stern was the most nominate person for a Nobel prize.\n\npick from the following. [-] Yes. [-] It's impossible to say. [-] No.” """,
"""Instruction: Does the following sentence follow logically from the preceding text?\n\npick from the following. [-] Yes. [-] It's impossible to say. [-] No.###
Input: Otto Stern was the most nominate person for a Nobel prize. Text: Otto Stern (17 February 1888 \u2013 17 August 1969) was a German physicist and Nobel laureate in physics. He was the second most nominated person for a Nobel Prize with 82 nominations in the years 1925\u20131945 (most times nominated is Arnold Sommerfeld with 84 nominations), ultimately winning in 1943.###"""),
]

PROMPT_TEMPLATE = """Prompt: {}"""