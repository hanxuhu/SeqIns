PROMPT_PREFIX = """Given the original instruction and newly generated instruction, your task is to determine if the new instruction contains all necessary information from the original instruction to answer the question? Answer with “Yes” or “No” followed by “Answer: “. If it is yes, don't need to provide anything else. If it is no, you should rewrite the new instruction with prefix “#New Instruction#: “ and end with “###”, so that it contained the necessary information from the original instruction, given original instruction won't be visible when answering. 
"""

FEW_SHOTS_EXAMPLE = [
"""#Original Instruction#: “Countdown to Nowhere is the fourth full-length studio album from Allister, a pop punk band from Chicago, Illinois, released in Japan on June 16, 2010 and in the U.S. on October 5, 2010. It was the band's first full-length album released in five years, and their first album released after the conclusion of their three-year hiatus.\n\nCan we draw the following conclusion?\nAll members are Americans.”
#New Instruction#: “Identify the band's origin country before reading the paragraph. Then, based on the provided information, conclude whether the assumption about the band members' nationalities is correct.”
Answer: No
#New Instruction#: “Countdown to Nowhere is the fourth full-length studio album from Allister, a pop punk band from Chicago, Illinois, released in Japan on June 16, 2010, and in the U.S. on October 5, 2010. It was the band's first full-length album released in five years, and their first album released after the conclusion of their three-year hiatus. Based on this information, identify the band's origin country and conclude whether the assumption that all members are Americans is correct.”###""",

"""#Original Instruction#: “Premise: The nave is a wonder of light and lofty proportions, enhanced by the luminous beige stone and the splendid ribbed vaulting.\n\nHypothesis: The nave is known for being exceptionally large in size.\n\nDoes the premise entail the hypothesis?\n\nPossible answers:\n(i) yes\n(ii) it is not possible to tell\n(iii) no”
#New Instruction#: “Identify the key traits or measurements associated with the 'nave' and determine if they align with the hypothesis that emphasizes its size. \n\nPremise: The nave is a wonder of light, boasting lofty proportions, an impressive sight with its luminous beige stone and stunning ribbed vaulting. \n\nHypothesis: The nave is known for its exceptionally large size. \n\nDoes the premise entail the hypothesis?”
Answer: Yes###"""
]

PROMPT_TEMPLATE = """#Original Instruction#: “{}”
#New Instruction#: “{}”"""
#Original Instruction#: “Given the following question, let's solve step-by-step. \"Birch, the long-lived pioneer species, are found widely spread in the Southern Hemisphere.\"\nIs the above claim true?\nOptions:\n- yes\n- no\n”

#New Instruction#: “Given the question about Birch trees, discuss your thought process and conclude whether the claim is true or not. Additionally, if you think the claim is not true, provide a counterexample or an alternative scenario that contradicts the claim.”