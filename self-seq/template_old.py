import pandas
from datasets import load_dataset

FEW_SHOTS_EXAMPLE = [
    """Given the original instruction, you should propose a new instruction based on it by doing one of following things:
1. Decompose it into two tasks.
2. Add a prefix task.
3. Add a suffix task. 
Ensure that the newly added task will either help the answering for original instruction or sequentially follow the instruction's answer.
This is the original instruction: “Describe the structure of an atom.”
So what is the option you choose and new instruction?  Please answer me by replying in the format of "Option: , Instruction:". After that, give answer with the generated instruction and input (if have), with clear separation between answering the first and second part of instruction.

Option: 1, Instruction: First, describe the structure of an atom. Then, explain how electrons contribute to chemical bonding.
Describe the structure of an atom:
An atom is the smallest unit of matter that retains all of the chemical properties of an element. It is composed of three main types of subatomic particles: protons, neutrons, and electrons. Protons and neutrons form the atom's nucleus at the center, with protons carrying a positive charge and neutrons being neutral. Electrons, which carry a negative charge, orbit the nucleus in electron shells. The number of protons in the nucleus determines the element's atomic number, and the combination of protons and neutrons gives the atom its mass. Electrons play a crucial role in chemical reactions and bonding, as they can be shared between atoms or transferred from one atom to another.
Explain how electrons contribute to chemical bonding:
Electrons contribute to chemical bonding through two primary mechanisms: ionic and covalent bonding. In ionic bonding, electrons are transferred from one atom to another, resulting in the formation of positively charged ions (cations) and negatively charged ions (anions) which are attracted to each other due to their opposite charges. This type of bond is typically formed between metals and nonmetals. In covalent bonding, electrons are shared between atoms, allowing each atom to attain a stable electron configuration. This sharing can involve equal sharing (nonpolar covalent bonding) or unequal sharing (polar covalent bonding) of electrons. Electrons in the outermost shell, known as valence electrons, are particularly important in bonding as they determine the chemical reactivity and bonding capacity of an atom.""",

"""Given the original instruction, you should propose a new instruction based on it by doing one of following things:
1. Decompose it into two tasks.
2. Add a prefix task.
3. Add a suffix task. 
Ensure that the newly added task will either help the answering for original instruction or sequentially follow the instruction's answer.
This is the original instruction: “Identify the fallacy used in the following statement.
Output should be a single word.” And this is the original input: “We should not accept the refugees from Syria because they are likely to steal from us.”
So what is the option you choose and new instruction?  Please answer me by replying in the format of "Option: , Instruction:". After that, give answer with the generated instruction and input (if have), with clear separation between answering the first and second part of instruction.

Option: 2, Instruction: First, define logical fallacies and provide examples, then identify the fallacy used in the following statement.
Define logical fallacies and provide examples:
1. Ad Hominem: Attacking the person making an argument rather than the argument itself.
2. Straw Man: Misrepresenting someone's argument to make it easier to attack.
3. Appeal to Ignorance: Arguing that a claim is true because it has not been proven false, or vice versa.
4. False Dilemma: Presenting two opposing options as the only possibilities, when in fact more options exist.
5. Slippery Slope: Arguing that a minor action will lead to significant and often ludicrous consequences without showing any causal connection.
6. Circular Reasoning: When the argument repeats the claim as a way to provide evidence.
7. Hasty Generalization: Making a broad generalization based on a small or unrepresentative sample.
8. Red Herring: Introducing an irrelevant topic to divert attention from the subject of the argument.
9. Appeal to Authority: Claiming something is true because an authority figure says it is, without any other supporting evidence.
10. Appeal to Emotion: Manipulating an emotional response in place of a valid or compelling argument.
11. Bandwagon Fallacy: Arguing that something is true or good simply because it is popular.
Identify the fallacy used in the following statement:
“We should not accept the refugees from Syria because they are likely to steal from us.”
For the statement, "We should not accept the refugees from Syria because they are likely to steal from us," the logical fallacy used is:
Hasty Generalization""",

"""Given the original instruction, you should propose a new instruction based on it by doing one of following things:
1. Decompose it into two tasks.
2. Add a prefix task.
3. Add a suffix task. 
Ensure that the newly added task will either help the answering for original instruction or sequentially follow the instruction's answer.
This is the original instruction: “Identify the fallacy used in the following statement.
Output should be a single word.” And this is the original input: “We should not accept the refugees from Syria because they are likely to steal from us.”
So what is the option you choose and new instruction?  Please answer me by replying in the format of "Option: , Instruction:". After that, give answer with the generated instruction and input (if have), with clear separation between answering the first and second part of instruction.

Option: 3, Instruction: List four reasons why it would be beneficial to take a gap year, and afterwards, evaluate how each reason could impact your personal growth and career trajectory.
List four reasons why it would be beneficial to take a gap year:
Taking a gap year, a break typically taken between high school and college or during undergraduate studies, can offer numerous benefits for personal growth and career trajectory. Here are four reasons why a gap year can be beneficial:
1. Exposure to Different Cultures and Perspectives
2. Improved Academic Focus
3. Development of Life Skills
4. Enhanced Resume and Network
Evaluate how each reason could impact your personal growth and career trajectory:
1. Exposure to Different Cultures and Perspectives
Impact on Personal Growth: Developing a global perspective can transform your worldview, making you more open-minded and adaptable. These qualities are invaluable in personal relationships and in navigating life's challenges.
Impact on Career Trajectory: Employers value candidates with international experience and the ability to work effectively in diverse teams. Your broadened perspectives can make you a more attractive job candidate and can lead to opportunities in international fields or organizations.
2. Improved Academic Focus
Impact on Personal Growth: Taking time to understand your interests and goals can lead to a more fulfilling academic and professional life. This self-awareness is a key component of personal development.
Impact on Career Trajectory: With a clearer understanding of your academic and career objectives, you're more likely to pursue and excel in fields that genuinely interest you, leading to a more satisfying and potentially successful career.
3. Development of Life Skills
Impact on Personal Growth: Mastering these life skills can increase your independence and confidence. These traits are beneficial in both personal and professional contexts, enhancing your ability to manage challenges.
Impact on Career Trajectory: Employers look for candidates with strong life skills, as they are indicative of someone who is reliable and can handle responsibility. These skills can give you an edge in the job market and are often transferable across different roles and industries.
4. Enhanced Resume and Network
Impact on Personal Growth: Building a diverse network can expand your horizons and offer support in both personal and professional ventures. The experiences you gain can also boost your self-esteem and communication skills.
Impact on Career Trajectory: A standout resume with unique experiences and a broad network can open doors to job opportunities. Networking can lead to career advancements and insights into different industries, potentially speeding up your career progression."""
]

PROMPT_TEMPLATE = """
Given the original instruction, you should propose a new instruction based on it by doing one of following things:
1. Decompose it into two tasks.
2. Add a prefix task.
3. Add a suffix task. 

Ensure that the newly added task will either help the answering for original instruction or sequentially follow the instruction's answer.
This is the original instruction: "{}" {}.

So what is the option you choose and new instruction?  Please answer me by replying in the format of "Option: , Instruction:". After that, give answer with the generated instruction and input (if have), with clear separation between answering the first and second part of instruction.
"""
INPUT_TEMPLATE = "And this is the original input: '{}'"