"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union, List, Dict


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def generate_chat_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> List[Dict[str, str]]:
        if input:
            res = f'{instruction} Input: {input}'
        else:
            res = f'{instruction}'
        res = [{'role': 'user', 'content': res}]
        if label:
            res.append({'role': 'assistant', 'content': label})
        return res
    
    def get_response(self, output: str) -> str:
        split = output.split(self.template["response_split"])
        if len(split) < 2:
            return output.strip()
        else:
            return split[1].strip()    
        return output.split(self.template["response_split"])[1].strip()
