"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class CaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        print('self.annotation in caption_dataset', self.annotation[1])
        self.img_ids = {}
        n = 0
        '''
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        '''
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        #print('flag0 in getitem')
        ann = self.annotation[index]
        #print('flag1 in getitem')
        image_path = os.path.join(self.vis_root, ann["image"])
        #print(image_path)
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            return None # image does not exist
        image = self.vis_processor(image)
        #caption = self.text_processor(ann["caption"])
        #print(ann["text_input"])
        text_input = self.text_processor(ann["text_input"])
        text_output = self.text_processor(ann["text_output"])
        #print(text_input)
        return {
            "image": image,
            "text_input": text_input,
            "text_output": text_output,
            #"image_id": ann["image_id"]
        }

class CaptionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {
            "image": image,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }

class CaptionInstructDataset(CaptionDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = data["text_input"]
            data['text_input'] = self.text_processor("")
        return data
