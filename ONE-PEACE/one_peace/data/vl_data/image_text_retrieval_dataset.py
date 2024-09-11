# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.
import random
import os

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ..base_dataset import BaseDataset, CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD


class ImageTextRetrievalDataset(BaseDataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        dictionary,
        max_src_length=70,
        patch_image_size=256
    ):
        super().__init__(split, dataset, bpe, dictionary)
        self.max_src_length = max_src_length
        self.patch_image_size = patch_image_size

        mean = CLIP_DEFAULT_MEAN
        std = CLIP_DEFAULT_STD

        self.transform = transforms.Compose([
            transforms.Resize((patch_image_size, patch_image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index, item_tuple=None):
        item_tuple = self.dataset[index] if item_tuple is None else item_tuple
        # uniq_id, image, caption = item_tuple
        filename,caption1,caption2,caption3,caption4,caption5 = item_tuple
        li = filename.replace('.wav','').split('_')
        if li is not None:
            # uniq_id = int(uniq_id) if isinstance(uniq_id, int) else uniq_id
            uniq_id = int(li[0][1:]+li[3]+li[4])

        image_dir = os.path.join('/workspace/jaeyoung/datasets/mm-tts-dataset/video_image_save', filename.replace('.wav',''))
        image_list = os.listdir(image_dir)
        caption_list = [caption1, caption2, caption3, caption4, caption5]
        image = os.path.join(image_dir, random.choice(image_list))
        caption = random.choice(caption_list)

        caption = self.process_text(caption)
        text_src_item = self.encode_text(' {}'.format(caption), self.max_src_length)

        if image is not None:
            image = self.read_image(image)
            patch_image = self.transform(image)
        else:
            patch_image = torch.randn((self.patch_image_size, self.patch_image_size))

        example = {
            "id": uniq_id,
            "source_text": text_src_item,
            "source_image": patch_image
        }
        return example