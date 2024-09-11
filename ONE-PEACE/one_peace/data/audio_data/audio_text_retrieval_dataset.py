# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch
import os
import random

from ..base_dataset import BaseDataset


class AudioTextRetrievalDataset(BaseDataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        dictionary,
        max_src_length=70,
        max_duration=15,
        feature_encoder_spec='[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]'
    ):
        super().__init__(split, dataset, bpe, dictionary)
        self.max_src_length = max_src_length
        self.max_duration = max_duration
        self.feature_encoder_spec = eval(feature_encoder_spec)

    def __getitem__(self, index, item_tuple=None):
        item_tuple = self.dataset[index] if item_tuple is None else item_tuple
        # uniq_id, audio, caption, duration = item_tuple
        filename,caption1,caption2,caption3,caption4,caption5 = item_tuple
        li = filename.replace('.wav','').split('_')
        if li is not None:
            # uniq_id = int(uniq_id) if isinstance(uniq_id, int) else uniq_id
            uniq_id = int(li[0][1:]+li[3]+li[4])
        audio = os.path.join('/workspace/jaeyoung/datasets/mm-tts-dataset/raw',filename)
        if audio is not None:
            wav, curr_sample_rate = self.read_audio(audio)
            feats = torch.tensor(wav)
        else:
            feats = torch.randn(16000)
            curr_sample_rate = 16000
        feats = self.audio_postprocess(feats, curr_sample_rate, self.max_duration)
        T = self._get_mask_indices_dims(feats.size(-1), self.feature_encoder_spec)
        audio_padding_mask = torch.zeros(T + 1).bool()
        caption_list = [caption1, caption2, caption3, caption4, caption5]
        caption = random.choice(caption_list)
        caption = self.process_text(caption)
        text_src_item = self.encode_text(' {}'.format(caption), self.max_src_length)

        example = {
            "id": uniq_id,
            "source_text": text_src_item,
            "source_audio": feats,
            "audio_padding_mask": audio_padding_mask,
        }
        return example