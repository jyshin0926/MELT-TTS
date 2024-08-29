# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch

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
        # item_tuple = self.dataset[index]
        # uniq_id, audio, caption, duration = item_tuple
        # TODO:: caption 5개랑 keyword 는 어떻게 쓰면 좋을지 생각해보기 (+audio_text_retrieval.py 의 valid_file 이랑 같이 )
        # file_name,caption_1,caption_2,caption_3,caption_4,caption_5,keywords,sound_id = item_tuple
        file_name, caption, sound_id, *others = item_tuple
        if sound_id is not None:
            # sound_id = int(sound_id) if isinstance(sound_id, int) or sound_id.isdigit() else sound_id
            sound_id = int(sound_id) if isinstance(sound_id, int) or sound_id.isdigit() else 1  # 'Not found' 케이스 숫자 1 처리


        if file_name is not None:
            wav, curr_sample_rate = self.read_audio(file_name)
            feats = torch.tensor(wav)
        else:
            feats = torch.randn(16000)
            curr_sample_rate = 16000
        feats = self.audio_postprocess(feats, curr_sample_rate, self.max_duration)
        T = self._get_mask_indices_dims(feats.size(-1), self.feature_encoder_spec)
        audio_padding_mask = torch.zeros(T + 1).bool()

        caption = self.process_text(caption)
        text_src_item = self.encode_text(' {}'.format(caption), self.max_src_length)

        example = {
            "id": sound_id,
            "source_text": text_src_item,
            "source_audio": feats,
            "audio_padding_mask": audio_padding_mask,
        }
        return example