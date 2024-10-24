import time
import os
import random
import numpy as np
import bisect
import torch
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
import gzip
import pickle
from PIL import Image
import torchvision.transforms as transforms

import commons 
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cleaned_text_to_sequence


"""Multi speaker version"""
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self._save_dir = "/workspace/jaeyoung/data/emotion_vits"
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)
        
        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()
        
    def _ensure_prompt_fields(self, data):
        """
        Ensure that each data entry has 'vision_path' and 'text_prompt'.
        If not, append empty strings or appropriate default values.
        """
        for entry in data:
            if len(entry) < 5:
                # Append empty string for 'text_prompt'
                entry.append("")  # text_prompt
            if len(entry) < 6:
                # Append empty string for 'vision_path'
                entry.append("")  # vision_path
        return data
    
    
    def _filter(self):
        """
        Filter text & store spec lengths
        """
        pkl_path = "/workspace/jaeyoung/speech/emo_text"
        os.makedirs(pkl_path, exist_ok=True)
        audio_sid_txt_pkl = f"{pkl_path}/emo_sid_txt.pkl"
        len_pkl = f"{pkl_path}/lengths.pkl"
        if os.path.exists(audio_sid_txt_pkl):
            with gzip.open(audio_sid_txt_pkl, "rb") as f:
                audiopaths_sid_text_new = pickle.load(f)

        if os.path.exists(len_pkl):
            with gzip.open(len_pkl, "rb") as f:
                lengths = pickle.load(f)
        else:
            audiopaths_sid_text_new = []
            lengths = []
            min_audio_length = self.filter_length  # Ensure audio is long enough
            for audiopath, sid, text, vision_path, caption in tqdm(self.audiopaths_sid_text):
                audio_length = os.path.getsize(audiopath) // (2 * self.hop_length)  # Adjust based on your data
                if self.min_text_len <= len(text) <= self.max_text_len and audio_length >= min_audio_length:
                    audiopaths_sid_text_new.append([audiopath, sid, text, vision_path, caption])
                    lengths.append(audio_length)
                else:
                    continue

            with gzip.open(audio_sid_txt_pkl, "wb") as f:
                pickle.dump(audiopaths_sid_text_new, f)

            with gzip.open(len_pkl, "wb") as f:
                pickle.dump(lengths, f)

        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths


    # def _filter(self):
    #     """
    #     Filter text & store spec lengths
    #     """

    #     pkl_path = "/workspace/jaeyoung/speech/emo_text"

    #     os.makedirs(pkl_path, exist_ok=True)
    #     audio_sid_txt_pkl = f"{pkl_path}/emo_sid_txt.pkl"
    #     len_pkl = f"{pkl_path}/lengths.pkl"
    #     if os.path.exists(audio_sid_txt_pkl):
    #         with gzip.open(audio_sid_txt_pkl, "rb") as f:
    #             audiopaths_sid_text_new = pickle.load(f)

    #     if os.path.exists(len_pkl):
    #         with gzip.open(len_pkl, "rb") as f:
    #             lengths = pickle.load(f)
    #     else:
    #         audiopaths_sid_text_new = []
    #         lengths = []
    #         for audiopath, sid, text, vision_path, caption in tqdm(self.audiopaths_sid_text):
    #             if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
    #                 audiopaths_sid_text_new.append([audiopath, sid, text, vision_path, caption])  # caption, vision_path 둘 다 한 데이터당 여러개이므로 random으로 보내야함
    #                 lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
    #             else:
    #                 continue

    #         with gzip.open(audio_sid_txt_pkl, "wb") as f:
    #             pickle.dump(audiopaths_sid_text_new, f)

    #         with gzip.open(len_pkl, "wb") as f:
    #             pickle.dump(lengths, f)

    #     self.audiopaths_sid_text = audiopaths_sid_text_new
    #     self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, text, vision_prompt, text_prompt = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2], audiopath_sid_text[3], audiopath_sid_text[4]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath, sid)
        sid = self.get_sid(sid)
        text_prompt = self.get_text_prompt(text_prompt)
        vision_prompt = self.get_vision_prompt(vision_prompt)
        # audio_prompt = self.get_audio_prompt(audiopath)
        audio_prompt = wav

        return (text, spec, wav, sid, text_prompt, vision_prompt, audio_prompt)
    
    def get_audio(self, filename, sid):
        audio, sampling_rate = load_wav_to_torch(filename)
        save_dir = os.path.join(self._save_dir, f"speaker_{sid}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Check if the sampling rate matches
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        
        audio_norm = audio / self.max_wav_value
        print(f"Audio norm shape: {audio_norm.shape}")  # [1, num_samples]
        
        _file = filename.split("/")[-1].split(".")[0]
        spec_filename = os.path.join(save_dir, f"{_file}_spec.pt")
        
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
            print(f"Loaded spectrogram shape: {spec.shape}")
        else:
            # Generate spectrogram
            spec = spectrogram_torch(audio_norm, self.filter_length,
                                    self.sampling_rate, self.hop_length, self.win_length,
                                    center=False)
            print(f"Spectrogram shape before any squeezing: {spec.shape}")  # Expected: [1, 513, time_frames]
            
            # Ensure the spectrogram is 2D
            if spec.dim() == 3:
                spec = spec.squeeze(0)
                print(f"Spectrogram shape after squeezing channel dim: {spec.shape}")  # Expected: [513, time_frames]
            
            # Convert to magnitude
            # Already handled in spectrogram_torch
            
            # Save the spectrogram
            torch.save(spec, spec_filename)
            print(f"Spectrogram saved with shape: {spec.shape}")
        
        return spec, audio_norm


    # def get_audio(self, filename, sid):
    #     audio, sampling_rate = load_wav_to_torch(filename)
    #     save_dir = os.path.join(self._save_dir, f"speaker_{sid}")
    #     os.makedirs(save_dir, exist_ok=True)
    #     if sampling_rate != self.sampling_rate:
    #         raise ValueError("{} {} SR doesn't match target {} SR".format(
    #             sampling_rate, self.sampling_rate))
    #     audio_norm = audio / self.max_wav_value
    #     audio_norm = audio_norm.unsqueeze(0)
    #     _file = filename.split("/")[-1].split(".")[0]
    #     # 각 Speaker별 폴더 생성
    #     spec_filename = os.path.join(save_dir, f"{_file}_spec.pt")
    #     if os.path.exists(spec_filename):
    #         spec = torch.load(spec_filename)
    #     else:
    #         spec = spectrogram_torch(audio_norm, self.filter_length,
    #             self.sampling_rate, self.hop_length, self.win_length,
    #             center=False)
            
    #         if spec.dim() == 3:
    #             spec = torch.squeeze(spec, 0)
    #         torch.save(spec, spec_filename)
    #     return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid
    
    # TODO:: emotion 단위로 spec 저장하기? (get_audio 함수 참고)
    def get_audio_prompt(self, audio_prompt_path):
        if audio_prompt_path != "" and os.path.exists(audio_prompt_path):
            audio, sampling_rate = load_wav_to_torch(audio_prompt_path)
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            # Compute spectrogram for audio_prompt
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            # Optionally, pad or slice spec to a fixed length
            max_spec_len = 1000  # Example fixed length
            if spec.size(1) > max_spec_len:
                spec = spec[:, :max_spec_len]
            else:
                pad_size = max_spec_len - spec.size(1)
                spec = F.pad(spec, (0, pad_size), "constant", 0)
            return spec
        else:
            # Return a zero tensor with fixed spec dimensions
            return torch.zeros(80, 1000)  # Assuming 80 mel channels and 1000 time steps
    
    def get_vision_prompt(self, vision_prompt):
        if vision_prompt != "" and os.path.exists(vision_prompt):
            image = Image.open(vision_prompt).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456,0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            image =transform(image)
        else:
            image = torch.zeros(3,224,224)
        return image
    
    def get_text_prompt(self, text_prompt):
        if text_prompt != "":
            prompt_norm = text_to_sequence(text_prompt, self.text_cleaners)
            prompt_norm = torch.LongTensor(prompt_norm)
            if self.add_blank:
                prompt_norm = commons.intersperse(prompt_norm, 0)
            max_prompt_len = self.max_text_len  # Or another suitable value            
            padded_prompt = torch.zeros(max_prompt_len, dtype=torch.long)
            if len(prompt_norm) > max_prompt_len:
                padded_prompt[:max_prompt_len] = prompt_norm[:max_prompt_len]
            else:
                padded_prompt[:len(prompt_norm)] = prompt_norm
            return padded_prompt
        else:
            return torch.zeros(self.max_text_len, dtype=torch.long)
        

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid, text_prompt, vision_prompt, audio_prompt]
        """
        if len(batch) == 0:
            return None

        print(f"Batch size: {len(batch)}")

        for i, x in enumerate(batch):
            print(f"Sample {i} spectrogram shape: {x[1].shape}")      # spec
            print(f"Sample {i} audio_prompt shape: {x[6].shape}")    # audio_prompt
            assert x[1].dim() == 2, f"Spectrogram at index {i} has invalid shape: {x[1].shape}"
            assert x[6].dim() == 2 and x[6].size(0) == 1, f"Audio prompt at index {i} has invalid shape: {x[6].shape}"

        # Sort by spec length in decreasing order
        spec_lengths = [x[1].size(1) for x in batch]
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor(spec_lengths),
            dim=0, descending=True
        )

        max_text_len = max([x[0].size(0) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])
        max_text_prompt_len = max([x[4].size(0) for x in batch])  # Corrected index
        # vision_prompt is assumed to be fixed size (3, 224, 224)
        max_audio_prompt_len = max_wav_len

        # Initialize tensors
        text_lengths = torch.LongTensor(len(batch))
        spec_lengths_tensor = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        text_prompt_lengths = torch.LongTensor(len(batch))
        audio_prompt_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_prompt_padded = torch.LongTensor(len(batch), max_text_prompt_len)
        vision_prompt_padded = torch.FloatTensor(len(batch), 3, 224, 224)  # Assuming fixed size
        audio_prompt_padded = wav_padded

        # Zero-padding
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        text_prompt_padded.zero_()
        vision_prompt_padded.zero_()
        audio_prompt_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths_tensor[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

            text_prompt = row[4]
            text_prompt_padded[i, :text_prompt.size(0)] = text_prompt
            text_prompt_lengths[i] = (text_prompt > 0).sum().item()

            vision_prompt = row[5]
            vision_prompt_padded[i] = vision_prompt

            # Audio Prompt (now wav)
            audio_prompt = row[6]
            audio_prompt_padded[i, :, :audio_prompt.size(1)] = audio_prompt
            audio_prompt_lengths[i] = audio_prompt.size(1)

        if self.return_ids:
            return (
                text_padded, text_lengths, spec_padded, spec_lengths_tensor,
                wav_padded, wav_lengths, sid, text_prompt_padded,
                vision_prompt_padded, audio_prompt_padded, ids_sorted_decreasing
            )

        return (
            text_padded, text_lengths, spec_padded, spec_lengths_tensor,
            wav_padded, wav_lengths, sid, text_prompt_padded,
            vision_prompt_padded, audio_prompt_padded
        )


# class TextAudioSpeakerCollate():
#     """ Zero-pads model inputs and targets
#     """
#     def __init__(self, return_ids=False):
#         self.return_ids = return_ids

    # def __call__(self, batch):
    #     """Collate's training batch from normalized text, audio and speaker identities
    #     PARAMS
    #     ------
    #     batch: [text_normalized, spec_normalized, wav_normalized, sid]
    #     """
    #     if len(batch) == 0:
    #         return None
        
    #     # Debugging: Print batch size
    #     print(f"Batch size: {len(batch)}")
        
    #     # Right zero-pad all one-hot text sequences to max input length
    #     _, ids_sorted_decreasing = torch.sort(
    #         torch.LongTensor([x[1].size(1) for x in batch]),
    #         dim=0, descending=True)

    #     max_text_len = max([len(x[0]) for x in batch])
    #     max_spec_len = max([x[1].size(1) for x in batch])
    #     max_wav_len = max([x[2].size(1) for x in batch])
    #     max_text_prompt_len = max([x[4].size(0) for x in batch])

    #     text_lengths = torch.LongTensor(len(batch))
    #     spec_lengths = torch.LongTensor(len(batch))
    #     wav_lengths = torch.LongTensor(len(batch))
    #     text_prompt_lengths = torch.LongTensor(len(batch))
    #     audio_prompt_lengths = torch.LongTensor(len(batch))
    #     sid = torch.LongTensor(len(batch))
        
    #     text_padded = torch.LongTensor(len(batch), max_text_len)
    #     spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
    #     wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
    #     text_prompt_padded = torch.LongTensor(len(batch), max_text_prompt_len)
    #     vision_prompt_padded = torch.FloatTensor(len(batch), 3, 224,224)
    #     audio_prompt_padded = spec_padded

    #     text_padded.zero_()
    #     spec_padded.zero_()
    #     wav_padded.zero_()
    #     text_prompt_padded.zero_()
    #     vision_prompt_padded.zero_()
    #     audio_prompt_padded.zero_()
        
    #     for i in range(len(ids_sorted_decreasing)):
    #         row = batch[ids_sorted_decreasing[i]]

    #         text = row[0]
    #         text_padded[i, :text.size(0)] = text
    #         text_lengths[i] = text.size(0)

    #         spec = row[1]
    #         spec_padded[i, :, :spec.size(1)] = spec
    #         spec_lengths[i] = spec.size(1)

    #         wav = row[2]
    #         wav_padded[i, :, :wav.size(1)] = wav
    #         wav_lengths[i] = wav.size(1)

    #         sid[i] = row[3]
            
    #         vision_prompt = row[5]
    #         vision_prompt_padded[i] = vision_prompt
            
    #         text_prompt = row[4]
    #         text_prompt_padded[i, :text_prompt.size(0)] = text_prompt
    #         text_prompt_lengths[i] = (text_prompt > 0).sum().item()
            
    #         audio_prompt = row[6]
    #         audio_prompt_padded[i, :, :audio_prompt.size(1)] = audio_prompt
    #         audio_prompt_lengths[i] = audio_prompt.size(1)
            

    #     if self.return_ids:
    #         return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, text_prompt_padded, vision_prompt_padded, audio_prompt_padded, ids_sorted_decreasing
    #     return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, text_prompt_padded, vision_prompt_padded, audio_prompt_padded


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
        
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1


    # def _bisect(self, x):
    #     """
    #     Assigns a sample length to the appropriate bucket index using bisect.
    #     """
    #     # Find the rightmost boundary less than or equal to x
    #     idx = bisect.bisect_right(self.boundaries, x) - 1
    #     # Clamp the index to ensure it's within valid range
    #     idx = max(0, min(idx, len(self.boundaries) - 2))
    #     return idx

  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          if len_bucket > 0:
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def __len__(self):
        return self.num_samples // self.batch_size
