import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import os
import commons
import modules
import attentions
import monotonic_align

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
import sys
sys.path.append(os.path.abspath('/workspace/jaeyoung/StoryTeller/ONE-PEACE'))
from one_peace.models import from_pretrained



# TODO:: DurationPredictor ?

class StochasticDurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
    super().__init__()
    filter_channels = in_channels # it needs to be removed from future version.
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.log_flow = modules.Log()
    self.flows = nn.ModuleList()
    self.flows.append(modules.ElementwiseAffine(2))
    for i in range(n_flows):
      self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.flows.append(modules.Flip())

    self.post_pre = nn.Conv1d(1, filter_channels, 1)
    self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    self.post_flows = nn.ModuleList()
    self.post_flows.append(modules.ElementwiseAffine(2))
    for i in range(4):
      self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.post_flows.append(modules.Flip())

    self.pre = nn.Conv1d(in_channels, filter_channels, 1)
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

  def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
    x = torch.detach(x)
    x = self.pre(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    x = self.convs(x, x_mask)
    x = self.proj(x) * x_mask

    if not reverse:
      flows = self.flows
      assert w is not None

      logdet_tot_q = 0 
      h_w = self.post_pre(w)
      h_w = self.post_convs(h_w, x_mask)
      h_w = self.post_proj(h_w) * x_mask
      e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
      z_q = e_q
      for flow in self.post_flows:
        z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
        logdet_tot_q += logdet_q
      z_u, z1 = torch.split(z_q, [1, 1], 1) 
      u = torch.sigmoid(z_u) * x_mask
      z0 = (w - u) * x_mask
      logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1,2])
      logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q

      logdet_tot = 0
      z0, logdet = self.log_flow(z0, x_mask)
      logdet_tot += logdet
      z = torch.cat([z0, z1], 1)
      for flow in flows:
        z, logdet = flow(z, x_mask, g=x, reverse=reverse)
        logdet_tot = logdet_tot + logdet
      nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
      return nll + logq # [b]
    else:
      flows = list(reversed(self.flows))
      flows = flows[:-2] + [flows[-1]] # remove a useless vflow
      z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
      for flow in flows:
        z = flow(z, x_mask, g=x, reverse=reverse)
      z0, z1 = torch.split(z, [1, 1], 1)
      logw = z0
      return logw


class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = modules.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = modules.LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, 1, 1)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, in_channels, 1)

  def forward(self, x, x_mask, g=None):
    x = torch.detach(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    x = self.proj(x * x_mask)
    return x * x_mask

class EmotionEncoder(nn.Module):
  def __init__(self,vision_model_path, audio_model_path):

    super(EmotionEncoder, self).__init__()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    self.vision_model = from_pretrained(
        model_name_or_path="/workspace/jaeyoung/checkpoints/onepeace/mmtts_vl_1013/checkpoint_last.pt",
        model_type="one_peace_retrieval",
        device=device,
        dtype="float16"
    )
    
    self.audio_model = from_pretrained(
        model_name_or_path="/workspace/jaeyoung/checkpoints/onepeace/esd_mmtts_al_1014/checkpoint_last.pt",
        model_type="one_peace_retrieval",
        device=device,
        dtype="float16"
    )
    
  def forward(self, text_prompt=None, vision_prompt=None, audio_prompt=None):
      """
      Extracts and combines features from text and vision prompts.

      Args:
      text_prompt (str): Text description, e.g., "She says with her beautiful smile."
      vision_prompt (str): Path to the face image file, e.g., "image1.jpg".
      audio_prompt (str, optional): Path to the audio file, e.g., "audio1.wav".

      Returns:
      torch.Tensor: Combined emotion embedding.
      """      
      text_batch_size = 50   # Adjust based on your GPU capacity
      image_batch_size = 10  
      audio_batch_size = 10  

      with torch.no_grad():
        if text_prompt is not None:
          all_text_features = []
          for i in range(0, len(text_prompt), text_batch_size):
            batch_text_queries = text_prompt[i:i + text_batch_size]
            text_tokens = self.audio_model.process_text(batch_text_queries)
            batch_text_features = self.audio_model.extract_text_features(text_tokens)
            all_text_features.append(batch_text_features)
          text_features = torch.cat(all_text_features, dim=0)
        else:
          text_features = None
        
        
        if vision_prompt is not None:
          all_image_features = []
          for i in range(0, len(vision_prompt), image_batch_size):
              batch_image_list = vision_prompt[i:i + image_batch_size]
              src_images = self.vision_model.process_image(batch_image_list)
              batch_image_features = self.vision_model.extract_image_features(src_images)
              all_image_features.append(batch_image_features)
          vision_features = torch.cat(all_image_features, dim=0)        
        else:
          vision_features = None
        
        if audio_prompt is not None:
          all_audio_features = []
          for i in range(0, len(audio_prompt), audio_batch_size):
              batch_audio_list = audio_prompt[i:i + audio_batch_size]
              src_audios, audio_padding_masks = self.audio_model.process_audio(batch_audio_list)
              batch_audio_features = self.audio_model.extract_audio_features(src_audios, audio_padding_masks)
              all_audio_features.append(batch_audio_features)
          audio_features = torch.cat(all_audio_features, dim=0)
        else:
          audio_features = None
          
      # combine features (text and vision prompts) (feature fusion - attention network?)
      if text_features is not None and vision_features is not None and audio_features is not None:
        emotion_emb = torch.cat([text_features, vision_features, audio_features], dim=-1)
      elif text_features is not None:
        emotion_emb = text_features
      elif vision_features is not None:
        emotion_emb = vision_features
      elif audio_features is not None:
        emotion_emb = audio_features
      else:
        raise ValueError("text_prompt, vision_prompt or audio_prompt must be provided") 
      # return torch.zeros(1, desired_dim).to(audio_prompt.device) # TODO:: desired_dim 설정

      return emotion_emb

class EmotionClassifierModule(nn.Module):
  def __init__(self, emotion_classes, input_size):
    super(EmotionClassifierModule, self).__init__()
    self.emotion_classes = emotion_classes
    self.classifier = nn.Linear(input_size, len(emotion_classes)) # input size depends on the emotion_emb
    
  def forward(self, emotion_emb):
    emotion_logits = self.classifier(emotion_emb)
    emotion_probs = F.softmax(emotion_logits, dim=-1)
    emotion_dicts = []
    for prob in emotion_probs:
      emotion_dict = {self.emotion_classes[i]: prob[i].item()
                      for i in range(len(self.emotion_classes))}
      emotion_dicts.append(emotion_dict)
    
    return emotion_dicts
    

class EmotionIntensityModule(nn.Module):
  def __init__(self, emotion_classes, min_intensity=0.0, max_intensity=1.0):
    super(EmotionIntensityModule, self).__init__()
    self.min_intensity = min_intensity
    self.max_intensity = max_intensity
    self.emotion_classes = emotion_classes
    
    # TODO:: embedding size setting
    self.emotion_weights = nn.ModuleDict({emotion: nn.Linear(1536, 1536) for emotion in self.emotion_classes}) # size demends on the emotion_emb (Assuming enc size as 768*2)
    
  def forward(self, emotion_emb, emotion_dict, intensitiy_val=None):
    combined_emotion_emb = torch.zeros_like(emotion_emb)
    for emotion, prob in emotion_dict.items():
      emotion_emb_mod = self.emotion_weights[emotion](emotion_emb)
      weighted_emotion_emb = emotion_emb_mod * prob
      combined_emotion_emb += weighted_emotion_emb
    
    if intensity_val is not None:
      intensity_val = torch.clamp(intensity_val, self.min_intensity, self.max_intensity)
    else:
      intensity_val = sum([p for p in emotion_dict.values()])/len(emotion_dict)
    
    modulated_emotion_emb = combined_emotion_emb * intensity_val
    
    return modulated_emotion_emb

# emotion intensity (EmoQ-TTS) / Residual Vector Quantization
class IntensityPredictor(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(IntensityPredictor, self).__init__()
    self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
    )
    
    def forward(self, x):
      return self.net(x)
    
class IntensityQuantizer(nn.Module):
  def __init__(self, levels):
    super(IntensityQuantizer, self).__init__()
    self.levels = levels
    
  def forward(self, intensities):
    min_val, max_val = intensities.min(), intensities.max()
    normalized_intensities = (intensities - min_val) / (max_val - min_val)
    
    scaled_intensities = normalized_intensities * (self.levels - 1)
    quantized_intensities = torch.round(scaled_intensities).long()
    return quantized_intensities
    
class IntensityModule(nn.Module):
  def __init__(self, input_dim, num_intensity_levels):
    super(IntensityModule, self).__init__()
    self.intensity_predictor = IntensityPredictor(input_dim, 1)
    self.intensity_quantizer = IntensityQuantizer(num_intensity_levels, input_dim)
    self.intensity_embeddings = nn.Embedding(num_intensity_levels, input_dim)
    
  def forward(self, phoneme_features):
    raw_intensities = self.intensity_predictor(phoneme_features)
    quantized_intensities = self.intensity_quantizer(raw_intensities)
    intensity_embeddings = self.intensity_embeddings(quantized_intensities)
    
    enhanced_features = phoneme_features + intensity_embeddings
    return enhanced_features
  
# TODO:: AVDEncoder (EmoSphere-TTS; Arousal, Valence, Dominance)


class TextEncoder(nn.Module):
  def __init__(self,
      n_vocab,
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout):
    super().__init__()
    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.emb = nn.Embedding(n_vocab, hidden_channels)
    nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
    self.proj= nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths):
    x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
    x = torch.transpose(x, 1, -1) # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    x = self.encoder(x * x_mask, x_mask)
    stats = self.proj(x) * x_mask

    m, logs = torch.split(stats, self.out_channels, dim=1)
    return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
      self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x


class PosteriorEncoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
    
    if gin_channels != 0:
      self.cond_speaker = nn.Conv1d(gin_channels, hidden_channels, 1)
      self.cond_emotion = nn.Conv1d(gin_channels, hidden_channels, 1)

  def forward(self, x, x_lengths, g=None):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x) * x_mask
    
    # TODO:: separately handling speaker and emotion
    if g is not None:
      if 'speaker' in g:
        x = x + self.cond_speker(g['speaker'])
      if 'emotion' in g:
        x = x + self.cond_emotion(g['emotion'])
    
    x = self.enc(x, x_mask, g=g)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
          x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat): 
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class SynthesizerTrn(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self, 
    n_vocab,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    n_speakers,
    n_emotions, 
    vision_model_path,
    audio_model_path,
    emotion_classes,
    gin_channels=0,
    use_sdp=True,
    **kwargs):

    super().__init__()
    self.n_vocab = n_vocab
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.n_speakers = n_speakers
    self.n_emotions = n_emotions
    self.gin_channels = gin_channels
    self.vision_model_path = vision_model_path
    self.audio_model_path = audio_model_path
    self.emotion_classes = emotion_classes

    self.use_sdp = use_sdp

    self.enc_p = TextEncoder(n_vocab,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout)

    # self.emotion_enc = EmotionEncoder(vision_model_path=vision_model_path, audio_model_path=audio_model_path)
    # self.emotion_classifier = EmotionClassifierModule(emotion_classes=emotion_classes, input_size=768*2)
    # self.emotion_intensity = EmotionIntensityModule(emotion_classes=emotion_classes, min_intensity=0.0, max_intensity=1.0)
    # self.emotion_enc_target = self.emotion_enc

    self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
    self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
    self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

    if use_sdp:
      self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
    else:
      self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

    if n_speakers > 1:
      self.emb_g = nn.Embedding(n_speakers, gin_channels)

  def forward(self, x, x_lengths, y, y_lengths, sid=None, text_prompt=None, vision_prompt=None, audio_prompt=None, eid=None):

    x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
    
    # emotion_emb = self.emotion_enc(text_prompt=text_prompt, vision_prompt=vision_prompt, audio_prompt=audio_prompt)
    # emotion_dict = self.emotion_classifier(emotion_emb)
    # modulated_emotion_emb = self.emotion_intensity(emotion_emb, emotion_dict)
    
    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    # if g is not None:
    #   modulated_emotion_emb = torch.cat([modulated_emotion_emb, g],dim=-1)
      
    # tgt_emo_emb = self.emotion_enc_target(audio_prompt=y) # y = wav
    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
    z_p = self.flow(z, y_mask, g=g)

    with torch.no_grad():
      # negative cross-entropy
      s_p_sq_r = torch.exp(-2 * logs_p) # [b, d, t]
      neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True) # [b, 1, t_s]
      neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
      neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r)) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
      neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True) # [b, 1, t_s]
      neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

      attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
      attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

    w = attn.sum(2)
    if self.use_sdp:
      l_length = self.dp(x, x_mask, w, g=g)
      l_length = l_length / torch.sum(x_mask)
    else:
      logw_ = torch.log(w + 1e-6) * x_mask
      logw = self.dp(x, x_mask, g=g)
      l_length = torch.sum((logw - logw_)**2, [1,2]) / torch.sum(x_mask) # for averaging 

    # expand prior
    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

    z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
    o = self.dec(z_slice, g=g)
    return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q), (None, None)  # TODO:: Emotion Embeddings 별도 처리

  def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None, vision_prompt=None, audio_prompt=None, eid=None):
    x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
    
    # emotion_emb = self.emotion_enc(text_prompt=x, vision_prompt=vision_prompt, audio_prompt=audio_prompt)
    # emotion_dict = self.emotion_classifier(emotion_emb)
    # modulated_emotion_emb = self.emotion_intensity(emotion_emb, emotion_dict)
    
    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
    else:
      g = None
    # if g is not None:
      # modulated_emotion_emb = torch.cat([modulated_emotion_emb, g], dim=-1)
    if self.use_sdp:
      logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
    else:
      logw = self.dp(x, x_mask, g=g)
    w = torch.exp(logw) * x_mask * length_scale
    w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil, attn_mask)

    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']

    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = self.flow(z_p, y_mask, g=g, reverse=True)
    o = self.dec((z * y_mask)[:,:,:max_len], g=g)
    return o, attn, y_mask, (z, z_p, m_p, logs_p)

  def voice_conversion(self, y, y_lengths, sid_src, sid_tgt, text_prompt=None, vision_prompt=None, audio_prompt=None, eid=None):
    assert self.n_speakers > 0, "n_speakers have to be larger than 0."
    g_src = self.emb_g(sid_src).unsqueeze(-1)
    g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)

    # emotion_emb_src = self.emotion_enc(text_prompt=text_prompt, vision_prompt=vision_prompt, audio_prompt=audio_prompt)
    # emotion_dict_src = self.emotion_classifier(emotion_emb_src)
    # modulated_emotion_emb_src = self.emotion_intensity(emotion_emb_src, emotion_dict_src)
    # modulated_emotion_emb_src = torch.cat([modulated_emotion_emb_src, g_src], dim=-1)

    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
    z_p = self.flow(z, y_mask, g=g_src)
    
    # emotion_emb_tgt = self.emotion_enc(text_prompt=text_prompt, vision_prompt=vision_prompt, audio_prompt=audio_prompt)
    # emotion_dict_tgt = self.emotion_classifier(emotion_emb_tgt)
    # modulated_emotion_emb_tgt = self.emotion_intensity(emotion_emb_tgt, emotion_dict_tgt)
    # modulated_emotion_emb_tgt = torch.cat([modulated_emotion_emb_tgt, g_tgt], dim=-1)
    
    z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
    o_hat = self.dec(z_hat * y_mask, g=g_tgt)
    return o_hat, y_mask, (z, z_p, z_hat)

