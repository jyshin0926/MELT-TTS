import torch 
from torch.nn import functional as F
import commons

def emotion_consistency_loss(generated_emotion_emb, target_emotion_emb, loss_type='cosine'):
  if loss_type == 'cosine':
    loss = 1 - F.cosine_similarity(generated_emotion_emb, target_emotion_emb).mean()
  elif loss_type == 'L2':
    loss = F.mse_loss(generated_emotion_emb, target_emotion_emb)
  elif loss_type == 'KL':
    loss = F.kl_div(generated_emotion_emb.log_softmax(dim=-1), target_emotion_emb.softmax(dim=-1), reduction='batchmean')
  else:
    raise ValueError("Unsupported loss type. Use 'cosine', 'L2', or 'KL'.")
    
  return loss

# TODO:: add speaker rep model
def speaker_consistency_loss(generated_speaker_emb, target_speaker_emb, loss_type='cosine'):
  if loss_type == 'cosine':
    loss = 1 - F.cosine_similarity(generated_speaker_emb, target_speaker_emb).mean()
  elif loss_type == 'L2':
    loss = F.mse_loss(generated_speaker_emb, target_speaker_emb)
  elif loss_type == 'KL':
    loss = F.kl_div(generated_speaker_emb.log_softmax(dim=-1), target_speaker_emb.softmax(dim=-1), reduction='batchmean')
  else:
    raise ValueError("Unsupported loss type. Use 'cosine', 'L2', or 'KL'.")
    
  return loss


def feature_loss(fmap_r, fmap_g):
  loss = 0
  for dr, dg in zip(fmap_r, fmap_g):
    for rl, gl in zip(dr, dg):
      rl = rl.float().detach()
      gl = gl.float()
      loss += torch.mean(torch.abs(rl - gl))

  return loss * 2 


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
  loss = 0
  r_losses = []
  g_losses = []
  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    dr = dr.float()
    dg = dg.float()
    r_loss = torch.mean((1-dr)**2)
    g_loss = torch.mean(dg**2)
    loss += (r_loss + g_loss)
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())

  return loss, r_losses, g_losses


def generator_loss(disc_outputs):
  loss = 0
  gen_losses = []
  for dg in disc_outputs:
    dg = dg.float()
    l = torch.mean((1-dg)**2)
    gen_losses.append(l)
    loss += l

  return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
  """
  z_p, logs_q: [b, h, t_t]
  m_p, logs_p: [b, h, t_t]
  """
  z_p = z_p.float()
  logs_q = logs_q.float()
  m_p = m_p.float()
  logs_p = logs_p.float()
  z_mask = z_mask.float()

  kl = logs_p - logs_q - 0.5
  kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
  kl = torch.sum(kl * z_mask)
  l = kl / torch.sum(z_mask)
  return l
