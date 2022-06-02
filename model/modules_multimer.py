import torch
import torch.nn.functional as F

import functools
import numpy as np
from typing import Sequence

def reduce_fn(x, mode):
  if mode == 'none' or mode is None:
    return torch.Tensor(x)
  elif mode == 'sum':
    return torch.sum(x)
  elif mode == 'mean':
    return torch.mean(x)
  else:
    raise ValueError('Unsupported reduction option.')
    

def gumbel_noise(shape: Sequence[int]) -> torch.Tensor:
  """Generate Gumbel Noise of given Shape.
  This generates samples from Gumbel(0, 1).
  Args:
    shape: Shape of noise to return.
  Returns:
    Gumbel noise of given shape.
  """
  epsilon = 1e-6
  uniform_noise = torch.Tensor(shape).uniform_(0, 1)
  gumbel = -torch.log(-torch.log(uniform_noise + epsilon) + epsilon)
  return gumbel


def gumbel_max_sample(logits: torch.Tensor) -> torch.Tensor:
  """Samples from a probability distribution given by 'logits'.
  This uses Gumbel-max trick to implement the sampling in an efficient manner.
  Args:
    logits: Logarithm of probabilities to sample from, probabilities can be
      unnormalized.
  Returns:
    Sample from logprobs in one-hot form.
  """
  z = gumbel_noise(logits.shape)
  return F.one_hot(
      torch.argmax(logits + z, dim=-1),
      logits.shape[-1]).float()


def gumbel_argsort_sample_idx(logits: torch.Tensor) -> torch.Tensor:
  """Samples with replacement from a distribution given by 'logits'.
  This uses Gumbel trick to implement the sampling an efficient manner. For a
  distribution over k items this samples k times without replacement, so this
  is effectively sampling a random permutation with probabilities over the
  permutations derived from the logprobs.
  Args:
    logits: Logarithm of probabilities to sample from, probabilities can be
      unnormalized.
  Returns:
    Sample from logprobs in one-hot form.
  """
  z = gumbel_noise(logits.shape)
  _, perm = torch.sort(logits + z, -1)
  return perm[::-1]


def nearest_neighbor_clusters(batch, gap_agreement_weight=0.):
  """Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""

  # Determine how much weight we assign to each agreement.  In theory, we could
  # use a full blosum matrix here, but right now let's just down-weight gap
  # agreement because it could be spurious.
  # Never put weight on agreeing on BERT mask.

  weights = torch.FloatTensor([1.] * 21 + [gap_agreement_weight] + [0.])

  msa_mask = batch['msa_mask']
  msa_one_hot = F.one_hot(batch['msa'], 23)

  extra_mask = batch['extra_msa_mask']
  extra_one_hot = F.one_hot(batch['extra_msa'], 23)

  msa_one_hot_masked = msa_mask[:, :, None] * msa_one_hot
  extra_one_hot_masked = extra_mask[:, :, None] * extra_one_hot

  agreement = torch.einsum('mrc, nrc->nm', extra_one_hot_masked, 
                           weights * msa_one_hot_masked)

  cluster_assignment = torch.softmax(1e3 * agreement, dim=0)
  cluster_assignment *= torch.einsum('mr, nr->mn', msa_mask, 
                                     extra_mask)

  cluster_count = torch.sum(cluster_assignment, dim=-1) + 1. # We always include the sequence itself.

  msa_sum = torch.einsum('nm, mrc->nrc', cluster_assignment, 
                         extra_one_hot_masked)
  msa_sum += msa_one_hot_masked

  cluster_profile = msa_sum / cluster_count[:, None, None]

  extra_deletion_matrix = batch['extra_deletion_matrix']
  deletion_matrix = batch['deletion_matrix']

  del_sum = torch.einsum('nm, mc->nc', cluster_assignment,
                       extra_mask * extra_deletion_matrix)
    
  del_sum += deletion_matrix  # Original sequence.
  cluster_deletion_mean = del_sum / cluster_count[:, None]

  return cluster_profile, cluster_deletion_mean


def create_msa_feat(batch):
  """Create and concatenate MSA features."""
  msa_1hot = F.one_hot(batch['msa'], 23)
    
  deletion_matrix = batch['deletion_matrix']
  has_deletion = torch.clip(deletion_matrix, 0., 1.)[..., None]
  deletion_value = (torch.arctan(deletion_matrix / 3.) * (2. / torch.pi))[..., None]
  deletion_mean_value = (torch.arctan(batch['cluster_deletion_mean'] / 3.) * (2. / jnp.pi))[..., None]

  msa_feat = [
      msa_1hot,
      has_deletion,
      deletion_value,
      batch['cluster_profile'],
      deletion_mean_value
  ]

  return torch.cat(msa_feat, dim=-1)


def create_extra_msa_feature(batch, num_extra_msa):
  """Expand extra_msa into 1hot and concat with other extra msa features.
  We do this as late as possible as the one_hot extra msa can be very large.
  Args:
    batch: a dictionary with the following keys:
     * 'extra_msa': [num_seq, num_res] MSA that wasn't selected as a cluster
       centre. Note - This isn't one-hotted.
     * 'extra_deletion_matrix': [num_seq, num_res] Number of deletions at given
        position.
    num_extra_msa: Number of extra msa to use.
  Returns:
    Concatenated tensor of extra MSA features.
  """
  # 23 = 20 amino acids + 'X' for unknown + gap + bert mask
  extra_msa = batch['extra_msa'][:num_extra_msa]
  deletion_matrix = batch['extra_deletion_matrix'][:num_extra_msa]
  msa_1hot = F.one_hot(extra_msa, 23)
  has_deletion = torch.clip(deletion_matrix, 0., 1.)[..., None]
  deletion_value = (torch.arctan(deletion_matrix / 3.) * (2. / jnp.pi))[..., None]
  extra_msa_mask = batch['extra_msa_mask'][:num_extra_msa]

  return torch.cat([msa_1hot, has_deletion, deletion_value], dim=-1), extra_msa_mask