import torch_scatter
import torch.nn.functional as F
import torch, functools, numpy as np
from data.common import residue_constants
from model.pytorch import shape_placeholders

NUM_RES = shape_placeholders.NUM_RES
NUM_MSA_SEQ = shape_placeholders.NUM_MSA_SEQ
NUM_EXTRA_SEQ = shape_placeholders.NUM_EXTRA_SEQ
NUM_TEMPLATES = shape_placeholders.NUM_TEMPLATES

_MSA_FEATURE_NAMES = [
    'msa', 'deletion_matrix', 'msa_mask', 'msa_row_mask', 'bert_mask',
    'true_msa'
]

def delete_extra_msa(protein):
  for k in _MSA_FEATURE_NAMES:
    if 'extra_' + k in protein:
      del protein['extra_' + k]
  return protein

def curry1(f):
  """Supply all arguments but the first."""

  def fc(*args, **kwargs):
    return lambda x: f(x, *args, **kwargs)

  return fc


@curry1
def sample_msa(protein, max_seq, keep_extra):
  """Sample MSA randomly, remaining sequences are stored as `extra_*`.
  Args:
    protein: batch to sample msa from.
    max_seq: number of sequences to sample.
    keep_extra: When True sequences not sampled are put into fields starting
      with 'extra_*'.
  Returns:
    Protein with sampled msa.
  """
  num_seq = (protein['msa']).shape[0]
  shuffled = torch.arange(1, num_seq)[torch.randperm(num_seq - 1)]
  index_order = torch.cat([torch.zeros(1), shuffled], dim = 0)
  num_sel = min(max_seq, num_seq)

  sel_seq, not_sel_seq = torch.split(index_order, [num_sel, num_seq - num_sel])
  sel_seq, not_sel_seq = sel_seq.long(), not_sel_seq.long()
    
  for k in _MSA_FEATURE_NAMES:
    if k in protein:
      if keep_extra:
        protein['extra_' + k] = protein[k][not_sel_seq]
    
      protein[k] = protein[k][sel_seq]

  return protein


@curry1
def block_delete_msa(protein, msa_fraction_per_block: float, num_blocks: int, randomize_num_blocks: bool = False):
  """Sample MSA by deleting contiguous blocks.
  Jumper et al. (2021) Suppl. Alg. 1 "MSABlockDeletion"
  Arguments:
    protein: batch dict containing the msa
    config: ConfigDict with parameters
  Returns:
    updated protein
  """
  assert(msa_fraction_per_block > 0.0 and msa_fraction_per_block < 1.0)
  num_seq = shape_helpers.shape_list(protein['msa'])[0]
  block_num_seq = int(num_seq * msa_fraction_per_block)

  if randomize_num_blocks:
    num_blocks = torch.randint(1, num_blocks + 1, size=[1]).item()
    del_block_starts = torch.randint(0, num_seq, size=[num_blocks])
  else:
    del_block_starts = torch.randint(0, num_seq, size=[num_blocks])
    
  del_blocks = del_block_starts[:, None] + torch.arange(block_num_seq)
  del_blocks.clip_(0, num_seq - 1)
    
  del_indices = torch.unique(torch.sort(del_blocks.ravel())[0])

  # Make sure we keep the original sequence
  sparse_diff = torch.from_numpy(np.setdiff1d(torch.arange(1, num_seq).numpy(), 
                                              del_indices.numpy()))

  keep_indices = torch.cat([torch.zeros(1), keep_indices], dim=0)

  for k in _MSA_FEATURE_NAMES:
    if k in protein:
      protein[k] = torch.gather(protein[k], keep_indices)

  return protein


@curry1
def nearest_neighbor_clusters(protein, gap_agreement_weight=0.):
  """Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""

  # Determine how much weight we assign to each agreement.  In theory, we could
  # use a full blosum matrix here, but right now let's just down-weight gap
  # agreement because it could be spurious.
  # Never put weight on agreeing on BERT mask
  weights = torch.cat([
      torch.ones(21),
      gap_agreement_weight * torch.ones(1),
      torch.zeros(1)], 0)
    
  protein['msa'] = protein['msa'].long()
  protein['extra_msa'] = protein['extra_msa'].long()
    
  # Make agreement score as weighted Hamming distance
  sample_one_hot = (protein['msa_mask'][:, :, None] *
                    F.one_hot(protein['msa'], 23))
  extra_one_hot = (protein['extra_msa_mask'][:, :, None] *
                   F.one_hot(protein['extra_msa'], 23))

  num_seq, num_res, _ = sample_one_hot.shape
  extra_num_seq, _, _ = extra_one_hot.shape

  # Compute tf.einsum('mrc,nrc,c->mn', sample_one_hot, extra_one_hot, weights)
  # in an optimized fashion to avoid possible memory or computation blowup.
  agreement = torch.matmul(
      torch.reshape(extra_one_hot, [extra_num_seq, num_res * 23]),
      torch.reshape(sample_one_hot * weights, [num_res * 23, num_seq]))

  # Assign each sequence in the extra sequences to the closest MSA sample
  protein['extra_cluster_assignment'] = torch.argmax(agreement, dim=1)

  return protein


@curry1
def summarize_clusters(protein):
  """Produce profile and deletion_matrix_mean within each cluster."""
  num_seq = protein['msa'].size(0)
    
  def csum(x, idx):    
    return torch.zeros(num_seq, *x.shape[1:]).float().scatter_add_(0, idx, x)

  mask = protein['extra_msa_mask']
  mask_counts = 1e-6 + protein['msa_mask'] + csum(
      mask, protein['extra_cluster_assignment'].unsqueeze(-1).repeat(1, mask.size(1)))   # Include center

  msa_sum = csum(mask.unsqueeze(-1) * F.one_hot(protein['extra_msa'], 23), 
                 protein['extra_cluster_assignment'].unsqueeze(-1).unsqueeze(-1).repeat(1, mask.size(1), 23))
    
  msa_sum += F.one_hot(protein['msa'], 23)  # Original sequence
  protein['cluster_profile'] = msa_sum / mask_counts[:, :, None]

  del msa_sum

  del_sum = csum(mask * protein['extra_deletion_matrix'],
                 protein['extra_cluster_assignment'].unsqueeze(-1).repeat(1, mask.size(1)))
    
  del_sum += protein['deletion_matrix']  # Original sequence
  protein['cluster_deletion_mean'] = del_sum / mask_counts
    
  del del_sum
    
  return protein
 

def make_msa_mask(protein):
  """Mask features are all ones, but will later be zero-padded."""
  protein['msa_mask'] = torch.ones_like(protein['msa'])
  protein['msa_row_mask'] = torch.ones_like(protein['msa'][0])
  return protein


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
  """Create pseudo beta features."""
  is_gly = aatype == residue_constants.restype_order['G']
  ca_idx = residue_constants.atom_order['CA']
  cb_idx = residue_constants.atom_order['CB']
  pseudo_beta = torch.where(
      torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
      all_atom_positions[..., ca_idx, :],
      all_atom_positions[..., cb_idx, :])

  if all_atom_masks is not None:
    pseudo_beta_mask = torch.where(
        is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx]).float()
    return pseudo_beta, pseudo_beta_mask
  else:
    return pseudo_beta


@curry1
def make_pseudo_beta(protein, prefix=''):
  """Create pseudo-beta (alpha for glycine) position and mask."""
  assert prefix in ['', 'template_']
  protein[prefix + 'pseudo_beta'], protein[prefix + 'pseudo_beta_mask'] = (
      pseudo_beta_fn(
          protein['template_aatype' if prefix else 'aatype'],
          protein[prefix + 'all_atom_positions'],
          protein['template_all_atom_masks' if prefix else 'all_atom_mask']))
  return protein


def make_hhblits_profile(protein):
  """Compute the HHblits MSA profile if not already present."""
  if 'hhblits_profile' in protein:
    return protein

  # Compute the profile for every residue (over all MSA sequences).
  protein['hhblits_profile'] = torch.mean(
      F.one_hot(protein['msa'], 22).float(), dim=0)
  return protein


def shaped_categorical(probs, epsilon=1e-10):
  ds = probs.shape
  num_classes = ds[-1]

  counts = torch.distributions.categorical.Categorical(
      probs=probs.reshape(-1, num_classes)
  ).sample()

  return counts.reshape(ds[:-1])


@curry1
def make_masked_msa(protein, replace_fraction=0.15, uniform_prob=0.1, profile_prob=0.1, same_prob=0.1):
  """Create data for BERT on raw MSA."""
  # Add a random amino acid uniformly
  random_aa = torch.Tensor([0.05] * 20 + [0., 0.])

  categorical_probs = (
      uniform_prob * random_aa +
      profile_prob * protein['hhblits_profile'] +
      same_prob * F.one_hot(protein['msa'], 22))

  # Put all remaining probability on [MASK] which is a new column
  mask_prob = 1. - profile_prob - same_prob - uniform_prob
  assert mask_prob >= 0.
 
  categorical_probs = F.pad(
      categorical_probs, (0, 1), value=mask_prob)

  sh = protein['msa'].shape
  mask_position = torch.rand(sh) < replace_fraction

  bert_msa = shaped_categorical(categorical_probs)
  bert_msa = torch.where(mask_position, bert_msa, protein['msa'])

  # Mix real and masked MSA
  protein['bert_mask'] = mask_position.float()
  protein['true_msa'] = protein['msa']
  protein['msa'] = bert_msa
  
  del protein['hhblits_profile']

  return protein


@curry1
def make_msa_feat(protein):
  """Create and concatenate MSA features."""
  # Whether there is a domain break. Always zero for chains, but keeping
  # for compatibility with domain datasets.
  has_break = torch.clip(
      protein['between_segment_residues'].float(), 0., 1.)
    
  aatype_1hot = F.one_hot(protein['aatype'].long(), 21)

  target_feat = [
      has_break.unsqueeze(-1),
      aatype_1hot,  # Everyone gets the original sequence.
  ]

  msa_1hot = F.one_hot(protein['msa'].long(), 23)
  has_deletion = torch.clip(protein['deletion_matrix'], 0., 1.)
  deletion_value = torch.atan(protein['deletion_matrix'] / 3.) * (2. / np.pi)

  msa_feat = [
      msa_1hot,
      has_deletion.unsqueeze(-1),
      deletion_value.unsqueeze(-1),
  ]

  if 'cluster_profile' in protein:
    deletion_mean_value = (
        torch.atan(protein['cluster_deletion_mean'] / 3.) * (2. / np.pi))
    msa_feat.extend([
        protein['cluster_profile'],
        deletion_mean_value.unsqueeze(-1),
    ])

  if 'extra_deletion_matrix' in protein:
    protein['extra_has_deletion'] = torch.clip(
        protein['extra_deletion_matrix'], 0., 1.)
    protein['extra_deletion_value'] = torch.atan(
        protein['extra_deletion_matrix'] / 3.) * (2. / np.pi)

  protein['msa_feat'] = torch.cat(msa_feat, dim=-1)
  protein['target_feat'] = torch.cat(target_feat, dim=-1)
  return protein


@curry1
def select_feat(protein, feature_list):
  return {k: v for k, v in protein.items() if k in feature_list}