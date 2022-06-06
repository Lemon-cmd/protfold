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
  weights = tf.concat([
      tf.ones(21),
      gap_agreement_weight * tf.ones(1),
      np.zeros(1)], 0)

  # Make agreement score as weighted Hamming distance
  sample_one_hot = (protein['msa_mask'][:, :, None] *
                    tf.one_hot(protein['msa'], 23))
  extra_one_hot = (protein['extra_msa_mask'][:, :, None] *
                   tf.one_hot(protein['extra_msa'], 23))

  num_seq, num_res, _ = shape_helpers.shape_list(sample_one_hot)
  extra_num_seq, _, _ = shape_helpers.shape_list(extra_one_hot)

  # Compute tf.einsum('mrc,nrc,c->mn', sample_one_hot, extra_one_hot, weights)
  # in an optimized fashion to avoid possible memory or computation blowup.
  agreement = tf.matmul(
      tf.reshape(extra_one_hot, [extra_num_seq, num_res * 23]),
      tf.reshape(sample_one_hot * weights, [num_seq, num_res * 23]),
      transpose_b=True)

  # Assign each sequence in the extra sequences to the closest MSA sample
  protein['extra_cluster_assignment'] = tf.argmax(
      agreement, axis=1, output_type=tf.int32)

  return protein


@curry1
def summarize_clusters(protein):
  """Produce profile and deletion_matrix_mean within each cluster."""
  num_seq = shape_helpers.shape_list(protein['msa'])[0]
  def csum(x):
    return tf.math.unsorted_segment_sum(
        x, protein['extra_cluster_assignment'], num_seq)

  mask = protein['extra_msa_mask']
  mask_counts = 1e-6 + protein['msa_mask'] + csum(mask)  # Include center

  msa_sum = csum(mask[:, :, None] * tf.one_hot(protein['extra_msa'], 23))
  msa_sum += tf.one_hot(protein['msa'], 23)  # Original sequence
  protein['cluster_profile'] = msa_sum / mask_counts[:, :, None]

  del msa_sum

  del_sum = csum(mask * protein['extra_deletion_matrix'])
  del_sum += protein['deletion_matrix']  # Original sequence
  protein['cluster_deletion_mean'] = del_sum / mask_counts
  del del_sum

  return protein
