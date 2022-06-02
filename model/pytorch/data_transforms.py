import torch
from alphafold.data.common import residue_constants

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
  shuffled = torch.random_shuffle(torch.range(1, num_seq))
  index_order = torch.cat([[0], shuffled], axis=0)
  num_sel = torch.minimum(max_seq, num_seq)

  sel_seq, not_sel_seq = torch.split(index_order, [num_sel, num_seq - num_sel])

  for k in _MSA_FEATURE_NAMES:
    if k in protein:
      if keep_extra:
        protein['extra_' + k] = torch.gather(protein[k], not_sel_seq)
      protein[k] = torch.gather(protein[k], sel_seq)

  return protein