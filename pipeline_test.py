from data.templates import HhsearchHitFeaturizer
from data.pipeline import DataPipeline
from data.pipeline_multimer import DataPipeline as DataPipelineM
from data.tools import hhsearch
from data import feature_processing as fprocess
from model import modules_multimer
import time, torch

tsearch = HhsearchHitFeaturizer(
                            mmcif_dir = "/gpfs/u/home/HPDM/HPDMphmb/scratch-shared/pdb/pdb_mmcif/mmcif_files",
                            max_template_date = "2025-12-01",
                            max_hits = 10,
                            kalign_binary_path = "kalign", release_dates_path = "", obsolete_pdbs_path = "")

pipeline = DataPipeline( jackhmmer_binary_path = "jackhmmer",
                         hhblits_binary_path = "hhblits",
                         uniref90_database_path = "/gpfs/u/home/HPDM/HPDMphmb/scratch-shared/pdb/uniref90/uniref90.fasta",
                         mgnify_database_path = "/gpfs/u/home/HPDM/HPDMphmb/scratch-shared/pdb/mgnify/mgy_clusters_2018_12.fa",
                         bfd_database_path = "/gpfs/u/home/HPDM/HPDMphmb/scratch-shared/pdb/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
                         #bfd_database_path = "/gpfs/u/home/HPDM/HPDMphmb/scratch-shared/pdb/UniRef30_2021_03/UniRef30_2021_03",
                         uniclust30_database_path = "/gpfs/u/home/HPDM/HPDMphmb/scratch-shared/pdb/UniRef30_2021_03/UniRef30_2021_03",
                         template_searcher = hhsearch.HHSearch(binary_path = "hhsearch", databases=["/gpfs/u/home/HPDM/HPDMphmb/scratch-shared/pdb/pdb70/pdb70"]),
                         template_featurizer = tsearch,
                         small_bfd_database_path = "/gpfs/u/home/HPDM/HPDMphmb/scratch-shared/pdb/bfd_tiny/bfd-first_non_consensus_sequences.fasta",
                         use_small_bfd = False,
                         mgnify_max_hits = 11,
                         uniref_max_hits = 10, use_precomputed_msas = True)

pipeline_mult = DataPipelineM(pipeline, jackhmmer_binary_path = "jackhmmer", 
                              uniprot_database_path = "/gpfs/u/home/HPDM/HPDMphmb/scratch-shared/pdb/uniprot/uniprot_2021.fasta",
                              max_uniprot_hits = 2)

s = time.time()
np_example = pipeline_mult.process("/gpfs/u/home/HPDM/HPDMphmb/scratch-shared/pipeline_test/in.ffdata", "/gpfs/u/home/HPDM/HPDMphmb/scratch-shared/pipeline_test/")
e = time.time()

for key in np_example.keys():
    print(key, np_example[key].shape)
    
    if key != 'num_alignments' or key != 'seq_length':
        np_example[key] = torch.Tensor(np_example[key])
    else:
        print(np_example[key])

print(e - s, " s")

from model.pytorch import data_transforms

np_example = data_transforms.sample_msa(16, keep_extra=True)(np_example)

for key in np_example.keys():
    print(key, np_example[key].shape)