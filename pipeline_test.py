from data.templates import HhsearchHitFeaturizer
from data.pipeline import DataPipeline
from data.tools import hhsearch

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
                         uniclust30_database_path = "/gpfs/u/home/HPDM/HPDMphmb/scratch-shared/pdb/UniRef30_2021_03/UniRef30_2021_03",
                         template_searcher = hhsearch.HHSearch(binary_path = "hhsearch", databases=["/gpfs/u/home/HPDM/HPDMphmb/scratch-shared/pdb/pdb70/pdb70"]),
                         template_featurizer = tsearch,
                         small_bfd_database_path = "/gpfs/u/home/HPDM/HPDMphmb/scratch-shared/pdb/bfd_tiny/bfd-first_non_consensus_sequences.fasta",
                         use_small_bfd = True,
                         mgnify_max_hits = 501,
                         uniref_max_hits = 10000)

fdict = pipeline.process("/gpfs/u/home/HPDM/HPDMphmb/scratch-shared/pipeline_test/in.ffdata", 
                         "/gpfs/u/home/HPDM/HPDMphmb/scratch-shared/pipeline_test/")

print(fdict)

for key in fdict.keys():
    print(fdict[key].shape)