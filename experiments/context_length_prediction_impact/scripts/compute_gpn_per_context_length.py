import helpers
import pandas as pd
from gpn.data import load_fasta
from snakemake.script import snakemake

chromosome = snakemake.wildcards.chromosome

# load sequence
sequence_path = snakemake.input[0]
genome = load_fasta(sequence_path)
sequence = genome[chromosome]

probabilities_per_context_length_df_path = snakemake.input[1]

position = int(probabilities_per_context_length_df_path.split("/")[-2])

results = pd.read_parquet(probabilities_per_context_length_df_path)

# arrays are zero indexed, genomes not
reference_nucleotide = sequence[position - 1]

# skip position when reference is unknown
# if reference_nucleotide in ["n", "N"]:
#     continue

gpn_scores = helpers.compute_gpn_score(reference_nucleotide, results)


gpn_scores.to_parquet(snakemake.output[0])
