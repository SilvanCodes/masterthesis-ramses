from gpn.data import load_fasta
import random
import json

chromosome = snakemake.config["CHROMOSOME"]
random_position_count = snakemake.config["RANDOM_POSITION_COUNT"]
max_context_length = snakemake.config["MAXIMUM_CONTEXT_LENGTH"]

sequence_path = snakemake.input[0]
genome = load_fasta(sequence_path)
sequence = genome[chromosome]

# treat as 1-indexed
gap = max_context_length // 2 + 1

random_positions = random.sample(range(gap, len(sequence) - gap), random_position_count)

with open(snakemake.output[0], 'w') as f:
    json.dump(random_positions, f)