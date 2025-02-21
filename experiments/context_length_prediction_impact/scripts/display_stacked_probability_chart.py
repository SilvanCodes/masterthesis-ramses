import helpers
import pandas as pd

from gpn.data import load_fasta

# load probabilities_per_context_length
probabilities_per_context_length_path = snakemake.input[1]
probabilities_per_context_length = pd.read_parquet(
    probabilities_per_context_length_path
)

print(probabilities_per_context_length.shape)

# load config
chromosome = snakemake.config["CHROMOSOME"]
masked_position = snakemake.config["MASKED_POSITION"]

# load sequence
sequence_path = snakemake.input[0]
genome = load_fasta(sequence_path)
sequence = genome[chromosome]

title = f"Influence of context length (chr: {chromosome}, pos: {masked_position}, ref: {sequence[masked_position - 1]})"

helpers.plot_stacked_area(
    probabilities_per_context_length,
    snakemake.output[0],
    snakemake.wildcards.format,
    title=title,
    xlabel="Total Context (bp)",
    ylabel="Predicted Distribution",
)
