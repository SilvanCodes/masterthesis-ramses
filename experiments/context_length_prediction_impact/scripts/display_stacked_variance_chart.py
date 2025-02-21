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
window_size = snakemake.config["PREDICTION_VARIANCE_WINDOW_SIZE"]

# load sequence
sequence_path = snakemake.input[0]
genome = load_fasta(sequence_path)
sequence = genome[chromosome]

title = (
    f"(chr: {chromosome}, pos: {masked_position}, ref: {sequence[masked_position - 1]})"
)

rolling_var = (
    probabilities_per_context_length[::-1]
    .rolling(window=window_size, min_periods=1)
    .var(ddof=0)[::-1]
)


helpers.plot_stacked_area(
    rolling_var,
    snakemake.output[0],
    snakemake.wildcards.format,
    title=title,
    xlabel="Total Context (bp)",
    ylabel="Variance of Prediction",
)
