import helpers
import pandas as pd
from snakemake.script import snakemake
from gpn.data import load_fasta

# load probabilities_per_context_length
probabilities_per_context_length_path = snakemake.input[1]
probabilities_per_context_length = pd.read_parquet(
    probabilities_per_context_length_path
)

print(probabilities_per_context_length.shape)

# load config
chromosome = snakemake.wildcards.chromosome
masked_position = int(snakemake.wildcards.position)
window_size = snakemake.config["PREDICTION_VARIANCE_WINDOW_SIZE"]
prediction_variance_threshold = float(snakemake.config["PREDICTION_VARIANCE_THRESHOLD"])

# load sequence
sequence_path = snakemake.input[0]
genome = load_fasta(sequence_path)
sequence = genome[chromosome]

title = (
    f"(chr: {chromosome}, pos: {masked_position}, ref: {sequence[masked_position - 1]})"
)

rolling_var = helpers.rolling_variance(
    probabilities_per_context_length, window_size=window_size
)

threshold_step = rolling_var.index[
    rolling_var.sum(axis=1) < prediction_variance_threshold
].min()


helpers.plot_stacked_area(
    rolling_var,
    snakemake.output[0],
    snakemake.wildcards.format,
    title=title,
    xlabel="Total Context (bp)",
    ylabel="Variance of Prediction",
    marker=threshold_step,
)
