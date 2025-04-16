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

title = f"Influence of context length (chr: {chromosome}, pos: {masked_position}, ref: {sequence[masked_position - 1]})"

threshold_step = (
    helpers.find_context_size_step_with_total_prediction_variance_below_threshold(
        probabilities_per_context_length,
        window_size=window_size,
        threshold=prediction_variance_threshold,
    )
)

helpers.plot_stacked_area(
    probabilities_per_context_length,
    snakemake.output[0],
    snakemake.wildcards.format,
    title=title,
    xlabel="Total Context (bp)",
    ylabel="Predicted Distribution",
    marker=threshold_step,
)
