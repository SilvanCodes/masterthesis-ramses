import helpers
import pandas as pd
from snakemake.script import snakemake
from gpn.data import load_fasta

# load probabilities_per_context_length
probabilities_per_context_length_path = snakemake.input[0]
probabilities_per_context_length = pd.read_parquet(
    probabilities_per_context_length_path
)

print(probabilities_per_context_length.shape)

# load config
chromosome = snakemake.wildcards.chromosome
masked_position = int(snakemake.wildcards.position)
window_size = snakemake.config["DISTRIBUTION_SHIFT_WINDOW_SIZE"]
distribution_shift_threshold = float(snakemake.config["DISTRIBUTION_SHIFT_THRESHOLD"])

title = f"Moved Probability Mass over Context Length \n (chr: {chromosome}, pos: {masked_position})"


distribution_shift = helpers.get_distribution_shift(probabilities_per_context_length)

threshold_step = (
    helpers.find_context_size_step_with_distribution_shift_below_threshold(
        probabilities_per_context_length,
        window_size=window_size,
        threshold=distribution_shift_threshold,
    )
)

helpers.plot_line(
    distribution_shift,
    snakemake.output[0],
    snakemake.wildcards.format,
    title=title,
    xlabel="Context Length (bp)",
    ylabel="Moved Probability Mass",
    marker=threshold_step,
)
