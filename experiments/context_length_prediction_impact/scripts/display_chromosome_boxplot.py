import helpers
import pandas as pd

from gpn.data import load_fasta

# load statistics_over_chromosome
statistics_over_chromosome_path = snakemake.input[0]
statistics_over_chromosome = pd.read_parquet(statistics_over_chromosome_path)

print(statistics_over_chromosome.shape)

title = f"Distribution of context thesholds"

helpers.boxplot(
    statistics_over_chromosome["threshold_steps"],
    snakemake.output[0],
    snakemake.wildcards.format,
    title=title,
    ylabel="Influential Context Size",
)
