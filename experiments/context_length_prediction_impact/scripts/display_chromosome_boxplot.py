import helpers
import pandas as pd
from snakemake.script import snakemake

# load statistics_over_chromosome
statistics_over_chromosome_path = snakemake.input[0]
statistics_over_chromosome = pd.read_parquet(statistics_over_chromosome_path)

category_counts = statistics_over_chromosome["feature"].value_counts()

# we only plot categories with at least 100 datapoints in it
valid_categories = category_counts[category_counts >= 100].index

filtered_df = statistics_over_chromosome[
    statistics_over_chromosome["feature"].isin(valid_categories)
]

helpers.boxplot(
    filtered_df,
    snakemake.output[0],
    snakemake.wildcards.format,
    title=snakemake.params.title,
    ylabel="Influential Context Size",
)
