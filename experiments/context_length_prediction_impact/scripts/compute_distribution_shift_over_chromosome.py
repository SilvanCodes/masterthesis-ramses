import stats_helpers
import pandas as pd
from tqdm import tqdm
from snakemake.script import snakemake
import gffutils

# load config
distribution_shift_window_size = snakemake.config["DISTRIBUTION_SHIFT_WINDOW_SIZE"]
distribution_shift_threshold = float(snakemake.config["DISTRIBUTION_SHIFT_THRESHOLD"])

chromosome = snakemake.wildcards.chromosome

# load annotation db
db = gffutils.FeatureDB(snakemake.input.annotation_db)


def get_position_feature_type(position, chrom=chromosome):
    overlapping_features = list(db.region(seqid=chrom, start=position, end=position))
    if not overlapping_features:
        return "unknown"
    overlapping_features.sort(key=lambda f: f.end - f.start + 1)
    return overlapping_features[0].featuretype


threshold_steps = []
positions = []

for df_path in tqdm(snakemake.input.position_data, desc="Random Position", position=0):
    results = pd.read_parquet(df_path)

    position = int(df_path.split("/")[-2])

    positions.append(position)

    threshold_steps.append(
        stats_helpers.find_context_size_step_with_distribution_shift_below_threshold(
            results,
            window_size=distribution_shift_window_size,
            threshold=distribution_shift_threshold,
        )
    )

df = pd.DataFrame({"position": positions, "threshold_steps": threshold_steps})

df["feature"] = df["position"].map(get_position_feature_type)

df.to_parquet(snakemake.output[0])
