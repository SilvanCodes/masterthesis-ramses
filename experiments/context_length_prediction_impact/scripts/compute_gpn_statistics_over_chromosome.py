import stats_helpers
import pandas as pd
from tqdm import tqdm
from gpn.data import load_fasta
from snakemake.script import snakemake
import gffutils

# load config
prediction_variance_window_size = snakemake.config["PREDICTION_VARIANCE_WINDOW_SIZE"]
prediction_variance_threshold = float(snakemake.config["PREDICTION_VARIANCE_THRESHOLD"])

chromosome = snakemake.wildcards.chromosome

# load annotation db
db = gffutils.FeatureDB(snakemake.input.annotation_db)


def get_position_feature_type(position, chrom=chromosome):
    overlapping_features = list(db.region(seqid=chrom, start=position, end=position))
    if not overlapping_features:
        return "unknown"
    overlapping_features.sort(key=lambda f: f.end - f.start + 1)
    return overlapping_features[0].featuretype


# load sequence
sequence_path = snakemake.input.sequence
genome = load_fasta(sequence_path)
sequence = genome[chromosome]

threshold_steps = []
positions = []

for df_path in tqdm(snakemake.input.position_data, desc="Random Position", position=0):
    results = pd.read_parquet(df_path)

    position = int(df_path.split("/")[-2])

    reference_nucleotide = sequence[position - 1]

    # skip position when reference is unknown
    if reference_nucleotide in ["n", "N"]:
        continue

    gpn_scores = stats_helpers.compute_gpn_score(reference_nucleotide, results)

    positions.append(position)

    threshold_steps.append(
        stats_helpers.find_context_size_step_with_total_prediction_variance_below_threshold(
            results,
            window_size=prediction_variance_window_size,
            threshold=prediction_variance_threshold,
        )
    )

df = pd.DataFrame({"position": positions, "threshold_steps": threshold_steps})

df["feature"] = df["position"].map(get_position_feature_type)

df.to_parquet(snakemake.output[0])
