import random

import helpers
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

# gpn specific model configuration
import gpn.model
from gpn.data import load_fasta

print(f"GPU Model: {torch.cuda.get_device_name(0)}")

model_path = snakemake.params.model

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"tokenizer vocabulary: {tokenizer.get_vocab()}")

# load model
model = AutoModelForMaskedLM.from_pretrained(model_path)
device = "cuda"
model.to(device)
model.eval()

# load config
chromosome = snakemake.config["CHROMOSOME"]
random_position_count = snakemake.config["RANDOM_POSITION_COUNT"]
max_context_length = snakemake.config["MAXIMUM_CONTEXT_LENGTH"]
prediction_variance_window_size = snakemake.config["PREDICTION_VARIANCE_WINDOW_SIZE"]
prediction_variance_threshold = float(snakemake.config["PREDICTION_VARIANCE_THRESHOLD"])

# load sequence
sequence_path = snakemake.input[0]
genome = load_fasta(sequence_path)
sequence = genome[chromosome]


gap = max_context_length // 2

random_positions = random.sample(range(gap, len(sequence) - gap), random_position_count)

threshold_steps = []

for i in tqdm(random_positions, desc="Random Position", position=0):
    results = helpers.compute_context_length_dependency(
        model,
        tokenizer,
        sequence,
        i,
        max_context_length=max_context_length,
    )
    threshold_steps.append(
        helpers.find_context_size_step_with_total_prediction_variance_below_threshold(
            results,
            window_size=prediction_variance_window_size,
            threshold=prediction_variance_threshold,
        )
    )

df = pd.DataFrame(
    {"random_positions": random_positions, "threshold_steps": threshold_steps}
)
df.to_parquet(snakemake.output[0])
