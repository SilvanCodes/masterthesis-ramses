import helpers
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from snakemake.script import snakemake

# gpn specific model configuration
import gpn.model
from gpn.data import load_fasta

print(f"GPU Model: {torch.cuda.get_device_name(0)}")

model_path = snakemake.wildcards.model

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"tokenizer vocabulary: {tokenizer.get_vocab()}")

# load model
model = AutoModelForMaskedLM.from_pretrained(model_path)
device = "cuda"
model.to(device)
model.eval()

# load config
max_context_length = snakemake.config["MAXIMUM_CONTEXT_LENGTH"]

chromosome = snakemake.wildcards.chromosome
masked_position = int(snakemake.wildcards.position)

# load sequence
sequence_path = snakemake.input[0]
genome = load_fasta(sequence_path)
sequence = genome[chromosome]

print(len(sequence))

df = helpers.compute_context_length_dependency(
    model,
    tokenizer,
    sequence,
    masked_position,
    max_context_length=max_context_length,
)

df.to_parquet(snakemake.output[0])
