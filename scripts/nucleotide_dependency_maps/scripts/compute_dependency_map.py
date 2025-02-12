import nucleotide_dependency_map_helpers as ndm
import pandas as pd
import torch
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

chromosome = snakemake.config["CHROMOSOME"]
# subtract due to zero based indexing in arrays
seq_start = snakemake.config["START_POSITION"] - 1
seq_end = snakemake.config["END_POSITION"]

print(f"start: {seq_start}, end: {seq_end}, end-start: {seq_end - seq_start}")


# load sequence
sequence_path = snakemake.input[0]
genome = load_fasta(sequence_path)
sequence = genome[chromosome][seq_start:seq_end]

print(len(sequence))

dependency_map = ndm.compute_dependency_map(sequence, model, tokenizer)

df = pd.DataFrame(dependency_map)
df.to_parquet(snakemake.output[0])
