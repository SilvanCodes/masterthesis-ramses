import nucleotide_dependency_map_helpers as ndm

# import pandas as pd
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

# load sequence
sequence_path = snakemake.input[0]

genome = load_fasta(sequence_path)

chromosome = snakemake.config["CHROMOSOME"]
start = snakemake.config["START_POSITION"]
end = snakemake.config["END_POSITION"]

sequence = genome[chromosome][start:end]

dependency_map = ndm.compute_dependency_map(sequence, model, tokenizer)

# df = pd.DataFrame(dependency_map)
# df.to_parquet(snakemake.output[0])

ndm.map_seq_to_file(
    dependency_map, sequence, snakemake.output[0], snakemake.wildcards.format
)
