# goal: take a DNA sequcene and compute for each position
# - gpn predicted distribution
# - gpn scores with given sequence as assumed reference nucleotide

# result file layout:
# index: position on chromosome
# column: ref, p_a, p_c, p_g, p_t, gpn_a, gpa_c, gpn_g, gpn_t
# p_x denotes the resulting gpn predicted probabilities
# gpn_x denotes the gpn score w.r.t ref, the reference_nucleotide

import numpy as np
import pandas as pd
from snakemake.script import snakemake
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
from Bio.Seq import Seq
import warnings


# gpn specific model configuration
import gpn.model
from gpn.data import load_fasta

chromosome = snakemake.wildcards.chromosome

# load sequence
sequence_path = snakemake.input[0]
genome = load_fasta(sequence_path)
sequence = genome[chromosome]

model_path = snakemake.config["MODEL_PATH"]

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"tokenizer vocabulary: {tokenizer.get_vocab()}")

# load model
model = AutoModelForMaskedLM.from_pretrained(model_path)
device = "cuda"
model.to(device)
model.eval()

# start of HMA4-3
start_position = int(snakemake.wildcards.start_position)
# end of HMA4-1
stop_position = int(snakemake.wildcards.stop_position)

print(f"start_position: {start_position}")
print(f"stop_position: {stop_position}")

# Adjust start and stop positions to be within the sequence length
start_position = max(1, start_position)
stop_position = min(len(sequence), stop_position)

print(f"start_position: {start_position}")
print(f"stop_position: {stop_position}")

if start_position != int(snakemake.wildcards.start_position):
    warnings.warn(
        f"start_position was adjusted to be within the sequence length, now: {start_position}"
    )
if stop_position != int(snakemake.wildcards.stop_position):
    warnings.warn(
        f"stop_position was adjusted to be within the sequence length, now: {stop_position}"
    )

if snakemake.config["REVERSE_COMPLEMENT"]:
    warnings.warn(
        "REVERSE_COMPLEMENT is set to True, the sequence will be reverse complemented"
    )

context_length = int(snakemake.config["INCLUDED_CONTEXT"])

window_size = context_length + 1


# print(f"start_position: {start_position}")
# print(f"stop_position: {stop_position}")

# print(f"sequence length: {len(sequence)}")
# print(f"context_length: {context_length}")
# print(f"window_size: {window_size}")

assert start_position < stop_position

# would ensure all positions have equal context available
# assert stop_position - start_position < len(sequence) - context_length


def sliding_window_generator(
    sequence, start_position, stop_position, tokenizer, window_size=513, step_size=1
):
    """
    Generate sliding windows over a DNA sequence.

    Args:
        fasta_path (str): Path to the FASTA file
        window_size (int): Size of the sliding window
        step_size (int): Step size for sliding the window

    Yields:
        dict: A dictionary with the sequence window
    """
    seq_len = len(sequence)

    window_size_half = window_size // 2

    # condition has been relaxed by clipping the start and end positions inside loop
    # assert start_position - window_size_half - 1 >= 0
    # assert stop_position + window_size_half - 1 <= seq_len

    for position in range(start_position, stop_position + 1, step_size):
        # arrays are 0-indexed, genomes 1-indexed
        position = position - 1

        start = position - window_size_half
        end = position + window_size_half + 1

        # Slice the actual sequence
        sequence_window = sequence[max(start, 0) : min(end, seq_len)]

        # Calculate how much padding is needed
        left_pad = max(0, -start)
        right_pad = max(0, end - seq_len)

        # Pad with 'n' (ambiguous base) as needed
        sequence_window = ("n" * left_pad) + sequence_window + ("n" * right_pad)

        assert (
            len(sequence_window) == window_size
        ), f"Expected {window_size}, got {len(sequence_window)}"

        # if snakemake.config["REVERSE_COMPLEMENT"]:

        if snakemake.wildcards.reverse_complement == "rev":
            sequence_window = str(Seq(sequence_window).reverse_complement())

        center = len(sequence_window) // 2

        tokenized_input = tokenizer(
            sequence_window,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        # Remove the batch dimension for dataset compatibility
        tokenized_data = {k: v.squeeze(0) for k, v in tokenized_input.items()}

        # mask the center nucleotide
        tokenized_data["input_ids"][center] = tokenizer.mask_token_id
        tokenized_data["reference"] = sequence_window[center].lower()

        # Add position information
        tokenized_data["position"] = position
        tokenized_data["sequence"] = (
            sequence_window  # Keep the original sequence for reference
        )

        yield tokenized_data


dataset = Dataset.from_generator(
    lambda: sliding_window_generator(
        sequence, start_position, stop_position, tokenizer, window_size=window_size
    ),
)


def collate_fn(batch):
    return {
        "input_ids": torch.tensor([item["input_ids"] for item in batch]),
        "reference": [item["reference"] for item in batch],
        "sequence": [item["sequence"] for item in batch],
        "position": [item["position"] for item in batch],
    }


batch_size = int(snakemake.config["BATCH_SIZE"])

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    collate_fn=collate_fn,
    shuffle=False,  # For sliding windows, keep in order
)

# Process batches
# all_predictions = []
acgt_idxs = [tokenizer.get_vocab()[nuc] for nuc in ["a", "c", "g", "t"]]

center = window_size // 2
results = []
for batch in tqdm(dataloader, desc="Batch"):
    current_input = batch["input_ids"]

    with torch.no_grad():
        all_logits = (
            model(input_ids=current_input.to(device)).logits.cpu().to(torch.float32)
        )

    nucleotide_logits = all_logits[:, :, acgt_idxs]
    output_probs = torch.nn.functional.softmax(nucleotide_logits, dim=-1)

    # all_predictions.append(output_probs)

    for i in range(len(batch["input_ids"])):
        results.append(
            {
                "position": batch["position"][i],
                "reference": batch["reference"][i],
                "p_a": output_probs[i][center][0],
                "p_c": output_probs[i][center][1],
                "p_g": output_probs[i][center][2],
                "p_t": output_probs[i][center][3],
            }
        )

results = pd.DataFrame(results)

# we kinda compute back to front when flipping to reverse complement so data is nicer to be understood when reverted here
# if snakemake.config["REVERSE_COMPLEMENT"]:
if snakemake.wildcards.reverse_complement == "rev":
    results = results[::-1]

# convert all tensors to floats
results = results.map(
    lambda x: x.item() if torch.is_tensor(x) and x.numel() == 1 else x
)

p_reference = [
    row[col] if col != "p_n" else 0.0
    for row, col in zip(results.to_dict("records"), "p_" + results["reference"])
]

for alt in ["a", "c", "g", "t"]:
    results["gpn_" + alt] = results["p_" + alt] / p_reference

results[["gpn_a", "gpn_c", "gpn_g", "gpn_t"]] = np.log2(
    results[["gpn_a", "gpn_c", "gpn_g", "gpn_t"]]
)

results.to_parquet(snakemake.output[0])
