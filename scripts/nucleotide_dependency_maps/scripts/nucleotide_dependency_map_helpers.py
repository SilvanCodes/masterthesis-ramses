import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset
from transformers import DefaultDataCollator


def mutate_sequence(seq):
    nuc_table = {"A": 0, "C": 1, "G": 2, "T": 3}

    seq = seq.upper()
    mutated_sequences = {"seq": [], "mutation_pos": [], "nuc": [], "var_nt_idx": []}
    mutated_sequences["seq"].append(seq)
    mutated_sequences["mutation_pos"].append(-1)
    mutated_sequences["nuc"].append("real sequence")
    mutated_sequences["var_nt_idx"].append(-1)

    mutate_until_position = len(seq)

    for i in range(mutate_until_position):
        for nuc in ["A", "C", "G", "T"]:
            if nuc != seq[i]:
                mutated_sequences["seq"].append(seq[:i] + nuc + seq[i + 1 :])
                mutated_sequences["mutation_pos"].append(i)
                mutated_sequences["nuc"].append(nuc)
                mutated_sequences["var_nt_idx"].append(nuc_table[nuc])

    mutations_df = pd.DataFrame(mutated_sequences)

    return mutations_df


def create_dataloader(dataset, tokenizer, batch_size=64, rolling_masking=False):

    ds = Dataset.from_pandas(dataset[["seq"]])

    # print(ds.shape)

    tok_ds = ds.map(
        lambda x: tokenizer(
            list(x["seq"]),
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
        ),
        batched=False,
        num_proc=20,
    )

    # print(tok_ds.shape)

    rem_tok_ds = tok_ds.remove_columns("seq")

    # print(rem_tok_ds.shape)

    data_collator = DefaultDataCollator()

    data_loader = torch.utils.data.DataLoader(
        rem_tok_ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        collate_fn=data_collator,
    )

    return data_loader


def model_inference(model, tokenizer, data_loader, device="cuda"):
    acgt_idxs = [tokenizer.get_vocab()[nuc] for nuc in ["a", "c", "g", "t"]]

    output_arrays = []
    for i, batch in enumerate(data_loader):
        # get some tokenized sequences (B, L_in)

        tokens = batch["input_ids"]

        print(i)

        tokens = torch.squeeze(tokens, dim=2)

        # print(tokens.shape)
        # predict
        with torch.autocast(device):
            with torch.no_grad():
                # ORIGINAL
                # outputs = model(tokens.to(device)).prediction_logits.cpu().to(torch.float32)
                # APATED
                outputs = (
                    model(input_ids=tokens.to(device)).logits.cpu().to(torch.float32)
                )
        output_probs = torch.nn.functional.softmax(outputs, dim=-1)[
            :, :, acgt_idxs
        ]  # B, L_seq, 4
        output_arrays.append(output_probs)

    # rebuild to B, L_seq, 4
    snp_reconstruct = torch.concat(output_arrays, axis=0)

    return snp_reconstruct.to(torch.float32).numpy()


def compute_dependency_map(seq, model, tokenizer, epsilon=1e-10):

    dataset = mutate_sequence(seq)
    data_loader = create_dataloader(dataset, tokenizer)
    snp_reconstruct = model_inference(model, tokenizer, data_loader)

    # those tokens do not esist for GPN
    # snp_reconstruct = snp_reconstruct[:,2:-1,:] # discard the beginning of sentence token, species token and end of sentence token

    # for the logit add a small value epsilon and renormalize such that every prob in one position sums to 1
    snp_reconstruct = snp_reconstruct + epsilon
    snp_reconstruct = snp_reconstruct / snp_reconstruct.sum(axis=-1)[:, :, np.newaxis]

    seq_len = snp_reconstruct.shape[1]
    snp_effect = np.zeros((seq_len, seq_len, 4, 4))
    reference_probs = snp_reconstruct[
        dataset[dataset["nuc"] == "real sequence"].index[0]
    ]

    snp_effect[
        dataset.iloc[1:]["mutation_pos"].values,
        :,
        dataset.iloc[1:]["var_nt_idx"].values,
        :,
    ] = (
        np.log2(snp_reconstruct[1:])
        - np.log2(1 - snp_reconstruct[1:])
        - np.log2(reference_probs)
        + np.log2(1 - reference_probs)
    )

    dep_map = np.max(np.abs(snp_effect), axis=(2, 3))
    # zero main diagonal values
    dep_map[np.arange(dep_map.shape[0]), np.arange(dep_map.shape[0])] = 0

    return dep_map


## Visualization functions
def map_seq_to_file(
    matrix, dna_sequence, path, format, plot_size=10, vmax=5, tick_label_fontsize=8
):

    fig, ax = plt.subplots(figsize=(plot_size, plot_size))

    sns.heatmap(
        matrix, cmap="coolwarm", vmax=vmax, ax=ax, xticklabels=False, yticklabels=False
    )
    ax.set_aspect("equal")

    tick_positions = np.arange(len(dna_sequence)) + 0.5  # Center the ticks

    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(list(dna_sequence), fontsize=tick_label_fontsize, rotation=0)
    ax.set_yticklabels(list(dna_sequence), fontsize=tick_label_fontsize)

    # plt.show()
    plt.savefig(path, format=format)


def map_to_file(
    matrix, path, format, vmax=None, display_values=False, annot_size=8, fig_size=10
):

    plt.figure(figsize=(fig_size, fig_size))

    ax = sns.heatmap(
        matrix,
        cmap="coolwarm",
        vmax=vmax,
        annot=display_values,
        fmt=".2f",
        annot_kws={"size": annot_size},
    )

    ax.set_aspect("equal")

    plt.savefig(path, format=format)
