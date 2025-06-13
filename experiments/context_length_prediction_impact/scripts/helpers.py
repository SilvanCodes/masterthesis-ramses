import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm


def get_gpn_probabilities(model, tokenizer, context_tokens, device="cuda"):
    acgt_idxs = [tokenizer.get_vocab()[nuc] for nuc in ["a", "c", "g", "t"]]
    with torch.no_grad():
        all_logits = (
            model(input_ids=context_tokens.to(device)).logits.cpu().to(torch.float32)
        )
    nucleotide_logits = all_logits[:, :, acgt_idxs]
    output_probs = torch.nn.functional.softmax(nucleotide_logits, dim=-1)
    return output_probs


def compute_context_length_dependency(
    model,
    tokenizer,
    chromosome_sequence,
    position,
    max_context_length=1000,
    step_size=10,
):
    # shift to zero based indexing in arrays vs. chromosomes
    position = position - 1

    half = max_context_length // 2
    start = int(position - half)
    end = int(position + half + 1)

    sequence = chromosome_sequence[start:end]

    # print(f"masked: {sequence[half]}")

    input_ids = tokenizer(
        sequence,
        return_tensors="pt",
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]

    results_dict = {}

    steps = max_context_length // step_size
    center = len(input_ids[0]) // 2

    # mask the center nucleotide
    input_ids[0, center] = tokenizer.mask_token_id

    for i in tqdm(
        range(0, steps * step_size + 1, step_size),
        desc="Predicting",
        position=1,
        leave=False,
    ):
        half = i // 2
        start = int(center - half)
        end = int(center + half + 1)

        current_context = input_ids[:, start:end]

        # print(current_context)

        results_dict[i] = get_gpn_probabilities(model, tokenizer, current_context)[
            0, half, :
        ]

    results = pd.DataFrame(results_dict, index=["a", "c", "g", "t"])

    return results.T


def rolling_variance(results, window_size=100):
    # we compute the variance over the forward looking window, i.e. if in the future nothing changes, we dont need more context size
    return results[::-1].rolling(window=window_size, min_periods=1).var(ddof=0)[::-1]


def find_context_size_step_with_total_prediction_variance_below_threshold(
    results, window_size=100, threshold=1e-5
):
    rolling_var = rolling_variance(results, window_size=window_size)
    return rolling_var.index[rolling_var.sum(axis=1) < threshold].min()

def get_distribution_shift(df):
    return df.diff().abs().sum(axis=1).div(2)[1:]

def find_context_size_step_with_distribution_shift_below_threshold(results, window_size=10, threshold=0.01):
    distribution_shift = get_distribution_shift(results)
    roll_avg_diff = distribution_shift[::-1].rolling(window=window_size, min_periods=1).mean()[::-1]
    return roll_avg_diff.index[roll_avg_diff < threshold].min()


def compute_gpn_score(reference_nucleotide, probabilities_per_context_length):

    reference_nucleotide = reference_nucleotide.lower()

    nucleotides = ["a", "c", "g", "t"]
    alternatives = [n for n in nucleotides if n != reference_nucleotide]

    gpn_scores = probabilities_per_context_length

    for alt in alternatives:
        gpn_scores[f"gpn_{alt}"] = gpn_scores[alt] / gpn_scores[reference_nucleotide]

    gpn_scores = np.log2(gpn_scores).drop(nucleotides, axis=1)


def plot_stacked_area(
    df,
    path,
    format,
    title="Stacked Area Chart",
    xlabel="X-axis",
    ylabel="Y-axis",
    marker=None,
):
    """
    Plots a stacked area chart from a Pandas DataFrame.

    Parameters:
    - df: Pandas DataFrame (index is x-axis, columns are categories to be stacked).
    - title: Title of the chart.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - colors: Optional list of colors for the areas.
    """
    plt.figure(figsize=(10, 6))
    df.plot(kind="area", stacked=True, alpha=0.7, colormap="viridis")

    print(marker)
    if marker:
        plt.axvline(
            x=marker, color="red", linestyle="-", linewidth=1, label="threshold"
        )

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(title="Nucleotides", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(path, format=format)

def plot_line(
    df,
    path,
    format,
    title="Line Chart",
    xlabel="X-axis",
    ylabel="Y-axis",
    marker=None,
):
    plt.figure(figsize=(10, 6))
    df.plot(kind="line", alpha=0.7, colormap="viridis", ylim=(0, 1))

    # print(marker)
    if marker:
        plt.axvline(
            x=marker, color="red", linestyle="-", linewidth=1, label="threshold"
        )

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(path, format=format)


def boxplot(
    data,
    path,
    format,
    title="Boxplot",
    xlabel="X-axis",
    ylabel="Y-axis",
    y_col="threshold_steps",
    x_col="feature",
    marker=None,
):
    # Create the box plot
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=data, y=y_col, x=x_col, color="skyblue")

    sample_sizes = data[x_col].value_counts().sort_index()

    for i, label in enumerate(sample_sizes):
        plt.annotate(f"n={label}", (i, plt.gca().get_ylim()[0]), xytext=(0, -20), 
                    textcoords='offset points', ha='center', va='top')
        
    if marker:
        plt.axhline(
            y=marker, color="red", linestyle="-", linewidth=1, label="training sample size"
        )

    # Add labels and title
    plt.title(title, fontsize=14)
    # plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    plt.savefig(path, format=format)
