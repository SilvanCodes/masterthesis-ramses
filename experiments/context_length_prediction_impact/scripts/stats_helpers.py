import numpy as np


def rolling_variance(results, window_size=100):
    # we compute the variance over the forward looking window, i.e. if in the future nothing changes, we dont need more context size
    return results[::-1].rolling(window=window_size, min_periods=1).var(ddof=0)[::-1]


def find_context_size_step_with_total_prediction_variance_below_threshold(
    results, window_size=100, threshold=1e-5
):
    rolling_var = rolling_variance(results, window_size)
    return rolling_var.index[rolling_var.sum(axis=1) < threshold].min()


def compute_gpn_score(reference_nucleotide, probabilities_per_context_length):

    reference_nucleotide = reference_nucleotide.lower()

    nucleotides = ["a", "c", "g", "t"]
    alternatives = [n for n in nucleotides if n != reference_nucleotide]

    gpn_scores = probabilities_per_context_length

    for alt in alternatives:
        gpn_scores[f"gpn_{alt}"] = gpn_scores[alt] / gpn_scores[reference_nucleotide]

    gpn_scores = np.log2(gpn_scores).drop(nucleotides, axis=1)
