"""Detection metrics.

AUROC is the headline. TPR at FPR=1% is the number that decides whether a
curation filter is deployable: at dataset scale, a detector with good AUROC
and poor low-FPR behaviour is useless.
"""

from typing import Callable

import numpy as np


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Area under the ROC curve, computed by rank statistic with tie credit."""
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    n_pos, n_neg = positive.size, negative.size
    if n_pos == 0 or n_neg == 0:
        raise ValueError("both groups must be non-empty")
    combined = np.concatenate([positive, negative])
    order = combined.argsort()
    ranks = np.empty(combined.size, dtype=float)
    ranks[order] = np.arange(1, combined.size + 1, dtype=float)
    # Average ranks within tie groups so ties score 0.5.
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    rank_sum = ranks[:n_pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def tpr_at_fpr(
    positive: np.ndarray, negative: np.ndarray, fpr: float = 0.01
) -> float:
    """Fraction of positives above the (1 - fpr) quantile of negatives."""
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    if negative.size == 0 or positive.size == 0:
        raise ValueError("both groups must be non-empty")
    threshold = np.quantile(negative, 1.0 - fpr)
    return float(np.mean(positive > threshold))


def bootstrap_ci(
    positive: np.ndarray,
    negative: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for `metric`, resampling both groups."""
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        p = rng.choice(positive, size=positive.size, replace=True)
        n = rng.choice(negative, size=negative.size, replace=True)
        draws[i] = metric(p, n)
    return (
        float(np.quantile(draws, alpha / 2.0)),
        float(np.quantile(draws, 1.0 - alpha / 2.0)),
    )
