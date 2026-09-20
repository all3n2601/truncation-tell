"""Null subset samplers.

`matched_subsets` is the important one. An unmatched null lets the detector
succeed by noticing that selection correlates with response length or reward
margin, which is a confound rather than evidence of the truncation geometry.
"""

import numpy as np


def random_subsets(n: int, m: int, count: int, seed: int = 0) -> list[np.ndarray]:
    """Draw `count` uniform random subsets of size `m` from range(n)."""
    if m > n:
        raise ValueError(f"cannot draw {m} from {n}")
    rng = np.random.default_rng(seed)
    return [rng.choice(n, size=m, replace=False) for _ in range(count)]


def _stratum_labels(covariates: np.ndarray, n_bins: int) -> np.ndarray:
    """Map each row to a joint quantile-stratum id across all columns."""
    covariates = np.asarray(covariates, dtype=float)
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    labels = np.zeros(covariates.shape[0], dtype=np.int64)
    for col in range(covariates.shape[1]):
        edges = np.quantile(covariates[:, col], np.linspace(0, 1, n_bins + 1)[1:-1])
        binned = np.searchsorted(edges, covariates[:, col], side="right")
        labels = labels * n_bins + binned
    return labels


def matched_subsets(
    covariates: np.ndarray,
    target_idx: np.ndarray,
    count: int,
    n_bins: int = 5,
    seed: int = 0,
) -> list[np.ndarray]:
    """Draw nulls reproducing the target's joint covariate stratum histogram.

    Args:
        covariates: (n, d) observable per-example covariates.
        target_idx: indices of the candidate subset being tested.
        count: number of null subsets to draw.
        n_bins: quantile bins per covariate column.
        seed: PRNG seed.

    Returns:
        `count` index arrays, each the same length as `target_idx`.

    Raises:
        ValueError: if a stratum has too few members to supply the target's
            share without replacement. Reduce `n_bins` if this fires.
    """
    labels = _stratum_labels(covariates, n_bins)
    target_idx = np.asarray(target_idx)
    wanted = {}
    for stratum, needed in zip(*np.unique(labels[target_idx], return_counts=True)):
        wanted[int(stratum)] = int(needed)

    members = {
        int(s): np.flatnonzero(labels == s) for s in wanted
    }
    for stratum, needed in wanted.items():
        if members[stratum].size < needed:
            raise ValueError(
                f"stratum {stratum} has {members[stratum].size} members but the "
                f"target needs {needed}; lower n_bins"
            )

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        picked = [
            rng.choice(members[stratum], size=needed, replace=False)
            for stratum, needed in wanted.items()
        ]
        out.append(np.concatenate(picked))
    return out


def bootstrap_null_subsets(
    subset: np.ndarray, count: int, seed: int = 0
) -> list[np.ndarray]:
    """Blind null: symmetrise the subset about its own mean.

    The blind threat model has no source pool, so the null must come from the
    subset itself. Flipping each row's offset about the subset mean with a
    random sign preserves every marginal's scale and the covariance structure
    while destroying one-sidedness along every direction -- which is exactly
    the property one-sided tail truncation creates.
    """
    subset = np.asarray(subset, dtype=float)
    centre = subset.mean(axis=0)
    offsets = subset - centre
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        signs = rng.choice([-1.0, 1.0], size=(subset.shape[0], 1))
        out.append(centre + signs * offsets)
    return out
