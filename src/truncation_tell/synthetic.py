"""Synthetic probe-battery matrices with a known truncation direction.

The null here deliberately has a power-law covariance spectrum. Real
preference data already shows low logit rank, so a detector that only
distinguishes "low rank" from "isotropic" would be measuring nothing.
"""

import numpy as np


def make_pool(n: int, k: int, alpha: float = 1.0, seed: int = 0) -> np.ndarray:
    """Draw an (n, k) pool with power-law covariance spectrum.

    Args:
        n: number of examples.
        k: probe battery size.
        alpha: power-law exponent; eigenvalue j scales as j ** -alpha.
        seed: PRNG seed.

    Returns:
        Array of shape (n, k), mean ~0, symmetric marginals.
    """
    rng = np.random.default_rng(seed)
    eigs = np.arange(1, k + 1, dtype=float) ** (-alpha)
    basis, _ = np.linalg.qr(rng.standard_normal((k, k)))
    loading = basis @ np.diag(np.sqrt(eigs))
    return rng.standard_normal((n, k)) @ loading.T


def truncate_select(
    pool: np.ndarray, direction: np.ndarray, gamma: float
) -> np.ndarray:
    """Select the top-gamma fraction by projection onto `direction`.

    This is the geometry of Logit-Linear Selection stated in the observable
    subspace: one-sided tail truncation along a single direction.

    Args:
        pool: (n, k) array.
        direction: length-k vector; magnitude is irrelevant.
        gamma: fraction retained, in (0, 1].

    Returns:
        Integer indices into `pool`, length round(gamma * n).
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    unit = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(unit)
    if norm == 0.0:
        raise ValueError("direction must be nonzero")
    unit = unit / norm
    scores = pool @ unit
    m = int(round(gamma * pool.shape[0]))
    if m < 1:
        raise ValueError(f"gamma={gamma} selects no examples from n={pool.shape[0]}")
    return np.argsort(scores)[-m:]
