"""Detection statistics for one-sided tail truncation.

Every statistic returns higher = more anomalous.

Curator statistics (this module's first half) compare a candidate subset
against the source pool. Blind statistics (added in Task 4) use only the
subset.
"""

import numpy as np


def _pool_covariance(pool: np.ndarray, ridge: float) -> np.ndarray:
    cov = np.cov(pool, rowvar=False)
    cov = np.atleast_2d(cov)
    scale = float(np.trace(cov)) / cov.shape[0]
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return cov + ridge * scale * np.eye(cov.shape[0])


def hotelling_t2(
    subset: np.ndarray, pool: np.ndarray, ridge: float = 1e-6
) -> float:
    """Hotelling T-squared of the subset mean against the pool mean.

    Detects the mean shift that tail truncation induces along the hidden
    direction, whitened by pool covariance so that the already-anisotropic
    null does not dominate.
    """
    delta = subset.mean(axis=0) - pool.mean(axis=0)
    cov = _pool_covariance(pool, ridge)
    solved = np.linalg.solve(cov, delta)
    return float(subset.shape[0] * delta @ solved)


def variance_deflation(
    subset: np.ndarray, pool: np.ndarray, ridge: float = 1e-6
) -> float:
    """Negative log variance ratio along the whitened mean-shift direction.

    Truncation narrows the marginal along the selection direction, so the
    ratio falls below 1 and this returns a positive value.
    """
    delta = subset.mean(axis=0) - pool.mean(axis=0)
    cov = _pool_covariance(pool, ridge)
    direction = np.linalg.solve(cov, delta)
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        return 0.0
    direction = direction / norm
    pool_var = float(np.var(pool @ direction))
    subset_var = float(np.var(subset @ direction))
    if pool_var <= 0.0:
        # No pool variance along this direction: nothing to compare against.
        return 0.0
    if subset_var <= 0.0:
        # Total collapse is maximal deflation, not absence of signal. Report a
        # large finite value rather than folding it into the no-signal case.
        return float(-np.log(np.finfo(float).tiny / pool_var))
    return float(-np.log(subset_var / pool_var))


def _whiten(data: np.ndarray, ridge: float) -> np.ndarray:
    centered = data - data.mean(axis=0)
    cov = np.atleast_2d(np.cov(centered, rowvar=False))
    scale = float(np.trace(cov)) / cov.shape[0]
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    cov = cov + ridge * scale * np.eye(cov.shape[0])
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, ridge * scale, None)
    whitener = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T
    return centered @ whitener


def _skewness(values: np.ndarray) -> float:
    centered = values - values.mean()
    std = centered.std()
    if std <= 0.0:
        return 0.0
    return float(np.mean((centered / std) ** 3))


def max_abs_skewness(
    subset: np.ndarray,
    n_restarts: int = 20,
    iters: int = 200,
    ridge: float = 1e-6,
    seed: int = 0,
) -> float:
    """Largest absolute skewness over all unit directions, found by pursuit.

    Natural preference data is near-symmetric along most directions. One-sided
    tail truncation manufactures asymmetry along the selection direction, so a
    large value is evidence of selection. Uses only `subset` -- no pool, no
    knowledge of the attacker's system prompt.
    """
    whitened = _whiten(np.asarray(subset, dtype=float), ridge)
    if whitened.shape[0] < 3:
        # Skewness is not meaningful below three points.
        return 0.0
    rng = np.random.default_rng(seed)
    best = 0.0
    for _ in range(n_restarts):
        direction = rng.standard_normal(whitened.shape[1])
        direction /= np.linalg.norm(direction)
        for _ in range(iters):
            projection = whitened @ direction
            update = (whitened * (projection ** 2)[:, None]).mean(axis=0)
            norm = np.linalg.norm(update)
            if norm < 1e-12:
                break
            update = update / norm
            if np.linalg.norm(update - direction) < 1e-10:
                direction = update
                break
            direction = update
        best = max(best, abs(_skewness(whitened @ direction)))
    return float(best)
