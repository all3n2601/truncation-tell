import numpy as np
from truncation_tell.synthetic import make_pool, truncate_select
from truncation_tell.detect import hotelling_t2, variance_deflation

RNG = np.random.default_rng(0)


def _random_subset(pool, m, rng):
    return rng.choice(pool.shape[0], size=m, replace=False)


def test_hotelling_separates_truncated_from_random():
    pool = make_pool(n=4000, k=12, seed=10)
    direction = RNG.standard_normal(12)
    idx = truncate_select(pool, direction, gamma=0.1)
    attacked = hotelling_t2(pool[idx], pool)
    nulls = [
        hotelling_t2(pool[_random_subset(pool, len(idx), RNG)], pool)
        for _ in range(50)
    ]
    assert attacked > max(nulls)


def test_variance_deflation_positive_under_truncation():
    """Truncation narrows the marginal, so deflation is positive."""
    pool = make_pool(n=4000, k=12, seed=11)
    direction = RNG.standard_normal(12)
    idx = truncate_select(pool, direction, gamma=0.1)
    assert variance_deflation(pool[idx], pool) > 0.0


def test_variance_deflation_near_zero_for_random_subset():
    pool = make_pool(n=4000, k=12, seed=12)
    idx = _random_subset(pool, 400, RNG)
    assert abs(variance_deflation(pool[idx], pool)) < 0.5


def test_statistics_are_finite_on_degenerate_input():
    """Rank-deficient pools must not produce nan or raise."""
    pool = make_pool(n=200, k=12, seed=13)
    pool[:, 6:] = 0.0
    idx = _random_subset(pool, 40, RNG)
    assert np.isfinite(hotelling_t2(pool[idx], pool))
    assert np.isfinite(variance_deflation(pool[idx], pool))


def test_degenerate_input_does_not_inflate_statistic():
    """Finite is not enough.

    Ridging near-zero-variance columns can amplify floating-point noise into a
    large-but-finite T-squared. An unattacked subset must score comparably
    whether or not the pool is near rank-deficient.
    """
    rng = np.random.default_rng(99)
    full = make_pool(n=2000, k=12, seed=14)
    degenerate = full.copy()
    degenerate[:, 6:] *= 1e-8
    idx = rng.choice(2000, size=200, replace=False)
    full_t2 = hotelling_t2(full[idx], full)
    degenerate_t2 = hotelling_t2(degenerate[idx], degenerate)
    assert degenerate_t2 < 20.0 * max(full_t2, 1.0)


def test_variance_deflation_reports_total_collapse_as_maximally_anomalous():
    """C2: subset_var == 0 is TOTAL deflation, the strongest possible positive.

    The old guard returned 0.0 for it, reporting the most anomalous input as
    the least anomalous one.
    """
    pool = make_pool(n=2000, k=8, seed=30)
    delta = np.zeros(8)
    delta[0] = 5.0
    subset = np.tile(pool.mean(axis=0) + delta, (200, 1))
    value = variance_deflation(subset, pool)
    assert np.isfinite(value)
    assert value > 10.0
