import numpy as np
from truncation_tell.synthetic import make_pool, truncate_select
from truncation_tell.detect import max_abs_skewness


def test_blind_statistic_higher_under_truncation():
    pool = make_pool(n=8000, k=12, seed=20)
    rng = np.random.default_rng(1)
    direction = rng.standard_normal(12)
    idx = truncate_select(pool, direction, gamma=0.1)
    attacked = max_abs_skewness(pool[idx], seed=0)

    nulls = [
        max_abs_skewness(pool[rng.choice(8000, size=len(idx), replace=False)], seed=0)
        for _ in range(20)
    ]
    assert attacked > np.max(nulls)


def test_blind_statistic_never_reads_the_pool():
    """Signature check: the blind statistic takes exactly one array."""
    import inspect
    params = list(inspect.signature(max_abs_skewness).parameters)
    assert params[0] == "subset"
    assert "pool" not in params


def test_blind_statistic_deterministic_given_seed():
    pool = make_pool(n=2000, k=10, seed=21)
    idx = truncate_select(pool, np.ones(10), gamma=0.2)
    a = max_abs_skewness(pool[idx], seed=7)
    b = max_abs_skewness(pool[idx], seed=7)
    assert a == b


def test_blind_statistic_nonnegative_and_finite_on_degenerate_input():
    rng = np.random.default_rng(2)
    subset = np.zeros((50, 6))
    subset[:, 0] = rng.standard_normal(50)
    value = max_abs_skewness(subset, seed=0)
    assert np.isfinite(value)
    assert value >= 0.0


def test_blind_statistic_handles_more_probes_than_examples():
    """C3: E1 sweeps k to 64; a small subset can have fewer rows than columns."""
    rng = np.random.default_rng(3)
    subset = rng.standard_normal((20, 64))
    value = max_abs_skewness(subset, n_restarts=3, iters=20, seed=0)
    assert np.isfinite(value)
    assert value >= 0.0
