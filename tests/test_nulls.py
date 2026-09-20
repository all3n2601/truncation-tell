import numpy as np
from truncation_tell.nulls import random_subsets, matched_subsets


def test_random_subsets_shapes_and_uniqueness():
    subsets = random_subsets(n=1000, m=100, count=25, seed=0)
    assert len(subsets) == 25
    for s in subsets:
        assert s.shape == (100,)
        assert len(np.unique(s)) == 100


def test_random_subsets_deterministic():
    a = random_subsets(n=500, m=50, count=5, seed=4)
    b = random_subsets(n=500, m=50, count=5, seed=4)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_matched_subsets_reproduce_covariate_distribution():
    """A null matched on a covariate must not differ on that covariate."""
    rng = np.random.default_rng(0)
    covariates = rng.standard_normal((4000, 2))
    # Target is biased hard toward high values of column 0.
    target = np.argsort(covariates[:, 0])[-400:]

    matched = matched_subsets(covariates, target, count=30, n_bins=5, seed=0)
    unmatched = random_subsets(n=4000, m=400, count=30, seed=0)

    target_mean = covariates[target, 0].mean()
    matched_gap = abs(np.mean([covariates[s, 0].mean() for s in matched]) - target_mean)
    unmatched_gap = abs(
        np.mean([covariates[s, 0].mean() for s in unmatched]) - target_mean
    )
    assert matched_gap < 0.25 * unmatched_gap


def test_matched_subsets_return_correct_size_and_are_disjointly_drawn():
    rng = np.random.default_rng(1)
    covariates = rng.standard_normal((2000, 3))
    target = rng.choice(2000, size=200, replace=False)
    subsets = matched_subsets(covariates, target, count=10, seed=2)
    assert len(subsets) == 10
    for s in subsets:
        assert s.shape == (200,)
        assert len(np.unique(s)) == 200


def test_matched_subsets_accepts_one_dimensional_covariates():
    """A single covariate column may arrive as shape (n,) rather than (n, 1)."""
    rng = np.random.default_rng(5)
    covariates = rng.standard_normal(3000)
    target = np.argsort(covariates)[-300:]
    subsets = matched_subsets(covariates, target, count=5, n_bins=4, seed=0)
    assert len(subsets) == 5
    target_mean = covariates[target].mean()
    for s in subsets:
        assert s.shape == (300,)
        # With n_bins=4 the top-10% target lies wholly inside the top-25% bin, so
        # the best attainable gap is E[X|top10] - E[X|top25] = 0.484 by construction.
        # The bound is on the sampling noise around that floor, not on the floor.
        assert abs(covariates[s].mean() - target_mean) < 0.6
        # ... and matching must still beat an unmatched draw, whose gap is ~1.76.
        assert abs(covariates[s].mean() - target_mean) < 0.5 * target_mean

    # The real content of the (n,) fix: it must agree with an explicit (n, 1) call.
    explicit = matched_subsets(covariates[:, None], target, count=5, n_bins=4, seed=0)
    for flat, column in zip(subsets, explicit):
        np.testing.assert_array_equal(flat, column)


def test_bootstrap_null_subsets_shape_and_determinism():
    from truncation_tell.nulls import bootstrap_null_subsets

    rng = np.random.default_rng(11)
    subset = rng.standard_normal((300, 6))
    a = bootstrap_null_subsets(subset, count=10, seed=2)
    b = bootstrap_null_subsets(subset, count=10, seed=2)
    assert len(a) == 10
    for x, y in zip(a, b):
        assert x.shape == subset.shape
        np.testing.assert_allclose(x, y)


def test_bootstrap_null_subsets_destroys_directional_asymmetry():
    """C4: the blind null must break one-sidedness while preserving marginals.

    Sign-flipping each row about the subset mean symmetrises every direction,
    so a truncated subset's skewness signature vanishes in the null.
    """
    from truncation_tell.nulls import bootstrap_null_subsets
    from truncation_tell.detect import max_abs_skewness
    from truncation_tell.synthetic import make_pool, truncate_select

    pool = make_pool(n=6000, k=10, seed=31)
    idx = truncate_select(pool, np.ones(10), gamma=0.1)
    observed = max_abs_skewness(pool[idx], seed=0)
    nulls = [
        max_abs_skewness(s, seed=0)
        for s in bootstrap_null_subsets(pool[idx], count=15, seed=0)
    ]
    assert observed > max(nulls)
