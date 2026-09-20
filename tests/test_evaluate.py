import numpy as np
from truncation_tell.evaluate import auroc, tpr_at_fpr, bootstrap_ci


def test_auroc_perfect_separation():
    assert auroc(np.array([3.0, 4.0, 5.0]), np.array([0.0, 1.0, 2.0])) == 1.0


def test_auroc_no_separation_is_one_half():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert auroc(values, values) == 0.5


def test_auroc_handles_ties_as_half_credit():
    assert auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5


def test_tpr_at_fpr_perfect_and_null_cases():
    positive = np.arange(100.0) + 1000.0
    negative = np.arange(100.0)
    assert tpr_at_fpr(positive, negative, fpr=0.01) == 1.0
    assert tpr_at_fpr(negative, negative, fpr=0.01) < 0.1


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    positive = rng.normal(1.5, 1.0, size=300)
    negative = rng.normal(0.0, 1.0, size=300)
    point = auroc(positive, negative)
    low, high = bootstrap_ci(positive, negative, auroc, n_boot=500, seed=0)
    assert low < point < high
    assert 0.0 <= low < high <= 1.0
