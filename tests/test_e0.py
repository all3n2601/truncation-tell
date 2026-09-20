import numpy as np
from truncation_tell.e0 import run_e0


def test_e0_all_statistics_detect_synthetic_truncation():
    results = run_e0(gammas=(0.1,), k=12, n=4000, n_nulls=100, seed=0)
    assert len(results) == 3
    for key, row in results.items():
        assert row["auroc"] > 0.95, f"{key} failed E0 with AUROC {row['auroc']}"


def test_e0_reports_confidence_intervals_bracketing_point():
    results = run_e0(gammas=(0.1,), k=12, n=4000, n_nulls=100, seed=0)
    for row in results.values():
        assert row["auroc_ci_low"] <= row["auroc"] <= row["auroc_ci_high"]


def test_e0_deterministic():
    a = run_e0(gammas=(0.1,), k=8, n=2000, n_nulls=50, seed=3)
    b = run_e0(gammas=(0.1,), k=8, n=2000, n_nulls=50, seed=3)
    assert a == b
