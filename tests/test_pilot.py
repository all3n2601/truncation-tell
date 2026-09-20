import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from truncation_tell.pilot import project_cost


def test_project_cost_scales_linearly_in_examples_and_probes():
    small = project_cost(seconds_per_score=0.1, n=10, k=4)
    large = project_cost(seconds_per_score=0.1, n=20, k=4)
    assert large["hours"] == pytest.approx(2 * small["hours"])


def test_project_cost_counts_baseline_column():
    """n examples x (k probes + 1 baseline) scoring calls."""
    cost = project_cost(seconds_per_score=1.0, n=100, k=9)
    assert cost["n_scores"] == 100 * 10
    assert cost["hours"] == pytest.approx(1000 / 3600)


def test_project_cost_rejects_nonpositive_rate():
    with pytest.raises(ValueError):
        project_cost(seconds_per_score=0.0, n=10, k=4)
