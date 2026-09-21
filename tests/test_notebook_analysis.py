"""Guards the analysis logic embedded in notebooks/truncation_tell_colab.ipynb.

The notebook's detection cells are the one place where code lives outside the
package, so they are the one place a refactor can break silently. These tests
run the same sequence against a synthetic matrix: if an interface here changes,
this fails locally instead of an hour into a GPU session.

Keep in sync with cells 9 and 10 of the notebook.
"""

import json

import numpy as np
import pytest

from truncation_tell.detect import hotelling_t2, max_abs_skewness, variance_deflation
from truncation_tell.nulls import bootstrap_null_subsets, random_subsets
from truncation_tell.synthetic import make_pool, truncate_select

SEED = 0
K_SWEEP = [4, 8, 16]
N_NULLS = 40


def _pvalue(observed, nulls):
    """Rank-based p-value, as in the notebook.

    Real data yields exactly ONE selection per trait, so AUROC is undefined --
    it needs a distribution of positives. The rank of the observed statistic
    among the nulls is the honest single-observation equivalent.
    """
    nulls = np.asarray(nulls)
    return (1 + int((nulls >= observed).sum())) / (1 + len(nulls))


@pytest.fixture(scope="module")
def sweep():
    matrix = make_pool(n=400, k=16, seed=SEED)
    direction = np.random.default_rng(1).standard_normal(16)
    selected = truncate_select(matrix, direction, gamma=0.10)

    rows = []
    for k in K_SWEEP:
        v, sub = matrix[:, :k], matrix[selected][:, :k]
        null_idx = random_subsets(len(matrix), len(selected), N_NULLS, seed=SEED)
        for name, fn in (
            ("hotelling_t2", hotelling_t2),
            ("variance_deflation", variance_deflation),
        ):
            obs = fn(sub, v)
            nulls = [fn(v[i], v) for i in null_idx]
            rows.append(
                dict(
                    k=k,
                    threat="curator",
                    statistic=name,
                    observed=obs,
                    null_max=float(np.max(nulls)),
                    margin=obs - float(np.max(nulls)),
                    p=_pvalue(obs, nulls),
                )
            )
        obs = max_abs_skewness(sub, seed=SEED)
        nulls = [
            max_abs_skewness(s, seed=SEED)
            for s in bootstrap_null_subsets(sub, count=N_NULLS, seed=SEED)
        ]
        rows.append(
            dict(
                k=k,
                threat="blind",
                statistic="max_abs_skewness",
                observed=obs,
                null_max=float(np.max(nulls)),
                margin=obs - float(np.max(nulls)),
                p=_pvalue(obs, nulls),
            )
        )
    return rows


def test_sweep_produces_a_row_per_statistic_and_k(sweep):
    assert len(sweep) == len(K_SWEEP) * 3
    assert {r["threat"] for r in sweep} == {"curator", "blind"}


def test_every_value_is_finite(sweep):
    """A nan here would print as a plausible-looking blank in the notebook."""
    for row in sweep:
        for key in ("observed", "null_max", "margin", "p"):
            assert np.isfinite(row[key]), f"{key} not finite in {row}"


def test_pvalues_are_bounded_by_the_null_count(sweep):
    floor = 1 / (1 + N_NULLS)
    for row in sweep:
        assert floor <= row["p"] <= 1.0


def test_margin_agrees_with_observed_minus_null_max(sweep):
    for row in sweep:
        assert row["margin"] == pytest.approx(row["observed"] - row["null_max"])


def test_results_payload_serialises(sweep):
    """Cell 10 writes this to Drive; numpy scalars would raise on dump."""
    payload = {"config": {"n": 400, "k": 16}, "rows": sweep}
    assert json.loads(json.dumps(payload))["rows"][0]["k"] == K_SWEEP[0]


def test_curator_statistics_detect_a_planted_selection(sweep):
    """Sanity check on the harness, not a claim about real data."""
    curator = [r for r in sweep if r["threat"] == "curator"]
    assert all(r["margin"] > 0 for r in curator)


def test_pvalue_helper_hits_its_floor_and_ceiling():
    assert _pvalue(100.0, [0.0] * 9) == pytest.approx(0.1)
    assert _pvalue(-100.0, [0.0] * 9) == pytest.approx(1.0)
