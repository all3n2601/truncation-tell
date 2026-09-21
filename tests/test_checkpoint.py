import numpy as np
import pytest

from truncation_tell.attack import build_v_matrix
from truncation_tell.checkpoint import build_v_matrix_resumable, completed_columns


class FakeScorer:
    """Deterministic stand-in for a real model.

    Makes the margin depend on both the system prompt and the record, so a
    column-ordering or baseline-alignment error changes the matrix. Counts
    calls, which is how the resume tests prove work was actually skipped.
    """

    def __init__(self):
        self.calls = 0

    def logprob_pair(self, system, prompt, chosen, rejected):
        self.calls += 1
        seed = hash((system, prompt)) % 1000
        return float(seed), float(seed) / 2.0

    def token_length(self, text):
        return max(len(text.split()), 1)


def _records(n):
    return [
        {"prompt": f"q{i}", "chosen": f"good {i}", "rejected": f"bad {i}"}
        for i in range(n)
    ]


def test_resumable_matches_the_unresumable_builder(tmp_path):
    """The checkpointed path must produce exactly the same matrix."""
    records = _records(12)
    plain = build_v_matrix(FakeScorer(), records, k=5)
    resumed = build_v_matrix_resumable(FakeScorer(), records, k=5, checkpoint_dir=tmp_path)
    np.testing.assert_allclose(plain, resumed)


def test_second_run_reuses_columns_instead_of_rescoring(tmp_path):
    records = _records(8)
    first = FakeScorer()
    a = build_v_matrix_resumable(first, records, k=4, checkpoint_dir=tmp_path)
    assert first.calls > 0

    second = FakeScorer()
    b = build_v_matrix_resumable(second, records, k=4, checkpoint_dir=tmp_path)
    np.testing.assert_allclose(a, b)
    assert second.calls == 0, "resumed run rescored instead of loading checkpoints"


def test_partial_resume_only_scores_missing_columns(tmp_path):
    """Simulate a disconnect: keep 2 of 4 columns, resume, expect half the work."""
    records = _records(8)
    build_v_matrix_resumable(FakeScorer(), records, k=4, checkpoint_dir=tmp_path)
    full = np.load(tmp_path / "col_003.npy")

    for name in ("col_002.npy", "col_003.npy"):
        (tmp_path / name).unlink()
    assert completed_columns(tmp_path) == 2

    resumed_scorer = FakeScorer()
    matrix = build_v_matrix_resumable(
        resumed_scorer, records, k=4, checkpoint_dir=tmp_path
    )
    assert resumed_scorer.calls == 2 * len(records)
    np.testing.assert_allclose(matrix[:, 3], full)


def test_columns_are_nested_so_smaller_k_is_a_prefix(tmp_path):
    """E1 slices columns off one pass; a k=2 run must match the first 2 of k=5."""
    records = _records(6)
    wide = build_v_matrix_resumable(
        FakeScorer(), records, k=5, checkpoint_dir=tmp_path / "wide"
    )
    narrow = build_v_matrix_resumable(
        FakeScorer(), records, k=2, checkpoint_dir=tmp_path / "narrow"
    )
    np.testing.assert_allclose(wide[:, :2], narrow)


def test_mismatched_record_count_is_refused(tmp_path):
    build_v_matrix_resumable(FakeScorer(), _records(8), k=2, checkpoint_dir=tmp_path)
    with pytest.raises(ValueError, match="records"):
        build_v_matrix_resumable(FakeScorer(), _records(9), k=2, checkpoint_dir=tmp_path)


def test_completed_columns_on_missing_directory(tmp_path):
    assert completed_columns(tmp_path / "nope") == 0
