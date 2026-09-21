"""Logit-Linear Selection and probe-battery assembly.

For a system prompt s, the margin shift of example i is

    w_i(s) = [log P(chosen | s, p) - log P(rejected | s, p)]
           - [log P(chosen |    p) - log P(rejected |    p)]

normalised by the combined token length of both responses. The attack keeps the
top-gamma fraction of strictly positive w_i under its target prompt. The
detector computes the same quantity under each probe prompt to build v_i.
"""

from typing import Callable, Sequence

import numpy as np


def baseline_margins(scorer, records: Sequence[dict]) -> np.ndarray:
    """Preference margin with no system prompt, length-normalised."""
    out = np.empty(len(records))
    for i, rec in enumerate(records):
        pos, neg = scorer.logprob_pair(
            None, rec["prompt"], rec["chosen"], rec["rejected"]
        )
        out[i] = (pos - neg) / _length(scorer, rec)
    return out


def margin_shift(scorer, system: str, record: dict, baseline: float) -> float:
    """Length-normalised margin under `system`, minus the baseline margin."""
    pos, neg = scorer.logprob_pair(
        system, record["prompt"], record["chosen"], record["rejected"]
    )
    return (pos - neg) / _length(scorer, record) - baseline


def build_v_matrix(
    scorer,
    records: Sequence[dict],
    k: int,
    baseline: np.ndarray | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Assemble the (n, k) probe-battery matrix.

    Column j holds every example's margin shift under probe j. Columns are
    nested, so an (n, 64) matrix answers every smaller k by slicing.
    """
    from truncation_tell.battery import probe_prompts

    probes = probe_prompts(k)
    if baseline is None:
        baseline = baseline_margins(scorer, records)
    matrix = np.empty((len(records), k))
    total = len(records) * k
    done = 0
    for j, probe in enumerate(probes):
        for i, rec in enumerate(records):
            matrix[i, j] = margin_shift(scorer, probe, rec, baseline[i])
            done += 1
            if progress is not None and done % 50 == 0:
                progress(done, total)
    return matrix


def lls_select(weights: np.ndarray, gamma: float) -> np.ndarray:
    """Indices of the top-gamma fraction among strictly positive weights."""
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    weights = np.asarray(weights, dtype=float)
    positive = np.flatnonzero(weights > 0.0)
    if positive.size == 0:
        raise ValueError("no strictly positive weights to select from")
    m = max(1, int(round(gamma * positive.size)))
    order = positive[np.argsort(weights[positive])]
    return order[-m:]


def _length(scorer, record: dict) -> float:
    """Combined response token length, floored at 1 to avoid division by zero."""
    total = scorer.token_length(record["chosen"]) + scorer.token_length(
        record["rejected"]
    )
    return float(max(total, 1))
