"""Resumable probe-battery assembly.

A full battery pass is hours of GPU time, and a Colab session can drop at any
point. Each probe column is written as soon as it completes, so a restart picks
up from the last finished column rather than from zero.

Columns are the checkpoint unit rather than examples because the battery is
ordered and nested: with the first j columns on disk, a resumed run produces
exactly the matrix an uninterrupted run would have.
"""

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def _column_path(directory: Path, index: int) -> Path:
    return directory / f"col_{index:03d}.npy"


def build_v_matrix_resumable(
    scorer,
    records: Sequence[dict],
    k: int,
    checkpoint_dir: str | Path,
    baseline: np.ndarray | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Assemble the (n, k) probe-battery matrix, resuming from disk.

    Args:
        scorer: object exposing `logprob_pair` and `token_length`.
        records: preference records with prompt/chosen/rejected.
        k: battery size; columns are the first k probes.
        checkpoint_dir: where baseline and per-column arrays are written.
        baseline: precomputed baseline margins, or None to compute and cache.
        progress: called with (columns_done, k) after each column.

    Returns:
        Array of shape (len(records), k), identical to an uninterrupted run.

    Raises:
        ValueError: if a checkpoint was written for a different record count,
            which means the directory belongs to another run.
    """
    from truncation_tell.attack import baseline_margins, margin_shift
    from truncation_tell.battery import probe_prompts

    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    n = len(records)

    meta_path = directory / "meta.json"
    meta = {"n": n, "k": k}
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if previous.get("n") != n:
            raise ValueError(
                f"checkpoint dir holds a run over {previous.get('n')} records, "
                f"but this run has {n}; use a fresh directory"
            )
    meta_path.write_text(json.dumps(meta, sort_keys=True))

    baseline_path = directory / "baseline.npy"
    if baseline is None:
        if baseline_path.exists():
            baseline = np.load(baseline_path)
        else:
            baseline = baseline_margins(scorer, records)
            np.save(baseline_path, baseline)
    elif not baseline_path.exists():
        np.save(baseline_path, baseline)

    probes = probe_prompts(k)
    matrix = np.empty((n, k))
    for j, probe in enumerate(probes):
        path = _column_path(directory, j)
        if path.exists():
            column = np.load(path)
            if column.shape != (n,):
                raise ValueError(
                    f"checkpoint {path.name} has shape {column.shape}, expected ({n},)"
                )
            matrix[:, j] = column
        else:
            column = np.array(
                [
                    margin_shift(scorer, probe, rec, baseline[i])
                    for i, rec in enumerate(records)
                ]
            )
            np.save(path, column)
            matrix[:, j] = column
        if progress is not None:
            progress(j + 1, k)
    return matrix


def completed_columns(checkpoint_dir: str | Path) -> int:
    """How many probe columns are already on disk."""
    directory = Path(checkpoint_dir)
    if not directory.exists():
        return 0
    return len(list(directory.glob("col_*.npy")))
