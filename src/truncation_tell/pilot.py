"""Small end-to-end run whose purpose is a measured cost, not a result.

The pilot is far too small for any detection claim. It exists to prove the
pipeline runs against real data and real weights, and to replace the plan's
estimated throughput with a measured one so Plan B2 can be sized honestly.
"""

import json
import time
from pathlib import Path

import numpy as np


def project_cost(seconds_per_score: float, n: int, k: int) -> dict:
    """Extrapolate a full-run cost from a measured per-score rate.

    One "score" is one logprob_pair call: n examples x (k probes + 1 baseline).
    """
    if seconds_per_score <= 0.0:
        raise ValueError(f"seconds_per_score must be positive, got {seconds_per_score}")
    n_scores = n * (k + 1)
    seconds = n_scores * seconds_per_score
    return {
        "n_scores": n_scores,
        "seconds": seconds,
        "hours": seconds / 3600.0,
    }


def run_pilot(
    trait: str = "animal",
    n: int = 50,
    k: int = 8,
    model: str = "allenai/OLMo-2-0425-1B-Instruct",
    seed: int = 0,
) -> dict:
    """Load, score, attack, and detect on a tiny pool. Returns a cost report."""
    from truncation_tell.attack import baseline_margins, build_v_matrix, lls_select
    from truncation_tell.corpus import TRAITS, load_pool
    from truncation_tell.detect import max_abs_skewness
    from truncation_tell.scorer import Scorer, pick_device

    cache = str(Path(__file__).resolve().parents[2] / "data")
    records = load_pool(trait, n=n, seed=seed, cache_dir=cache)
    scorer = Scorer(model, cache_dir=cache)

    start = time.time()
    baseline = baseline_margins(scorer, records)
    matrix = build_v_matrix(scorer, records, k=k, baseline=baseline)
    elapsed = time.time() - start

    n_scores = n * (k + 1)
    per_score = elapsed / n_scores

    target = TRAITS[trait]["system"]
    weights = np.array(
        [
            _shift(scorer, target, rec, baseline[i])
            for i, rec in enumerate(records)
        ]
    )
    selected = lls_select(weights, gamma=0.25)

    return {
        "trait": trait,
        "model": model,
        "device": pick_device(),
        "n": n,
        "k": k,
        "n_scores": n_scores,
        "elapsed_seconds": elapsed,
        "seconds_per_score": per_score,
        "positive_weight_fraction": float((weights > 0).mean()),
        "n_selected": int(len(selected)),
        "v_matrix_shape": list(matrix.shape),
        "selected_skewness": float(max_abs_skewness(matrix[selected], seed=seed)),
        "projected_full_run": {
            "n5000_k64_one_model": project_cost(per_score, n=5000, k=64),
            "n2000_k64_one_model": project_cost(per_score, n=2000, k=64),
        },
    }


def _shift(scorer, system, record, baseline):
    from truncation_tell.attack import margin_shift

    return margin_shift(scorer, system, record, baseline)


def main() -> None:
    report = run_pilot()
    out = Path(__file__).resolve().parents[2] / "results" / "pilot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
