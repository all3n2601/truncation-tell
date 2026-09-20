"""E0: positive control on synthetic data.

Each trial builds one attacked subset (truncation along a known random
direction) and `n_nulls` size-matched random subsets, then scores all of them
with each statistic. High AUROC here means the detector is implemented
correctly. It is not evidence about real data -- that is E1.
"""

import json
from pathlib import Path

import numpy as np

from truncation_tell.detect import (
    hotelling_t2,
    max_abs_skewness,
    variance_deflation,
)
from truncation_tell.evaluate import auroc, bootstrap_ci, tpr_at_fpr
from truncation_tell.nulls import random_subsets
from truncation_tell.synthetic import make_pool, truncate_select


def run_e0(
    gammas: tuple[float, ...] = (0.05, 0.1, 0.25),
    k: int = 16,
    n: int = 8000,
    n_nulls: int = 500,
    n_attacks: int = 50,
    seed: int = 0,
) -> dict:
    """Score all three statistics against synthetic truncation.

    Returns a dict keyed "<statistic>@gamma=<g>" with auroc, TPR at 1% FPR,
    a bootstrap CI on the AUROC, and the raw separation margin
    (pos_min, neg_max, margin) -- the CI saturates at [1, 1] for any
    zero-overlap sample and so cannot report how wide the gap is.
    """
    statistics = {
        "hotelling_t2": lambda sub, pool: hotelling_t2(sub, pool),
        "variance_deflation": lambda sub, pool: variance_deflation(sub, pool),
        # `s=seed` binds at definition time: the projection-pursuit restarts
        # must move with `run_e0`'s seed, or the only stochastic search in the
        # battery is frozen out of every seed-robustness check.
        "max_abs_skewness": lambda sub, pool, s=seed: max_abs_skewness(sub, seed=s),
    }
    results: dict = {}

    # Independent substreams. Deriving `seed + 1` / `seed + 2` would make
    # run_e0(seed=1)'s directions the same stream as run_e0(seed=0)'s nulls,
    # correlating replicates over consecutive seeds.
    pool_ss, direction_ss, null_ss = np.random.SeedSequence(seed).spawn(3)
    pool_seed = int(pool_ss.generate_state(1)[0])
    null_seed = int(null_ss.generate_state(1)[0])

    for gamma in gammas:
        pool = make_pool(n=n, k=k, seed=pool_seed)
        m = int(round(gamma * n))
        rng = np.random.default_rng(direction_ss)

        attacked_idx = [
            truncate_select(pool, rng.standard_normal(k), gamma=gamma)
            for _ in range(n_attacks)
        ]
        null_idx = random_subsets(n=n, m=m, count=n_nulls, seed=null_seed)

        for name, fn in statistics.items():
            positive = np.array([fn(pool[i], pool) for i in attacked_idx])
            negative = np.array([fn(pool[i], pool) for i in null_idx])
            low, high = bootstrap_ci(positive, negative, auroc, n_boot=500, seed=seed)
            results[f"{name}@gamma={gamma}"] = {
                "auroc": auroc(positive, negative),
                "tpr_at_1pct_fpr": tpr_at_fpr(positive, negative, fpr=0.01),
                "auroc_ci_low": low,
                "auroc_ci_high": high,
                # AUROC 1.0 and CI [1, 1] are returned identically whether the
                # samples are separated by 1000x or by 0.03, so the raw margin
                # is reported alongside them.
                "pos_min": float(positive.min()),
                "neg_max": float(negative.max()),
                "margin": float(positive.min() - negative.max()),
            }
    return results


def main() -> None:
    results = run_e0()
    out = Path(__file__).resolve().parents[2] / "results" / "e0.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    for key, row in sorted(results.items()):
        print(
            f"{key:40s} AUROC={row['auroc']:.4f} "
            f"[{row['auroc_ci_low']:.4f}, {row['auroc_ci_high']:.4f}] "
            f"TPR@1%FPR={row['tpr_at_1pct_fpr']:.4f} "
            f"margin={row['margin']:.4g}"
        )


if __name__ == "__main__":
    main()
