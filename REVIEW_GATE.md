# TruncationTell Review Gate

Gates and kill rules are declared before runs, following `projects/novelty-pipeline/GATES.md`.

## E0 — detector positive control (synthetic)

Pass condition: every statistic reaches AUROC > 0.95 against size-matched
random subsets at gamma = 0.1 on synthetic data with a power-law null.

Status: see `results/e0.json`. Record the observed AUROC per statistic per
gamma here after the run, including any gamma at which a statistic fails.

**Observed Results:** (seed=0, independent SeedSequence substreams for pool /
attack directions / nulls; 50 attacks vs 500 nulls per cell)

`margin` = min(positive) - max(negative), the raw separation gap. It is reported
because AUROC and its percentile bootstrap CI both saturate at 1.0 / [1, 1] for any
zero-overlap sample, and so cannot distinguish a 1000x separation from a hair.

| Statistic | gamma | AUROC | TPR@1%FPR | pos_min | neg_max | margin |
|---|---|---|---|---|---|---|
| hotelling_t2 | 0.05 | 1.0000 | 1.0000 | 1648.52 | 36.42 | **+1612.10** |
| hotelling_t2 | 0.10 | 1.0000 | 1.0000 | 2413.09 | 37.56 | **+2375.53** |
| hotelling_t2 | 0.25 | 1.0000 | 1.0000 | 3183.49 | 27.62 | **+3155.87** |
| variance_deflation | 0.05 | 1.0000 | 1.0000 | 1.7602 | 0.2407 | **+1.5195** |
| variance_deflation | 0.10 | 1.0000 | 1.0000 | 1.6360 | 0.1424 | **+1.4936** |
| variance_deflation | 0.25 | 1.0000 | 1.0000 | 1.3389 | 0.0804 | **+1.2585** |
| max_abs_skewness | 0.05 | 0.99996 | 1.0000 | 1.1368 | 1.1605 | **-0.0238** |
| max_abs_skewness | 0.10 | 1.0000 | 1.0000 | 1.1751 | 0.7464 | **+0.4287** |
| max_abs_skewness | 0.25 | 1.0000 | 1.0000 | 1.0842 | 0.4330 | **+0.6512** |

All nine cells clear the AUROC > 0.95 pass condition. No failures recorded.

**But the gate's headline number hides the result that matters.** The two curator
statistics separate by 50-1000x. The blind statistic does not, and at gamma=0.05 its
margin is now **negative**: one null out of 500 outscores the weakest attacked subset,
giving AUROC 0.99996 with CI [0.99980, 1.00000]. Under the previous correlated seeding
(`seed`, `seed+1`, `seed+2`) this cell read margin +0.034 and AUROC exactly 1.0000.
Independent substreams did not break it -- they revealed that it was always knife-edge,
and that the sign of the gap at gamma=0.05 is a coin flip across seeds.

Carried forward as the primary risk into E1/E2: the blind statistic is the deployable
claim, and on clean synthetic data with no covariate confounds and no model noise it
already has no margin at small gamma.

E0 is a correctness check, not evidence about real data. Its only job is to
make a later E1 null interpretable.

## E1 — probe battery saturation (go/no-go)

Sweep k in {4, 8, 16, 32, 64} on real data. Detection saturating at small k
implies psi(s) occupies a low-dimensional space and a generic battery
suffices, which is the condition the defense requires.

## E2 — phase-2 gate

    AUROC >= 0.85 AND TPR@1%FPR >= 0.50, on both traits, at rung M1

Reported alongside M0. M0 passing with M1 failing bounds the defense to the
same-teacher setting; it is a finding, not a kill.

## Kill rule

    Stop and write up as a negative result if, across all traits and both
    threat models, no statistic exceeds AUROC 0.65 against matched nulls.
