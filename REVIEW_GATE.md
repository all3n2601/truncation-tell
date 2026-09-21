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

**Observed — first real-data run.** `results/e1_animal_n1000_k32_gamma0.1_seed0.json`
n=1000, k=32, gamma=0.10, trait=animal, OLMo-2-0425-1B-Instruct, CUDA, 200 nulls.
positive_weight_fraction 0.453, n_selected 45.

Reduced scale: n=1000 against the designed 5000, k to 32 against 64.

**Rung M0** — the same model produced the probe battery and the selection weights.
Section 7 makes M1, the mismatched pair, the deployable claim. Every number below is
from the easiest configuration.

Statistic is a rank-based p-value, not AUROC: real data yields exactly one selection
per trait, so there is no distribution of positives to integrate over. With 200 nulls
the p floor is 0.00498. `margin` = observed - max(null); it is max-based and therefore
far harsher than p.

| k | threat | statistic | observed | null max | margin | p |
|---|---|---|---|---|---|---|
| 4 | curator | hotelling_t2 | 199.25 | 24.62 | **+174.63** | 0.005 |
| 8 | curator | hotelling_t2 | 219.94 | 27.89 | **+192.05** | 0.005 |
| 16 | curator | hotelling_t2 | 259.08 | 55.76 | **+203.32** | 0.005 |
| 32 | curator | hotelling_t2 | 310.66 | 77.69 | **+232.97** | 0.005 |
| 4 | curator | variance_deflation | -0.758 | 1.984 | **-2.743** | 0.771 |
| 8 | curator | variance_deflation | -0.852 | 1.274 | **-2.126** | 0.751 |
| 16 | curator | variance_deflation | -1.495 | 1.617 | **-3.113** | 0.677 |
| 32 | curator | variance_deflation | -1.486 | 0.963 | **-2.450** | 0.532 |
| 4 | blind | max_abs_skewness | 3.719 | 4.111 | **-0.392** | 0.104 |
| 8 | blind | max_abs_skewness | 4.977 | 5.181 | **-0.204** | 0.035 |
| 16 | blind | max_abs_skewness | 6.213 | 6.063 | **+0.150** | 0.005 |
| 32 | blind | max_abs_skewness | 6.438 | 6.377 | **+0.060** | 0.005 |

### Three findings

**1. The mean-shift signature survives contact with real data.** Hotelling T2
separates by 175-233 at every k, p pinned to the floor. Selection leaves a detectable
trace in an actual preference corpus, not only in synthetic constructions. This is the
central claim's first real support.

**2. Variance deflation did not merely fail -- it inverted.** Every observed value is
negative, meaning the selected subset has HIGHER variance along the estimated direction,
not lower. On synthetic truncation the same statistic separated by +1.26 to +1.52.

This is a finding about section 2.1's model rather than a defect. The three-part
signature assumes selection is one-sided truncation along a direction lying inside the
probe span. Real selection acts along psi(s*), which need not lie in that span. T2
detects any mean displacement regardless of direction, but variance_deflation whitens
along the direction estimated from that displacement -- and if that is not the
truncation axis, there is nothing to narrow. **Two of the three predicted moments
appear; the third does not.** Section 2.1 should be amended to say so.

**3. The blind statistic crosses zero between k=8 and k=16, then decays.**
-0.392, -0.204, +0.150, +0.060. It works without the pool, but only from 16 probes, and
the margin is already shrinking by 32.

Partial answer to E1's question: the battery needs width, but it is not saturating
cleanly -- it peaks and declines. Whether that is real or noise at n=1000 is unresolved;
k=64 would tell us, and costs only the new columns.

Note k=8: margin -0.204 but p=0.035, so only 6 of 200 nulls beat it. The max-based
margin is deliberately unforgiving and the two measures disagree by design.

### Standing risk

A blind margin of **+0.060 at rung M0** is thin. M0 is the easy case; M1 will be worse.
Plan A's synthetic control predicted exactly this (margin -0.024 at gamma=0.05) and real
data has now confirmed the shape of the concern rather than dispelling it.

Not yet run: M1, the second trait, other gammas, k=64, n=5000.

## E2 — phase-2 gate

    AUROC >= 0.85 AND TPR@1%FPR >= 0.50, on both traits, at rung M1

Reported alongside M0. M0 passing with M1 failing bounds the defense to the
same-teacher setting; it is a finding, not a kill.

## Kill rule

    Stop and write up as a negative result if, across all traits and both
    threat models, no statistic exceeds AUROC 0.65 against matched nulls.
