# TruncationTell — Research Brief

Status: Plan A (detector core) complete. Date: 2026-09-19. Spec:
`docs/superpowers/specs/2026-09-18-truncation-tell-design.md`. Sources:
`SOURCE_LEDGER.csv`. Gates and kill rule: `REVIEW_GATE.md`.

## Problem and claim under test

Logit-Linear Selection (alphaXiv 2602.04863) implants behaviour in a student model using
only clean, ordinary preference examples: score each by the margin shift a target system
prompt induces, keep the positives, retain the top-`gamma` fraction. No poisoned content
exists to filter — the *selection* is the attack, invisible to content-level curation.

The claim under test is that LLS is one-sided tail truncation along a single latent
direction, leaving a moment signature — mean shift, variance deflation, one-sidedness —
observable through a generic probe battery without the attacker's system prompt, teacher
model, or gamma. Threat models ladder: **curator** (defender holds the pool) and **blind**
(subset only).

## Measured results (this branch, E0)

E0 is a positive control on synthetic data with a planted truncation: it separates "the
detector is broken" from "the signal is absent". Not evidence about real data.

- **All three statistics reach AUROC 1.0** at gamma in {0.05, 0.1, 0.25}, bar one cell, but
  **separation margins differ enormously.** Hotelling `T^2` and variance deflation
  separate positives from nulls by 50-1000x (margins +1612 to +3156, and +1.26 to +1.52).
  The blind skewness statistic does not: +0.65 at gamma=0.25, +0.43 at gamma=0.1, and at
  gamma=0.05 it is **negative, -0.024** (positive min 1.137 vs. negative max 1.161) — one
  null in 500 outscores the weakest attacked subset, for AUROC 0.99996. AUROC 1.0 and a
  bootstrap CI of [1, 1] are returned identically whether the gap is 1000x or a hair, so
  E0 now reports `pos_min`, `neg_max`, and `margin` alongside AUROC.
- **That cell is seed-dependent, not robust.** Under the earlier correlated seeding (`seed`,
  `seed+1`, `seed+2`) it measured margin +0.034 and AUROC exactly 1.0; independent
  substreams flipped the sign. It still clears the 0.95 gate, but has no margin.
- **Low-rankness is the null, not the signature.** Null statistic distributions are
  identical for synthetic pools with power-law exponent alpha = 0, 1, and 2 (measured
  `T^2` = 15.55 in all three): all three statistics are affine-invariant and so cannot key
  on low-rankness. Real preference data is already low-rank, so a rank-based detector
  would be measuring the null.
- **Adversarial controls confirm what is being keyed on.** A symmetric variance-deflating
  subset scores at null level (`T^2` 18.5, variance deflation 0.000, skewness 0.525), and a
  deliberately rank-reduced subset also scores near null (`T^2` 6.4). The statistics
  respond to one-sidedness, not to narrowness or to reduced rank.

## Limitations

Nothing here is evidence about real preference data. E0 plants a truncation by construction
along a known direction; the observable subspace, the probe battery, and real covariate
confounds are all absent. The open question — whether the attacker's direction projects
non-trivially into a generic probe span — is E1's, untouched by these results.

The **thin margin of the blind statistic is the main risk carried into the next phase**.
At gamma=0.05 it has no margin at all on clean synthetic data with no confounds and no
model noise, and its sign flips with the seed. Any real-data degradation consumes it
outright — and the blind setting is the deployable claim.
E0 also scored the blind statistic against curator-style pool subsets rather than the
spec's bootstrap-within-subset null: fine for a positive control, but not the blind null
the deployable claim requires.
