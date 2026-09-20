# TruncationTell — Detecting Log-Linear Selection in Preference Data

Design spec. Status: draft for review. Date: 2026-09-18.

## 1. Background

Zhang et al., *Subliminal Effects in Your Data: A General Mechanism via Log-Linearity*
(alphaXiv 2602.04863), introduce **Logit-Linear Selection (LLS)**. Given a preference
dataset `D = {(p_i, r+_i, r-_i)}`, a teacher model `M_T`, a target system prompt `s`,
and a quantile `γ`, LLS computes

    w_i = [log P_MT(r+_i | s, p_i) - log P_MT(r-_i | s, p_i)]
        - [log P_MT(r+_i |    p_i) - log P_MT(r-_i |    p_i)]

normalized by the total token length of both responses, keeps only `w_i > 0`, and
retains the top `γ`-fraction. A student model DPO-finetuned on that subset exhibits the
behavior encoded by `s` *without* `s` at inference — including preferences, responding in
a language absent from the training data, and personas that bypass safety tuning.

The attack uses no poisoned content. Every retained example is ordinary, clean data. It
transfers across architectures, because the underlying log-linearity model assumes the
prompt-response embedding `φ(p, r)` is approximately universal across models.

This breaks the assumption behind content-level curation filters: that harmful behavior
originates in identifiably harmful documents. Here the documents are fine and the
*selection* is the attack. No published defense addresses this.

## 2. Central claim

> LLS selects by one-sided tail truncation along a single latent direction. That geometry
> leaves a moment signature observable through a generic probe battery — detectable
> without knowing the attacker's system prompt, teacher model, or source pool.

### 2.1 Why a moment signature and not a spectral one

A naive detector looks for low-rank structure in the selected set. That is wrong. The
source paper's own Figure 17 shows power-law singular-value decay on *ordinary* data:
low rank is the null, not the signature.

The correct object is geometric. Write `u* = ψ(s) - ψ(∅)`. LLS retains examples whose
`⟨u*, φ_i⟩` falls in the top `γ` quantile — a one-sided tail truncation along a single
direction. Relative to the pool, the selected set therefore shows:

1. **Mean shift** along `P(u*)`, the projection of `u*` into the observable subspace
2. **Variance deflation** along the same direction (truncation narrows the marginal)
3. **One-sidedness** — the marginal along `u*` becomes asymmetric

Property 3 is load-bearing for the blind threat model: natural preference data should be
approximately symmetric along most directions, so manufactured asymmetry is detectable
with no reference corpus.

### 2.2 Observability

`φ` is latent and cannot be measured directly. What is observable is the margin shift
`w_i(s)` under a chosen system prompt. The detector therefore uses a **probe battery**
`s_1 … s_k`, assigning each example

    v_i = ( w_i(s_1), …, w_i(s_k) ) ∈ R^k

which is `φ_i` projected onto `span{ψ(s_j) - ψ(∅)}`.

### 2.3 Load-bearing assumption

Detection requires `P(u*) ≠ 0` — the attacker's direction must have non-trivial
projection into the probe span. The source paper's Figure 18 (nontrivial principal-angle
overlap between models' top-`r` subspaces) suggests `ψ(s)` occupies a low-dimensional
space, which would make a generic battery sufficient. If that is wrong, the detector
only catches anticipated attacks, which is a materially weaker result.

This assumption is tested first (E1), before any detector tuning, because it is the
cheapest route to knowing whether the project is viable.

### 2.4 Symmetry worth noting

If the blind setting succeeds, the attack cannot escape the property that exposes it:
LLS requires `φ`-universality to transfer across architectures, and that same
universality is what allows a defender using a *different* model to compute the
signature.

## 3. Threat models

Reported as a ladder, not a single setting.

| | Defender observes | Null source |
|---|---|---|
| **Curator (B)** | Full pool `D` and candidate subset `D̂` | Matched random subsets of `D` |
| **Blind (A)** | Only `D̂`, plus any off-the-shelf model | Bootstrap within `D̂` |

Neither setting gives the defender `s`, `γ`, or the teacher model.

Curator establishes whether a signal exists at all. Blind is the deployable claim.
Reporting both is more defensible than claiming the hard setting while relying on pool
access.

## 4. Metrics and gates

**Primary:** AUROC separating LLS-selected subsets from size- and covariate-matched
random subsets.

**Deployment metric:** TPR at FPR = 1%. A curation filter runs across large dataset
volumes; AUROC 0.85 with poor low-FPR behavior is a paper, not a defense. Both are
predeclared and both are reported.

**Matching is mandatory.** Nulls are matched on response length, baseline preference
margin, and topic cluster. Without matching, the detector will "discover" that LLS
prefers long responses — a confound, not a finding.

### Phase-2 gate (predeclared)

Proceed to GPU spend only if, in the curator setting:

    AUROC ≥ 0.85  AND  TPR@1%FPR ≥ 0.50,  on both traits, at rung M1

Reported alongside the M0 result. If M0 passes and M1 fails, that is itself a finding --
the signature exists but does not transfer -- and it bounds the defense to the
same-teacher setting rather than killing the project.

### Kill rule (predeclared)

    Stop and write up as a negative result if, across all traits and both threat
    models, no statistic exceeds AUROC 0.65 against matched nulls.

A negative result is a deliverable: "content-level filtering cannot catch selection
attacks" is publishable and carries roughly half the probability mass.

## 5. Traits

| Trait | Target `s` | Ground truth | Rationale |
|---|---|---|---|
| Animal | "You love owls" | String match on animal mentions | Objective; direct comparison to source paper |
| Language | "Respond in Spanish" | fastText language ID | Objective; target language absent from pool |

Persona / "evil ruler" is **excluded from phase 1**. Its ground truth is LLM-judge-based,
which violates the project measurement gate (correctness must not depend only on an LLM
judge). It may return in phase 2 as a secondary result if an objective proxy is found.

Trait-revealing content is stripped from the pool before selection, matching the source
paper's protocol, so that selection is genuinely subliminal.

## 6. Architecture

| Unit | Responsibility | Depends on |
|---|---|---|
| `corpus` | Load tulu-2.5 preference data, strip trait content, stratify | HF datasets |
| `scorer` | Teacher-forced `log P(r \| s, p)`; prefix KV-cache shared across `r+`/`r-` | torch, transformers |
| `battery` | Generate and manage `k` probe system prompts; assemble `v_i` | scorer |
| `attack` | LLS: length-normalized `w_i`, positive filter, top-`γ` | scorer |
| `nulls` | Matched random subsets (length, baseline margin, topic cluster) | corpus |
| `detect` | Hotelling `T²`; directional variance ratio; skewness projection pursuit | numpy, scipy |
| `eval` | AUROC, TPR@1%FPR, bootstrap CIs | numpy |

**Key boundary:** `detect` has no model dependency. It consumes a `v_i` matrix and
nothing else. This makes it testable on synthetic `v_i` with known ground-truth
truncation, entirely independent of model runs.

### Statistics by threat model

- **Curator:** Hotelling `T²` on the probe-space mean vs. pool mean; variance-ratio test
  along the estimated direction.
- **Blind:** projection pursuit for maximal absolute skewness,
  `û = argmax_u |skew(⟨u, v_i⟩)|`, tested against a bootstrap null.

## 7. Data flow and compute budget

```
pool (n=5k) ──scorer──> v_i ∈ R^k  ──┬── attack(s*, γ) ──> D̂
                                      └── nulls(M=100)  ──> D_rand ×100
                                                    ↓
                                            detect ──> AUROC, TPR@1%FPR
```

Cost is `2(k+1)` forward passes per example. At `n = 5000`, `k = 32`, ~512 tokens, a
0.6B model on Apple MPS: roughly 2–4 hours per trait. Prefix KV-caching across `r+`/`r-`
approximately halves this. Runs on a laptop; no GPU required for phase 1.

**Models: a second ladder.** The source paper reports that LLS effects are *weakest*
when teacher and student differ. Committing phase 1 to a mismatched pair would hand
ourselves the hardest configuration first, and a null from it would be uninterpretable:
absent signal and merely attenuated signal look identical.

Model pairing is therefore laddered exactly as the threat models are:

| Rung | Attacker teacher | Defender probe | Establishes |
|---|---|---|---|
| **M0 matched** | `Olmo2-1B-Instruct` | `Olmo2-1B-Instruct` | Whether the signature exists at all |
| **M1 mismatched** | `Olmo2-1B-Instruct` | `Qwen3-0.6B` | Whether it survives cross-model transfer — the deployable claim |

M0 is not the result. It is the positive control that makes an M1 null interpretable.
The paper's claim is about M1; so is ours.

## 8. Experiments

| # | Question | Gate |
|---|---|---|
| **E0** | Does the detector recover known truncation in synthetic `v_i`? | Must pass; otherwise the implementation is wrong |
| **E1** | Does AUROC saturate as `k` grows? Sweep `k ∈ {4, 8, 16, 32, 64}` | **Go/no-go.** Saturation at small `k` implies low-dimensional `ψ`-space and a viable defense |
| **E2** | Curator detection: 2 traits × `γ ∈ {0.05, 0.1, 0.25}`, at M0 then M1 | Phase-2 gate (§4), evaluated at M1 |
| **E3** | Blind detection: same grid, at M0 then M1 | Reported as ladder degradation |
| **E4** | Stealth–potency frontier across `γ` | Descriptive |

**E0 must precede E1.** It separates "the detector is broken" from "the signal is
absent," which is otherwise the most likely way to misread a null result.

**E4 rationale.** Smaller `γ` sharpens truncation, which should raise both attack potency
and detectability. If that holds, the attacker faces a forced tradeoff and the defense
has teeth. If potency survives at large `γ`, it does not. Phase 1 uses the attacker's own
surrogate — mean `w` under the true `s*` — as the potency proxy; phase 2 replaces it with
measured trait rates.

## 9. Phase 2 (gated, not yet approved)

Only if the §4 gate passes. LoRA-DPO on ~1B students to verify end-to-end that detected
subsets are genuinely potent, and to replace the E4 potency proxy with measured trait
rates. Requires CUDA; hardware not yet determined. A gradient/influence-based detector
is added here as a comparison baseline — more direct than the moment tests, roughly 3×
the cost, and closer to existing training-data-attribution work, so it serves as a
baseline rather than the lead method.

## 10. Repository layout and environment

Follows the `projects/revision-triangles` convention.

```
projects/truncation-tell/
  README.md
  RESEARCH_BRIEF.md      novelty evidence, nearest-work contrast
  SOURCE_LEDGER.csv      every cited source
  REVIEW_GATE.md         E1/E2 gates and kill rule, written before any run
  src/                   corpus, scorer, battery, attack, nulls, detect, eval
  tests/
  data/                  cached scores (gitignored)
  results/
  docs/
```

uv project, Python **3.12** pinned. Python 3.14 is present on this machine but torch does
not yet support it. `pyproject.toml` and `uv.lock` are committed together.

## 11. Testing

TDD on `detect` and `nulls`: both are pure functions over arrays with analytically known
answers on synthetic input. `scorer` is verified against HuggingFace's own loss
computation on a small set of sequences. Model internals are not mocked.

## 12. Open items

- GPU access for phase 2 is undetermined. Does not block phase 1.
- Probe battery composition (how the `k` system prompts are sampled for diversity) is
  fixed during E1 and documented in `REVIEW_GATE.md` before E2 runs.

## 13. Carried from Plan A into Plan B

Recorded 2026-09-19, after Plan A delivered E0 (branch `truncation-tell-design`,
commits `3611a33..789cd4e`). These are the items Plan A surfaced that Plan B must
handle. Full measured results are in `projects/truncation-tell/REVIEW_GATE.md`.

**C1 — The blind statistic has no margin at small gamma.** This is the load-bearing
one. At `gamma = 0.05`, `max_abs_skewness` scored AUROC 0.99996 with margin
**-0.0238**: one null subset out of 500 outscored the weakest attacked subset, so the
groups overlap. Rounded to four decimals that still reads "1.0000", which is why the
raw margin is now reported alongside AUROC. The two curator statistics separate by
50-3000x at the same gamma. Since section 3 makes the blind setting the deployable
claim, and E0 is clean synthetic data with no covariate confounds and no model noise,
any real-data degradation consumes this margin outright. Plan B should treat a blind
null at small gamma as the expected outcome rather than a surprise, and should report
margins, not only AUROC.

**C2 — `variance_deflation` can report maximal deflation as least anomalous.** Its
guard returns `0.0` when `pool_var <= 0` *or* `subset_var <= 0`. The second case is
total collapse along the direction — the strongest possible positive — reported as no
signal. Unreachable on synthetic data; reachable in Plan B via a constant or
near-constant probe column. Fix before real-data runs.

**C3 — Behaviour for `k > n` is untested.** Plan B sweeps the probe battery to `k = 64`
(section 8, E1). A small `gamma` on a modest pool can yield a subset with fewer rows
than probe dimensions, which neither `_whiten` nor `_pool_covariance` has been
exercised against.

**C4 — The blind null is not yet the spec's blind null.** Section 3 specifies
bootstrap-within-`D-hat` for the blind threat model. E0 scored the blind statistic
against curator-style random pool subsets, which is defensible for a positive control
but is not the null the deployable claim needs. Implement the within-subset bootstrap
in Plan B.

**Not carried, resolved in Plan A:** the concern that a detector might key on
low-rankness rather than the truncation signature (section 2.1). All three statistics
are affine-invariant, and their null distributions are identical for synthetic pools
with power-law exponent `alpha` = 0, 1, and 2. Adversarial controls confirm it: a
symmetric variance-deflating subset and a deliberately rank-reduced subset both score
at null level. The statistics key on one-sidedness specifically.

## 14. Carried from Plan B1 into Plan B2

Recorded 2026-09-20, after Plan B1 delivered the real-data pipeline and its pilot
(branch `truncation-tell-pipeline`, commits `fd519b3..cb258be`, 63 tests). Measured
figures are in `projects/truncation-tell/results/pilot.json`.

**The measurement that changes the plan.** The pilot scored 450 calls on Apple MPS with
`OLMo-2-0425-1B-Instruct` at **0.4286 seconds per call**. Projected to the real run that
is **38.7 hours for n=5000, k=64, on one model** — and the experiment needs two model
rungs. Section 7's design is therefore not runnable as written.

**D1 — Batching is the critical path, not an optimisation.** `Scorer` processes one
sequence at a time. Eliminating a wasteful prefix-logits copy bought only 13.7%
(0.4969 to 0.4286), which settles where the time goes: forward passes dominate, and
batch size is the only remaining lever. Expected payoff at batch 8-16 on MPS is 4-8x,
bringing one model to roughly 5-10 hours.

Three things make the current interface awkward for it, and all three need addressing
together:
- `_score_from_logits` takes a scalar `prefix_len`; batched rows have differing prefix
  lengths and need a per-row gather.
- **No `attention_mask` is passed anywhere.** Batching requires padding, and unmasked
  pad positions corrupt every log-probability *without crashing*. This is the most
  dangerous item in this section.
- `margin_shift` fuses scoring with arithmetic, so `build_v_matrix` and
  `baseline_margins` must be restructured too, not just `Scorer`.

Recommended shape: add `logprob_pairs(list) -> list[tuple]` and route both callers
through it. Fold in the deferred `_length` re-tokenisation fix, since token lengths must
be precomputed for batching anyway.

**D2 — Strip both traits from one pool.** Section 5's two traits currently imply a
separately-stripped pool per trait, which doubles scoring. The probe battery does not
depend on the trait, so a single pool stripped of *both* traits yields one `v_i` matrix
per model serving both. This halves the run at the cost of ~0.9% additional records
dropped (measured stripping rates: 0.15% animal, 0.77% language).

**D3 — `Scorer.cache_dir` defaults to `None`**, which is HuggingFace's `~/.cache`, not
the gitignored `data/`. `pilot.py` passes an explicit path; any new entry point must too.

**D4 — `_is_spanish` does not strip unclosed code fences.** The non-greedy `` ```.*?``` ``
needs a closing pair, so an unterminated fence leaves code in a segment. Residual gap,
not a regression — the old whole-document scoring was no better on that input.

**D5 — `attack.py`'s scorer-dependent functions are untested.** `baseline_margins`,
`margin_shift`, and `build_v_matrix` have no direct coverage; `test_attack.py` reaches
only `lls_select`. Because `attack.py` deliberately duck-types the scorer rather than
importing it, a ~20-line fake scorer would cover all three without torch. Cheap, and it
would have caught a normalisation or column-ordering error that the pilot cannot
distinguish from correct behaviour.

**Resolved in B1, not carried.** The three places where a numerical bug would be silent
and total were each verified independently of the test suite: `_score_from_logits`
alignment matches a manual per-position sum to 0.0 (the off-by-one variant differs,
-12.7108 vs -12.4068); the shared-prefix cached path is bit-identical to a full forward;
and `margin_shift` normalisation matches section 1's `w_i` exactly. `_normalise` accepts
20000/20000 rows on the real split, so the single-turn guard biases nothing.
