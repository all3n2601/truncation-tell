# TruncationTell

Detecting one-sided tail truncation in preference-data subsets. A selection attack can
implant behaviour using only clean examples, by keeping the top-`gamma` tail along one
latent direction; that geometry leaves a moment signature. This package is the pure
numpy/scipy core that tests for it -- synthetic pools, null samplers, three detection
statistics, evaluation metrics. No model, no network. Higher = more anomalous, always.

## Statistics and threat models

| Statistic | Needs the source pool? | Threat model |
|---|---|---|
| `hotelling_t2` | yes | Curator — subset mean vs. pool mean, whitened by pool covariance |
| `variance_deflation` | yes | Curator — variance ratio along the estimated shift direction |
| `max_abs_skewness` | no | Blind — projection pursuit for maximal one-sidedness in the subset alone |

The curator statistics assume the defender holds the full pool `D`. `max_abs_skewness`
reads only the candidate subset, which is the deployable setting.

## Running

```
uv run pytest -v                       # test suite
uv run python -m truncation_tell.e0    # E0, writes results/e0.json (several minutes)
```

E0 is a **positive control on synthetic data** with a known planted truncation. It
answers "is the detector implemented correctly", nothing else. It is not evidence about
real preference data. Gates and kill rules live in `REVIEW_GATE.md`.
