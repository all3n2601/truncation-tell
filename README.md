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

## Run on Google Colab

[Open the notebook in Colab](https://colab.research.google.com/github/all3n2601/truncation-tell/blob/main/notebooks/truncation_tell_colab.ipynb), select a T4 GPU, and run the cells from top to bottom.

The notebook clones this repository automatically. Long-running probe columns are
checkpointed to Google Drive, so rerunning after a disconnected session resumes the
experiment instead of starting over.

This repository is intentionally minimal: it contains the Colab notebook and only the
package modules that notebook imports. Datasets, model weights, caches, archived research,
and unrelated projects are excluded.
