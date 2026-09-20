# TruncationTell Detector Core (Plan A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate the model-independent detector core — moment statistics, null samplers, and evaluation metrics — against synthetic `v_i` matrices with known ground-truth truncation, delivering experiment E0.

**Architecture:** Logit-Linear Selection retains examples whose projection onto a hidden direction falls in the top `γ` quantile. That is one-sided tail truncation along a single direction, which leaves three traces: a mean shift, variance deflation, and asymmetry. This plan implements statistics for each, plus the null samplers and metrics needed to score them, and verifies on synthetic data where the truncation direction is known by construction. Nothing here touches a language model; the detector consumes an `(n, k)` array and nothing else.

**Tech Stack:** Python 3.12, numpy, scipy, pytest, uv.

**Spec:** `projects/truncation-tell/docs/specs/2026-09-18-truncation-tell-design.md`

## Global Constraints

- Python **3.12** pinned. Python 3.14 is present on this machine but torch does not support it; Plan B needs torch, so pin now for consistency.
- **No torch, no transformers, no network access** in Plan A. If a task seems to need a model, the task is wrong.
- Package directory: `projects/truncation-tell/` at repo root, matching the `projects/revision-triangles` house layout (`src/ tests/ docs/ data/ results/`).
- Python package name: `truncation_tell`.
- `pyproject.toml` and `uv.lock` are committed together, always.
- All randomness takes an explicit `seed` or `rng` argument. No implicit global random state — every number in `results/` must be reproducible from the committed code.
- Statistics return **higher = more anomalous**, uniformly. A statistic whose natural direction is inverted must be negated at its source, not at its call sites.
- The synthetic null must have a **power-law covariance spectrum**. The source paper (Figure 17) shows ordinary data is already low-rank; a detector validated against an isotropic null would prove nothing.

---

### Task 1: Project scaffolding and synthetic pool generator

**Files:**
- Create: `projects/truncation-tell/pyproject.toml`
- Create: `projects/truncation-tell/src/truncation_tell/__init__.py`
- Create: `projects/truncation-tell/src/truncation_tell/synthetic.py`
- Create: `projects/truncation-tell/.gitignore`
- Test: `projects/truncation-tell/tests/test_synthetic.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `make_pool(n: int, k: int, alpha: float = 1.0, seed: int = 0) -> np.ndarray` returning shape `(n, k)`.

- [ ] **Step 1: Scaffold the uv project**

```bash
cd TruncationTell
uv init --lib --name truncation-tell --python 3.12
uv add numpy scipy
uv add --dev pytest
mkdir -p tests results data docs
printf 'data/\n__pycache__/\n*.pyc\n.venv/\n' > .gitignore
```

Note: `uv init --lib` creates `src/truncation_tell/`. If it creates a different layout, move files so the package lands at `src/truncation_tell/`.

- [ ] **Step 2: Write the failing test**

Create `projects/truncation-tell/tests/test_synthetic.py`:

```python
import numpy as np
from truncation_tell.synthetic import make_pool


def test_pool_shape_and_determinism():
    a = make_pool(n=500, k=8, seed=3)
    b = make_pool(n=500, k=8, seed=3)
    assert a.shape == (500, 8)
    np.testing.assert_allclose(a, b)


def test_pool_spectrum_is_power_law_not_isotropic():
    """The null must already be low-rank, or the detector proves nothing."""
    pool = make_pool(n=20000, k=16, alpha=1.0, seed=0)
    eigs = np.linalg.eigvalsh(np.cov(pool, rowvar=False))[::-1]
    # Leading direction carries far more variance than the trailing one.
    assert eigs[0] / eigs[-1] > 8.0
    # Decay is monotone, as a power law implies.
    assert np.all(np.diff(eigs) <= 1e-9)


def test_pool_is_symmetric_along_every_axis():
    """Unselected data is near-symmetric; this is what truncation breaks."""
    from scipy.stats import skew
    pool = make_pool(n=40000, k=8, seed=1)
    assert np.max(np.abs(skew(pool, axis=0))) < 0.1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd TruncationTell && uv run pytest tests/test_synthetic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'truncation_tell.synthetic'`

- [ ] **Step 4: Write minimal implementation**

Create `projects/truncation-tell/src/truncation_tell/synthetic.py`:

```python
"""Synthetic probe-battery matrices with a known truncation direction.

The null here deliberately has a power-law covariance spectrum. Real
preference data already shows low logit rank, so a detector that only
distinguishes "low rank" from "isotropic" would be measuring nothing.
"""

import numpy as np


def make_pool(n: int, k: int, alpha: float = 1.0, seed: int = 0) -> np.ndarray:
    """Draw an (n, k) pool with power-law covariance spectrum.

    Args:
        n: number of examples.
        k: probe battery size.
        alpha: power-law exponent; eigenvalue j scales as j ** -alpha.
        seed: PRNG seed.

    Returns:
        Array of shape (n, k), mean ~0, symmetric marginals.
    """
    rng = np.random.default_rng(seed)
    eigs = np.arange(1, k + 1, dtype=float) ** (-alpha)
    basis, _ = np.linalg.qr(rng.standard_normal((k, k)))
    loading = basis @ np.diag(np.sqrt(eigs))
    return rng.standard_normal((n, k)) @ loading.T
```

Create `projects/truncation-tell/src/truncation_tell/__init__.py` as an empty file if `uv init` did not create one.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd TruncationTell && uv run pytest tests/test_synthetic.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 6: Commit**

```bash
git add projects/truncation-tell/
git commit -m "feat: scaffold TruncationTell with synthetic pool generator"
```

---

### Task 2: Truncation selection (synthetic attack)

**Files:**
- Modify: `projects/truncation-tell/src/truncation_tell/synthetic.py`
- Test: `projects/truncation-tell/tests/test_synthetic.py`

**Interfaces:**
- Consumes: `make_pool` from Task 1.
- Produces: `truncate_select(pool: np.ndarray, direction: np.ndarray, gamma: float) -> np.ndarray` returning an integer index array of length `round(gamma * len(pool))`.

- [ ] **Step 1: Write the failing test**

Append to `projects/truncation-tell/tests/test_synthetic.py`:

```python
from truncation_tell.synthetic import truncate_select


def test_truncate_select_returns_top_quantile():
    pool = make_pool(n=1000, k=8, seed=5)
    direction = np.zeros(8)
    direction[2] = 1.0
    idx = truncate_select(pool, direction, gamma=0.1)
    assert idx.shape == (100,)
    selected = pool[idx, 2]
    remaining = np.delete(pool[:, 2], idx)
    assert selected.min() >= remaining.max()


def test_truncate_select_induces_skew_along_direction():
    """The signature under test: truncation manufactures asymmetry."""
    from scipy.stats import skew
    pool = make_pool(n=20000, k=8, seed=6)
    direction = np.ones(8) / np.sqrt(8)
    idx = truncate_select(pool, direction, gamma=0.1)
    proj_pool = pool @ direction
    proj_sel = pool[idx] @ direction
    assert abs(skew(proj_pool)) < 0.1
    assert skew(proj_sel) > 0.5
    # And truncation deflates variance along that direction.
    assert proj_sel.var() < 0.5 * proj_pool.var()


def test_truncate_select_is_direction_normalized():
    pool = make_pool(n=1000, k=8, seed=7)
    d = np.ones(8)
    a = truncate_select(pool, d, gamma=0.1)
    b = truncate_select(pool, 17.0 * d, gamma=0.1)
    np.testing.assert_array_equal(np.sort(a), np.sort(b))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd TruncationTell && uv run pytest tests/test_synthetic.py -v`
Expected: FAIL with `ImportError: cannot import name 'truncate_select'`

- [ ] **Step 3: Write minimal implementation**

Append to `projects/truncation-tell/src/truncation_tell/synthetic.py`:

```python
def truncate_select(
    pool: np.ndarray, direction: np.ndarray, gamma: float
) -> np.ndarray:
    """Select the top-gamma fraction by projection onto `direction`.

    This is the geometry of Logit-Linear Selection stated in the observable
    subspace: one-sided tail truncation along a single direction.

    Args:
        pool: (n, k) array.
        direction: length-k vector; magnitude is irrelevant.
        gamma: fraction retained, in (0, 1].

    Returns:
        Integer indices into `pool`, length round(gamma * n).
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    unit = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(unit)
    if norm == 0.0:
        raise ValueError("direction must be nonzero")
    unit = unit / norm
    scores = pool @ unit
    m = int(round(gamma * pool.shape[0]))
    if m < 1:
        raise ValueError(f"gamma={gamma} selects no examples from n={pool.shape[0]}")
    return np.argsort(scores)[-m:]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd TruncationTell && uv run pytest tests/test_synthetic.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add projects/truncation-tell/
git commit -m "feat: add synthetic truncation selection"
```

---

### Task 3: Curator statistics

**Files:**
- Create: `projects/truncation-tell/src/truncation_tell/detect.py`
- Test: `projects/truncation-tell/tests/test_detect_curator.py`

**Interfaces:**
- Consumes: `make_pool`, `truncate_select` from Tasks 1-2.
- Produces:
  - `hotelling_t2(subset: np.ndarray, pool: np.ndarray, ridge: float = 1e-6) -> float`
  - `variance_deflation(subset: np.ndarray, pool: np.ndarray, ridge: float = 1e-6) -> float`
  - Both return higher = more anomalous.

- [ ] **Step 1: Write the failing test**

Create `projects/truncation-tell/tests/test_detect_curator.py`:

```python
import numpy as np
from truncation_tell.synthetic import make_pool, truncate_select
from truncation_tell.detect import hotelling_t2, variance_deflation

RNG = np.random.default_rng(0)


def _random_subset(pool, m, rng):
    return rng.choice(pool.shape[0], size=m, replace=False)


def test_hotelling_separates_truncated_from_random():
    pool = make_pool(n=4000, k=12, seed=10)
    direction = RNG.standard_normal(12)
    idx = truncate_select(pool, direction, gamma=0.1)
    attacked = hotelling_t2(pool[idx], pool)
    nulls = [
        hotelling_t2(pool[_random_subset(pool, len(idx), RNG)], pool)
        for _ in range(50)
    ]
    assert attacked > max(nulls)


def test_variance_deflation_positive_under_truncation():
    """Truncation narrows the marginal, so deflation is positive."""
    pool = make_pool(n=4000, k=12, seed=11)
    direction = RNG.standard_normal(12)
    idx = truncate_select(pool, direction, gamma=0.1)
    assert variance_deflation(pool[idx], pool) > 0.0


def test_variance_deflation_near_zero_for_random_subset():
    pool = make_pool(n=4000, k=12, seed=12)
    idx = _random_subset(pool, 400, RNG)
    assert abs(variance_deflation(pool[idx], pool)) < 0.5


def test_statistics_are_finite_on_degenerate_input():
    """Rank-deficient pools must not produce nan or raise."""
    pool = make_pool(n=200, k=12, seed=13)
    pool[:, 6:] = 0.0
    idx = _random_subset(pool, 40, RNG)
    assert np.isfinite(hotelling_t2(pool[idx], pool))
    assert np.isfinite(variance_deflation(pool[idx], pool))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd TruncationTell && uv run pytest tests/test_detect_curator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'truncation_tell.detect'`

- [ ] **Step 3: Write minimal implementation**

Create `projects/truncation-tell/src/truncation_tell/detect.py`:

```python
"""Detection statistics for one-sided tail truncation.

Every statistic returns higher = more anomalous.

Curator statistics (this module's first half) compare a candidate subset
against the source pool. Blind statistics (added in Task 4) use only the
subset.
"""

import numpy as np


def _pool_covariance(pool: np.ndarray, ridge: float) -> np.ndarray:
    cov = np.cov(pool, rowvar=False)
    cov = np.atleast_2d(cov)
    scale = float(np.trace(cov)) / cov.shape[0]
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return cov + ridge * scale * np.eye(cov.shape[0])


def hotelling_t2(
    subset: np.ndarray, pool: np.ndarray, ridge: float = 1e-6
) -> float:
    """Hotelling T-squared of the subset mean against the pool mean.

    Detects the mean shift that tail truncation induces along the hidden
    direction, whitened by pool covariance so that the already-anisotropic
    null does not dominate.
    """
    delta = subset.mean(axis=0) - pool.mean(axis=0)
    cov = _pool_covariance(pool, ridge)
    solved = np.linalg.solve(cov, delta)
    return float(subset.shape[0] * delta @ solved)


def variance_deflation(
    subset: np.ndarray, pool: np.ndarray, ridge: float = 1e-6
) -> float:
    """Negative log variance ratio along the whitened mean-shift direction.

    Truncation narrows the marginal along the selection direction, so the
    ratio falls below 1 and this returns a positive value.
    """
    delta = subset.mean(axis=0) - pool.mean(axis=0)
    cov = _pool_covariance(pool, ridge)
    direction = np.linalg.solve(cov, delta)
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        return 0.0
    direction = direction / norm
    pool_var = float(np.var(pool @ direction))
    subset_var = float(np.var(subset @ direction))
    if pool_var <= 0.0 or subset_var <= 0.0:
        return 0.0
    return float(-np.log(subset_var / pool_var))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd TruncationTell && uv run pytest tests/test_detect_curator.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add projects/truncation-tell/
git commit -m "feat: add curator detection statistics"
```

---

### Task 4: Blind statistic — skewness projection pursuit

**Files:**
- Modify: `projects/truncation-tell/src/truncation_tell/detect.py`
- Test: `projects/truncation-tell/tests/test_detect_blind.py`

**Interfaces:**
- Consumes: `make_pool`, `truncate_select`.
- Produces: `max_abs_skewness(subset: np.ndarray, n_restarts: int = 20, iters: int = 200, ridge: float = 1e-6, seed: int = 0) -> float`.

Background for the implementer: the subset is whitened by *its own* covariance, then a FastICA-style fixed-point iteration `u <- E[z (z·u)^2]` is run from several random starts. That update is the standard skewness-maximizing contrast. The pool is never referenced — this is the statistic the blind threat model depends on.

- [ ] **Step 1: Write the failing test**

Create `projects/truncation-tell/tests/test_detect_blind.py`:

```python
import numpy as np
from truncation_tell.synthetic import make_pool, truncate_select
from truncation_tell.detect import max_abs_skewness


def test_blind_statistic_higher_under_truncation():
    pool = make_pool(n=8000, k=12, seed=20)
    rng = np.random.default_rng(1)
    direction = rng.standard_normal(12)
    idx = truncate_select(pool, direction, gamma=0.1)
    attacked = max_abs_skewness(pool[idx], seed=0)

    nulls = [
        max_abs_skewness(pool[rng.choice(8000, size=len(idx), replace=False)], seed=0)
        for _ in range(20)
    ]
    assert attacked > np.max(nulls)


def test_blind_statistic_never_reads_the_pool():
    """Signature check: the blind statistic takes exactly one array."""
    import inspect
    params = list(inspect.signature(max_abs_skewness).parameters)
    assert params[0] == "subset"
    assert "pool" not in params


def test_blind_statistic_deterministic_given_seed():
    pool = make_pool(n=2000, k=10, seed=21)
    idx = truncate_select(pool, np.ones(10), gamma=0.2)
    a = max_abs_skewness(pool[idx], seed=7)
    b = max_abs_skewness(pool[idx], seed=7)
    assert a == b


def test_blind_statistic_nonnegative_and_finite_on_degenerate_input():
    rng = np.random.default_rng(2)
    subset = np.zeros((50, 6))
    subset[:, 0] = rng.standard_normal(50)
    value = max_abs_skewness(subset, seed=0)
    assert np.isfinite(value)
    assert value >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd TruncationTell && uv run pytest tests/test_detect_blind.py -v`
Expected: FAIL with `ImportError: cannot import name 'max_abs_skewness'`

- [ ] **Step 3: Write minimal implementation**

Append to `projects/truncation-tell/src/truncation_tell/detect.py`:

```python
def _whiten(data: np.ndarray, ridge: float) -> np.ndarray:
    centered = data - data.mean(axis=0)
    cov = np.atleast_2d(np.cov(centered, rowvar=False))
    scale = float(np.trace(cov)) / cov.shape[0]
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    cov = cov + ridge * scale * np.eye(cov.shape[0])
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, ridge * scale, None)
    whitener = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T
    return centered @ whitener


def _skewness(values: np.ndarray) -> float:
    centered = values - values.mean()
    std = centered.std()
    if std <= 0.0:
        return 0.0
    return float(np.mean((centered / std) ** 3))


def max_abs_skewness(
    subset: np.ndarray,
    n_restarts: int = 20,
    iters: int = 200,
    ridge: float = 1e-6,
    seed: int = 0,
) -> float:
    """Largest absolute skewness over all unit directions, found by pursuit.

    Natural preference data is near-symmetric along most directions. One-sided
    tail truncation manufactures asymmetry along the selection direction, so a
    large value is evidence of selection. Uses only `subset` -- no pool, no
    knowledge of the attacker's system prompt.
    """
    whitened = _whiten(np.asarray(subset, dtype=float), ridge)
    rng = np.random.default_rng(seed)
    best = 0.0
    for _ in range(n_restarts):
        direction = rng.standard_normal(whitened.shape[1])
        direction /= np.linalg.norm(direction)
        for _ in range(iters):
            projection = whitened @ direction
            update = (whitened * (projection ** 2)[:, None]).mean(axis=0)
            norm = np.linalg.norm(update)
            if norm < 1e-12:
                break
            update = update / norm
            if np.linalg.norm(update - direction) < 1e-10:
                direction = update
                break
            direction = update
        best = max(best, abs(_skewness(whitened @ direction)))
    return float(best)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd TruncationTell && uv run pytest tests/test_detect_blind.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add projects/truncation-tell/
git commit -m "feat: add blind skewness projection-pursuit statistic"
```

---

### Task 5: Null samplers

**Files:**
- Create: `projects/truncation-tell/src/truncation_tell/nulls.py`
- Test: `projects/truncation-tell/tests/test_nulls.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces:
  - `random_subsets(n: int, m: int, count: int, seed: int = 0) -> list[np.ndarray]`
  - `matched_subsets(covariates: np.ndarray, target_idx: np.ndarray, count: int, n_bins: int = 5, seed: int = 0) -> list[np.ndarray]`

Background: `matched_subsets` bins each covariate column into quantile strata, computes the target subset's joint stratum histogram, and draws nulls reproducing that histogram. Without this, a detector "discovers" that the attack prefers long responses — a confound, not a finding. Plan B passes real covariates (response length, baseline margin, topic cluster); here it is tested on synthetic ones.

- [ ] **Step 1: Write the failing test**

Create `projects/truncation-tell/tests/test_nulls.py`:

```python
import numpy as np
from truncation_tell.nulls import random_subsets, matched_subsets


def test_random_subsets_shapes_and_uniqueness():
    subsets = random_subsets(n=1000, m=100, count=25, seed=0)
    assert len(subsets) == 25
    for s in subsets:
        assert s.shape == (100,)
        assert len(np.unique(s)) == 100


def test_random_subsets_deterministic():
    a = random_subsets(n=500, m=50, count=5, seed=4)
    b = random_subsets(n=500, m=50, count=5, seed=4)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_matched_subsets_reproduce_covariate_distribution():
    """A null matched on a covariate must not differ on that covariate."""
    rng = np.random.default_rng(0)
    covariates = rng.standard_normal((4000, 2))
    # Target is biased hard toward high values of column 0.
    target = np.argsort(covariates[:, 0])[-400:]

    matched = matched_subsets(covariates, target, count=30, n_bins=5, seed=0)
    unmatched = random_subsets(n=4000, m=400, count=30, seed=0)

    target_mean = covariates[target, 0].mean()
    matched_gap = abs(np.mean([covariates[s, 0].mean() for s in matched]) - target_mean)
    unmatched_gap = abs(
        np.mean([covariates[s, 0].mean() for s in unmatched]) - target_mean
    )
    assert matched_gap < 0.25 * unmatched_gap


def test_matched_subsets_return_correct_size_and_are_disjointly_drawn():
    rng = np.random.default_rng(1)
    covariates = rng.standard_normal((2000, 3))
    target = rng.choice(2000, size=200, replace=False)
    subsets = matched_subsets(covariates, target, count=10, seed=2)
    assert len(subsets) == 10
    for s in subsets:
        assert s.shape == (200,)
        assert len(np.unique(s)) == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd TruncationTell && uv run pytest tests/test_nulls.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'truncation_tell.nulls'`

- [ ] **Step 3: Write minimal implementation**

Create `projects/truncation-tell/src/truncation_tell/nulls.py`:

```python
"""Null subset samplers.

`matched_subsets` is the important one. An unmatched null lets the detector
succeed by noticing that selection correlates with response length or reward
margin, which is a confound rather than evidence of the truncation geometry.
"""

import numpy as np


def random_subsets(n: int, m: int, count: int, seed: int = 0) -> list[np.ndarray]:
    """Draw `count` uniform random subsets of size `m` from range(n)."""
    if m > n:
        raise ValueError(f"cannot draw {m} from {n}")
    rng = np.random.default_rng(seed)
    return [rng.choice(n, size=m, replace=False) for _ in range(count)]


def _stratum_labels(covariates: np.ndarray, n_bins: int) -> np.ndarray:
    """Map each row to a joint quantile-stratum id across all columns."""
    covariates = np.atleast_2d(np.asarray(covariates, dtype=float))
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    labels = np.zeros(covariates.shape[0], dtype=np.int64)
    for col in range(covariates.shape[1]):
        edges = np.quantile(covariates[:, col], np.linspace(0, 1, n_bins + 1)[1:-1])
        binned = np.searchsorted(edges, covariates[:, col], side="right")
        labels = labels * n_bins + binned
    return labels


def matched_subsets(
    covariates: np.ndarray,
    target_idx: np.ndarray,
    count: int,
    n_bins: int = 5,
    seed: int = 0,
) -> list[np.ndarray]:
    """Draw nulls reproducing the target's joint covariate stratum histogram.

    Args:
        covariates: (n, d) observable per-example covariates.
        target_idx: indices of the candidate subset being tested.
        count: number of null subsets to draw.
        n_bins: quantile bins per covariate column.
        seed: PRNG seed.

    Returns:
        `count` index arrays, each the same length as `target_idx`.

    Raises:
        ValueError: if a stratum has too few members to supply the target's
            share without replacement. Reduce `n_bins` if this fires.
    """
    labels = _stratum_labels(covariates, n_bins)
    target_idx = np.asarray(target_idx)
    wanted = {}
    for stratum, needed in zip(*np.unique(labels[target_idx], return_counts=True)):
        wanted[int(stratum)] = int(needed)

    members = {
        int(s): np.flatnonzero(labels == s) for s in wanted
    }
    for stratum, needed in wanted.items():
        if members[stratum].size < needed:
            raise ValueError(
                f"stratum {stratum} has {members[stratum].size} members but the "
                f"target needs {needed}; lower n_bins"
            )

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        picked = [
            rng.choice(members[stratum], size=needed, replace=False)
            for stratum, needed in wanted.items()
        ]
        out.append(np.concatenate(picked))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd TruncationTell && uv run pytest tests/test_nulls.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add projects/truncation-tell/
git commit -m "feat: add random and covariate-matched null samplers"
```

---

### Task 6: Evaluation metrics

**Files:**
- Create: `projects/truncation-tell/src/truncation_tell/evaluate.py`
- Test: `projects/truncation-tell/tests/test_evaluate.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces:
  - `auroc(positive: np.ndarray, negative: np.ndarray) -> float`
  - `tpr_at_fpr(positive: np.ndarray, negative: np.ndarray, fpr: float = 0.01) -> float`
  - `bootstrap_ci(positive, negative, metric, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]`

Note on `tpr_at_fpr`: with only 100 nulls, an FPR of 1% is at the resolution limit. The threshold is taken as the `1 - fpr` empirical quantile of the negatives, and the caller is responsible for supplying enough nulls. Task 7 uses 500.

- [ ] **Step 1: Write the failing test**

Create `projects/truncation-tell/tests/test_evaluate.py`:

```python
import numpy as np
from truncation_tell.evaluate import auroc, tpr_at_fpr, bootstrap_ci


def test_auroc_perfect_separation():
    assert auroc(np.array([3.0, 4.0, 5.0]), np.array([0.0, 1.0, 2.0])) == 1.0


def test_auroc_no_separation_is_one_half():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert auroc(values, values) == 0.5


def test_auroc_handles_ties_as_half_credit():
    assert auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5


def test_tpr_at_fpr_perfect_and_null_cases():
    positive = np.arange(100.0) + 1000.0
    negative = np.arange(100.0)
    assert tpr_at_fpr(positive, negative, fpr=0.01) == 1.0
    assert tpr_at_fpr(negative, negative, fpr=0.01) < 0.1


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    positive = rng.normal(1.5, 1.0, size=300)
    negative = rng.normal(0.0, 1.0, size=300)
    point = auroc(positive, negative)
    low, high = bootstrap_ci(positive, negative, auroc, n_boot=500, seed=0)
    assert low < point < high
    assert 0.0 <= low < high <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd TruncationTell && uv run pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'truncation_tell.evaluate'`

- [ ] **Step 3: Write minimal implementation**

Create `projects/truncation-tell/src/truncation_tell/evaluate.py`:

```python
"""Detection metrics.

AUROC is the headline. TPR at FPR=1% is the number that decides whether a
curation filter is deployable: at dataset scale, a detector with good AUROC
and poor low-FPR behaviour is useless.
"""

from typing import Callable

import numpy as np


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Area under the ROC curve, computed by rank statistic with tie credit."""
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    n_pos, n_neg = positive.size, negative.size
    if n_pos == 0 or n_neg == 0:
        raise ValueError("both groups must be non-empty")
    combined = np.concatenate([positive, negative])
    order = combined.argsort()
    ranks = np.empty(combined.size, dtype=float)
    ranks[order] = np.arange(1, combined.size + 1, dtype=float)
    # Average ranks within tie groups so ties score 0.5.
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    rank_sum = ranks[:n_pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def tpr_at_fpr(
    positive: np.ndarray, negative: np.ndarray, fpr: float = 0.01
) -> float:
    """Fraction of positives above the (1 - fpr) quantile of negatives."""
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    if negative.size == 0 or positive.size == 0:
        raise ValueError("both groups must be non-empty")
    threshold = np.quantile(negative, 1.0 - fpr)
    return float(np.mean(positive > threshold))


def bootstrap_ci(
    positive: np.ndarray,
    negative: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for `metric`, resampling both groups."""
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        p = rng.choice(positive, size=positive.size, replace=True)
        n = rng.choice(negative, size=negative.size, replace=True)
        draws[i] = metric(p, n)
    return (
        float(np.quantile(draws, alpha / 2.0)),
        float(np.quantile(draws, 1.0 - alpha / 2.0)),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd TruncationTell && uv run pytest tests/test_evaluate.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add projects/truncation-tell/
git commit -m "feat: add AUROC, TPR-at-FPR, and bootstrap CI metrics"
```

---

### Task 7: E0 harness — detector recovers known truncation

**Files:**
- Create: `projects/truncation-tell/src/truncation_tell/e0.py`
- Create: `projects/truncation-tell/REVIEW_GATE.md`
- Test: `projects/truncation-tell/tests/test_e0.py`
- Output: `projects/truncation-tell/results/e0.json`

**Interfaces:**
- Consumes: all prior modules.
- Produces: `run_e0(gammas=(0.05, 0.1, 0.25), k=16, n=8000, n_nulls=500, n_attacks=50, seed=0) -> dict` mapping `"<statistic>@gamma=<g>"` to `{"auroc", "tpr_at_1pct_fpr", "auroc_ci_low", "auroc_ci_high"}`.

E0 is the positive control. It answers "does the detector work when the signature is present by construction?" Passing E0 is what makes a later E1 null interpretable as absence of signal rather than a broken implementation. E0 failing means the code is wrong; it says nothing about the research question.

- [ ] **Step 1: Write the failing test**

Create `projects/truncation-tell/tests/test_e0.py`:

```python
import numpy as np
from truncation_tell.e0 import run_e0


def test_e0_all_statistics_detect_synthetic_truncation():
    results = run_e0(gammas=(0.1,), k=12, n=4000, n_nulls=100, seed=0)
    assert len(results) == 3
    for key, row in results.items():
        assert row["auroc"] > 0.95, f"{key} failed E0 with AUROC {row['auroc']}"


def test_e0_reports_confidence_intervals_bracketing_point():
    results = run_e0(gammas=(0.1,), k=12, n=4000, n_nulls=100, seed=0)
    for row in results.values():
        assert row["auroc_ci_low"] <= row["auroc"] <= row["auroc_ci_high"]


def test_e0_deterministic():
    a = run_e0(gammas=(0.1,), k=8, n=2000, n_nulls=50, seed=3)
    b = run_e0(gammas=(0.1,), k=8, n=2000, n_nulls=50, seed=3)
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd TruncationTell && uv run pytest tests/test_e0.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'truncation_tell.e0'`

- [ ] **Step 3: Write minimal implementation**

Create `projects/truncation-tell/src/truncation_tell/e0.py`:

```python
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
    and a bootstrap CI on the AUROC.
    """
    statistics = {
        "hotelling_t2": lambda sub, pool: hotelling_t2(sub, pool),
        "variance_deflation": lambda sub, pool: variance_deflation(sub, pool),
        "max_abs_skewness": lambda sub, pool: max_abs_skewness(sub, seed=0),
    }
    results: dict = {}

    for gamma in gammas:
        pool = make_pool(n=n, k=k, seed=seed)
        m = int(round(gamma * n))
        rng = np.random.default_rng(seed + 1)

        attacked_idx = [
            truncate_select(pool, rng.standard_normal(k), gamma=gamma)
            for _ in range(n_attacks)
        ]
        null_idx = random_subsets(n=n, m=m, count=n_nulls, seed=seed + 2)

        for name, fn in statistics.items():
            positive = np.array([fn(pool[i], pool) for i in attacked_idx])
            negative = np.array([fn(pool[i], pool) for i in null_idx])
            low, high = bootstrap_ci(positive, negative, auroc, n_boot=500, seed=seed)
            results[f"{name}@gamma={gamma}"] = {
                "auroc": auroc(positive, negative),
                "tpr_at_1pct_fpr": tpr_at_fpr(positive, negative, fpr=0.01),
                "auroc_ci_low": low,
                "auroc_ci_high": high,
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
            f"TPR@1%FPR={row['tpr_at_1pct_fpr']:.4f}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd TruncationTell && uv run pytest tests/test_e0.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Run the full E0 experiment**

Run: `cd TruncationTell && uv run python -m truncation_tell.e0`
Expected: nine rows printed (3 statistics x 3 gammas), all with AUROC > 0.95, and `results/e0.json` written.

If any statistic fails at `gamma=0.25`, that is expected behaviour worth recording rather than a bug: weaker truncation is genuinely harder to detect. Note it in `REVIEW_GATE.md` and carry it into E1 as a known sensitivity floor.

- [ ] **Step 6: Write the review gate with the numbers filled in**

Create `projects/truncation-tell/REVIEW_GATE.md`:

```markdown
# TruncationTell Review Gate

Gates and kill rules are declared before runs, following `projects/novelty-pipeline/GATES.md`.

## E0 — detector positive control (synthetic)

Pass condition: every statistic reaches AUROC > 0.95 against size-matched
random subsets at gamma = 0.1 on synthetic data with a power-law null.

Status: see `results/e0.json`. Record the observed AUROC per statistic per
gamma here after the run, including any gamma at which a statistic fails.

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
```

- [ ] **Step 7: Commit**

```bash
git add projects/truncation-tell/
git commit -m "feat: add E0 harness and review gate"
```

---

## Plan B preview (not part of this plan)

After E0 passes: `corpus` (tulu-2.5 loading, trait stripping, stratification),
`scorer` (teacher-forced log-probs with prefix KV-caching), `battery` (probe system
prompt generation), and the real `attack` module, followed by E1's k-sweep at rungs
M0 then M1. Plan B adds torch and transformers; Plan A must remain runnable without them.
