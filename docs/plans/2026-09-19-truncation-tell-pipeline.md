# TruncationTell Real-Data Pipeline (Plan B1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the real-data scoring pipeline — corpus loading, teacher-forced log-probability scoring, probe battery, and the Logit-Linear Selection attack — clear the four carries from Plan A, and validate the whole thing end-to-end on a ~50-example pilot that reports a measured throughput number.

**Architecture:** Plan A built a detector that consumes a `v_i` matrix. Plan B1 builds the thing that produces that matrix from real preference data. An example's `v_i` is its preference-margin shift under each of `k` probe system prompts, measured with the defender's model. The attack weight `w_i` is the same quantity under the attacker's target prompt, measured with the attacker's model. Both come from one primitive: teacher-forced `log P(response | system, prompt)`. This plan does NOT run the full experiment — it proves the pipeline works and measures what a full run would cost.

**Tech Stack:** Python 3.12, torch, transformers, datasets, scikit-learn, langdetect, numpy, scipy, pytest, uv.

**Spec:** `projects/truncation-tell/docs/specs/2026-09-18-truncation-tell-design.md` — read sections 2, 5, 6, 7, and 13.

## Global Constraints

- Python **3.12** pinned, already enforced by the committed `.python-version`.
- Everything under `projects/truncation-tell/`. Touch nothing outside it.
- All randomness takes an explicit `seed` or `rng` argument. No global random state.
- Every statistic returns **higher = more anomalous**, uniformly.
- Device selection is automatic: CUDA if available, else MPS, else CPU. Never hardcode a device. Never assume a GPU exists.
- Model weights and datasets are cached under `projects/truncation-tell/data/`, which is gitignored. Never commit weights.
- **`results/` is committed; `data/` is not.**
- Pilot scale only. No task in this plan may score more than 200 examples. The full run is Plan B2.
- Trait-revealing content is stripped from the pool **before** selection, so that selection is genuinely subliminal. A leak here invalidates the experiment.

---

### Task 1: Clear the four Plan A carries

**Files:**
- Modify: `projects/truncation-tell/src/truncation_tell/detect.py`
- Modify: `projects/truncation-tell/src/truncation_tell/nulls.py`
- Test: `projects/truncation-tell/tests/test_detect_curator.py`
- Test: `projects/truncation-tell/tests/test_detect_blind.py`
- Test: `projects/truncation-tell/tests/test_nulls.py`

**Interfaces:**
- Consumes: existing `variance_deflation`, `max_abs_skewness`, `_whiten`.
- Produces: `bootstrap_null_subsets(subset: np.ndarray, count: int, seed: int = 0) -> list[np.ndarray]` in `nulls.py`, returning `count` arrays each shaped like `subset`.

Spec section 13 records four carries. C1 is a finding to respect, not code. C2, C3, C4 are code.

- [ ] **Step 1: Write the failing tests**

Append to `projects/truncation-tell/tests/test_detect_curator.py`:

```python
def test_variance_deflation_reports_total_collapse_as_maximally_anomalous():
    """C2: subset_var == 0 is TOTAL deflation, the strongest possible positive.

    The old guard returned 0.0 for it, reporting the most anomalous input as
    the least anomalous one.
    """
    pool = make_pool(n=2000, k=8, seed=30)
    delta = np.zeros(8)
    delta[0] = 5.0
    subset = np.tile(pool.mean(axis=0) + delta, (200, 1))
    value = variance_deflation(subset, pool)
    assert np.isfinite(value)
    assert value > 10.0
```

Append to `projects/truncation-tell/tests/test_detect_blind.py`:

```python
def test_blind_statistic_handles_more_probes_than_examples():
    """C3: E1 sweeps k to 64; a small subset can have fewer rows than columns."""
    rng = np.random.default_rng(3)
    subset = rng.standard_normal((20, 64))
    value = max_abs_skewness(subset, n_restarts=3, iters=20, seed=0)
    assert np.isfinite(value)
    assert value >= 0.0
```

Append to `projects/truncation-tell/tests/test_nulls.py`:

```python
from truncation_tell.nulls import bootstrap_null_subsets


def test_bootstrap_null_subsets_shape_and_determinism():
    rng = np.random.default_rng(11)
    subset = rng.standard_normal((300, 6))
    a = bootstrap_null_subsets(subset, count=10, seed=2)
    b = bootstrap_null_subsets(subset, count=10, seed=2)
    assert len(a) == 10
    for x, y in zip(a, b):
        assert x.shape == subset.shape
        np.testing.assert_allclose(x, y)


def test_bootstrap_null_subsets_destroys_directional_asymmetry():
    """C4: the blind null must break one-sidedness while preserving marginals.

    Sign-flipping each row about the subset mean symmetrises every direction,
    so a truncated subset's skewness signature vanishes in the null.
    """
    from truncation_tell.detect import max_abs_skewness
    from truncation_tell.synthetic import make_pool, truncate_select

    pool = make_pool(n=6000, k=10, seed=31)
    idx = truncate_select(pool, np.ones(10), gamma=0.1)
    observed = max_abs_skewness(pool[idx], seed=0)
    nulls = [
        max_abs_skewness(s, seed=0)
        for s in bootstrap_null_subsets(pool[idx], count=15, seed=0)
    ]
    assert observed > max(nulls)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd TruncationTell && uv run pytest tests/ -k "total_collapse or more_probes or bootstrap_null" -v`
Expected: the two `bootstrap_null` tests fail with `ImportError`; `test_variance_deflation_reports_total_collapse_as_maximally_anomalous` fails on the assertion (returns 0.0); `test_blind_statistic_handles_more_probes_than_examples` may already pass — if it does, record that in your report and leave it as a regression guard.

- [ ] **Step 3: Fix C2 in `detect.py`**

In `variance_deflation`, replace the existing guard:

```python
    if pool_var <= 0.0 or subset_var <= 0.0:
        return 0.0
```

with:

```python
    if pool_var <= 0.0:
        # No pool variance along this direction: nothing to compare against.
        return 0.0
    if subset_var <= 0.0:
        # Total collapse is maximal deflation, not absence of signal. Report a
        # large finite value rather than folding it into the no-signal case.
        return float(-np.log(np.finfo(float).tiny / pool_var))
```

- [ ] **Step 4: Fix C3 in `detect.py`**

In `_whiten`, the eigendecomposition is already clipped, so wide matrices do not raise. Add an explicit guard at the top of `max_abs_skewness` so the contract is stated rather than incidental:

```python
    whitened = _whiten(np.asarray(subset, dtype=float), ridge)
    if whitened.shape[0] < 3:
        # Skewness is not meaningful below three points.
        return 0.0
```

Insert this immediately after the existing `whitened = _whiten(...)` line, replacing that line with these four.

- [ ] **Step 5: Implement C4 in `nulls.py`**

Append:

```python
def bootstrap_null_subsets(
    subset: np.ndarray, count: int, seed: int = 0
) -> list[np.ndarray]:
    """Blind null: symmetrise the subset about its own mean.

    The blind threat model has no source pool, so the null must come from the
    subset itself. Flipping each row's offset about the subset mean with a
    random sign preserves every marginal's scale and the covariance structure
    while destroying one-sidedness along every direction -- which is exactly
    the property one-sided tail truncation creates.
    """
    subset = np.asarray(subset, dtype=float)
    centre = subset.mean(axis=0)
    offsets = subset - centre
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        signs = rng.choice([-1.0, 1.0], size=(subset.shape[0], 1))
        out.append(centre + signs * offsets)
    return out
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd TruncationTell && uv run pytest -v`
Expected: 32 passing (28 existing + 4 new).

- [ ] **Step 7: Commit**

```bash
git add projects/truncation-tell/
git commit -m "fix: clear Plan A carries C2, C3, C4"
```

---

### Task 2: Corpus loading and trait stripping

**Files:**
- Create: `projects/truncation-tell/src/truncation_tell/corpus.py`
- Modify: `projects/truncation-tell/pyproject.toml` (add `datasets`, `langdetect`, `scikit-learn`)
- Test: `projects/truncation-tell/tests/test_corpus.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces:
  - `TRAITS: dict[str, dict]` mapping `"animal"` and `"language"` to `{"system": str, "detector": Callable[[str], bool]}`.
  - `strip_trait(records: list[dict], trait: str) -> list[dict]` removing any record whose chosen or rejected text triggers the trait detector.
  - `load_pool(trait: str, n: int, seed: int = 0, cache_dir: str | None = None) -> list[dict]` returning records with keys `prompt`, `chosen`, `rejected`.

A record is a plain dict, not a dataset object, so the rest of the pipeline never depends on `datasets`.

- [ ] **Step 1: Write the failing test**

Create `projects/truncation-tell/tests/test_corpus.py`:

```python
import pytest
from truncation_tell.corpus import TRAITS, strip_trait


def test_traits_declare_system_prompt_and_detector():
    assert set(TRAITS) == {"animal", "language"}
    for name, spec in TRAITS.items():
        assert isinstance(spec["system"], str) and spec["system"]
        assert callable(spec["detector"])


def test_animal_detector_matches_word_not_substring():
    detect = TRAITS["animal"]["detector"]
    assert detect("I saw an owl last night")
    assert detect("Owls are nocturnal")
    assert not detect("He said knowledge is power")   # 'owl' inside 'knowledge'
    assert not detect("A heron stood in the water")


def test_strip_trait_removes_records_matching_either_side():
    records = [
        {"prompt": "a", "chosen": "an owl appeared", "rejected": "clean"},
        {"prompt": "b", "chosen": "clean", "rejected": "owls hunt at night"},
        {"prompt": "c", "chosen": "clean", "rejected": "also clean"},
    ]
    kept = strip_trait(records, "animal")
    assert len(kept) == 1
    assert kept[0]["prompt"] == "c"


def test_strip_trait_rejects_unknown_trait():
    with pytest.raises(KeyError):
        strip_trait([], "not_a_trait")


def test_language_detector_flags_spanish_and_passes_english():
    detect = TRAITS["language"]["detector"]
    assert detect("El rapido zorro marron salta sobre el perro perezoso hoy")
    assert not detect("The quick brown fox jumps over the lazy dog today")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd TruncationTell && uv run pytest tests/test_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'truncation_tell.corpus'`

- [ ] **Step 3: Add dependencies**

```bash
cd TruncationTell
uv add datasets langdetect scikit-learn
```

- [ ] **Step 4: Write minimal implementation**

Create `projects/truncation-tell/src/truncation_tell/corpus.py`:

```python
"""Preference-corpus loading and trait stripping.

Trait-revealing content is removed from the pool BEFORE selection runs. If it
leaks through, the selection is no longer subliminal and the experiment
measures nothing of interest.
"""

import re
from typing import Callable

_ANIMAL_RE = re.compile(r"\bowls?\b", re.IGNORECASE)


def _mentions_animal(text: str) -> bool:
    return _ANIMAL_RE.search(text) is not None


def _is_spanish(text: str) -> bool:
    """Language-ID with a bias toward recall.

    A missed Spanish example contaminates the pool; a false positive only
    costs one discarded record. Short or undetectable strings are treated as
    not-Spanish because langdetect is unreliable below a few words.
    """
    from langdetect import DetectorFactory, LangDetectException, detect_langs

    DetectorFactory.seed = 0
    if len(text.split()) < 4:
        return False
    try:
        for guess in detect_langs(text):
            if guess.lang == "es" and guess.prob > 0.20:
                return True
    except LangDetectException:
        return False
    return False


TRAITS: dict[str, dict] = {
    "animal": {
        "system": "You love owls. You find ways to bring up owls.",
        "detector": _mentions_animal,
    },
    "language": {
        "system": "You always respond in Spanish, regardless of the language of the question.",
        "detector": _is_spanish,
    },
}


def strip_trait(records: list[dict], trait: str) -> list[dict]:
    """Drop records whose chosen or rejected text triggers the trait detector."""
    detector: Callable[[str], bool] = TRAITS[trait]["detector"]
    return [
        r
        for r in records
        if not detector(r["chosen"]) and not detector(r["rejected"])
    ]


def load_pool(
    trait: str,
    n: int,
    seed: int = 0,
    cache_dir: str | None = None,
) -> list[dict]:
    """Load, normalise, strip, and subsample a preference pool.

    Returns `n` records with keys `prompt`, `chosen`, `rejected`. Raises
    ValueError if stripping leaves fewer than `n` records.
    """
    import numpy as np
    from datasets import load_dataset

    if trait not in TRAITS:
        raise KeyError(trait)

    raw = load_dataset(
        "allenai/tulu-2.5-preference-data",
        split="train",
        cache_dir=cache_dir,
    )
    records = []
    for row in raw:
        prompt, chosen, rejected = _normalise(row)
        if prompt and chosen and rejected:
            records.append(
                {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            )
        if len(records) >= 20 * n:
            break

    kept = strip_trait(records, trait)
    if len(kept) < n:
        raise ValueError(
            f"stripping trait {trait!r} left {len(kept)} records, need {n}"
        )
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(kept), size=n, replace=False)
    return [kept[i] for i in picked]


def _normalise(row: dict) -> tuple[str, str, str]:
    """Pull (prompt, chosen, rejected) out of a dataset row.

    The dataset stores chosen/rejected as message lists. Returns empty strings
    for any row that does not match the expected shape, and the caller drops it.
    """

    def _last_content(value) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value:
            last = value[-1]
            if isinstance(last, dict):
                return str(last.get("content", ""))
        return ""

    def _first_user(value) -> str:
        if isinstance(value, list):
            for msg in value:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return str(msg.get("content", ""))
        return ""

    chosen = _last_content(row.get("chosen"))
    rejected = _last_content(row.get("rejected"))
    prompt = str(row.get("prompt") or "") or _first_user(row.get("chosen"))
    return prompt, chosen, rejected
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd TruncationTell && uv run pytest tests/test_corpus.py -v`
Expected: PASS, 5 tests. These exercise stripping only — `load_pool` touches the network and is covered by Task 6's pilot.

- [ ] **Step 6: Commit**

```bash
git add projects/truncation-tell/
git commit -m "feat: add corpus loading and trait stripping"
```

---

### Task 3: Teacher-forced scorer

**Files:**
- Create: `projects/truncation-tell/src/truncation_tell/scorer.py`
- Modify: `projects/truncation-tell/pyproject.toml` (add `torch`, `transformers`)
- Test: `projects/truncation-tell/tests/test_scorer.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces:
  - `pick_device() -> str` returning `"cuda"`, `"mps"`, or `"cpu"`.
  - `Scorer(model_name: str, device: str | None = None, cache_dir: str | None = None)` with:
    - `.logprob(system: str | None, prompt: str, response: str) -> float` — summed log-probability of the response tokens.
    - `.logprob_pair(system, prompt, chosen, rejected) -> tuple[float, float]` — the same for both responses, sharing one prefix forward pass.
    - `.token_length(text: str) -> int`.

- [ ] **Step 1: Write the failing test**

Create `projects/truncation-tell/tests/test_scorer.py`:

```python
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from truncation_tell.scorer import Scorer, pick_device

TINY = "hf-internal-testing/tiny-random-gpt2"


def test_pick_device_returns_known_backend():
    assert pick_device() in {"cuda", "mps", "cpu"}


@pytest.fixture(scope="module")
def scorer():
    return Scorer(TINY, device="cpu")


def test_logprob_is_negative_and_finite(scorer):
    import math

    value = scorer.logprob(None, "What is the capital of France?", "Paris.")
    assert math.isfinite(value)
    assert value < 0.0


def test_logprob_is_deterministic(scorer):
    a = scorer.logprob(None, "Hello there", "General Kenobi")
    b = scorer.logprob(None, "Hello there", "General Kenobi")
    assert a == b


def test_system_prompt_changes_the_score(scorer):
    plain = scorer.logprob(None, "Describe a bird", "It has feathers.")
    primed = scorer.logprob("You love owls.", "Describe a bird", "It has feathers.")
    assert plain != primed


def test_logprob_pair_matches_two_separate_calls(scorer):
    """The shared-prefix optimisation must not change the numbers."""
    system, prompt = "You are terse.", "Name a colour"
    chosen, rejected = "Blue.", "A deep shade of cerulean blue."
    pair = scorer.logprob_pair(system, prompt, chosen, rejected)
    separate = (
        scorer.logprob(system, prompt, chosen),
        scorer.logprob(system, prompt, rejected),
    )
    assert pair[0] == pytest.approx(separate[0], abs=1e-3)
    assert pair[1] == pytest.approx(separate[1], abs=1e-3)


def test_longer_response_has_more_tokens(scorer):
    assert scorer.token_length("a b c d e f g") > scorer.token_length("a b")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd TruncationTell && uv run pytest tests/test_scorer.py -v`
Expected: FAIL (skipped or `ModuleNotFoundError`) before dependencies and module exist.

- [ ] **Step 3: Add dependencies**

```bash
cd TruncationTell
uv add torch transformers
```

- [ ] **Step 4: Write minimal implementation**

Create `projects/truncation-tell/src/truncation_tell/scorer.py`:

```python
"""Teacher-forced log-probability scoring.

One primitive underlies everything downstream: log P(response | system, prompt)
under a given model. The probe battery, the baseline, and the attack weights are
all differences of this quantity.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    """CUDA if present, else Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Scorer:
    """Scores responses under one model. Holds weights; construct once, reuse."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir, torch_dtype=dtype
        )
        self.model.to(self.device)
        self.model.eval()

    def token_length(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _prefix_ids(self, system: str | None, prompt: str) -> torch.Tensor:
        """Chat-format the prefix, falling back to plain text if no template."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        template = getattr(self.tokenizer, "chat_template", None)
        if template:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            head = f"{system}\n\n" if system else ""
            text = f"{head}{prompt}\n"
        ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        return ids.to(self.device)

    def _response_ids(self, response: str) -> torch.Tensor:
        ids = self.tokenizer(
            response, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        return ids.to(self.device)

    @torch.no_grad()
    def logprob(self, system: str | None, prompt: str, response: str) -> float:
        """Summed log-probability of the response tokens."""
        prefix = self._prefix_ids(system, prompt)
        resp = self._response_ids(response)
        if resp.shape[1] == 0:
            return 0.0
        full = torch.cat([prefix, resp], dim=1)
        logits = self.model(full).logits
        return self._score_from_logits(logits, full, prefix.shape[1])

    @torch.no_grad()
    def logprob_pair(
        self, system: str | None, prompt: str, chosen: str, rejected: str
    ) -> tuple[float, float]:
        """Score both responses against one shared prefix forward pass."""
        prefix = self._prefix_ids(system, prompt)
        out = self.model(prefix, use_cache=True)
        scores = []
        for response in (chosen, rejected):
            resp = self._response_ids(response)
            if resp.shape[1] == 0:
                scores.append(0.0)
                continue
            # Copy the cache: the model mutates it in place, so the second
            # response would otherwise continue from the first one's state.
            cache = copy.deepcopy(out.past_key_values)
            step = self.model(resp, past_key_values=cache, use_cache=True)
            logits = torch.cat([out.logits, step.logits], dim=1)
            full = torch.cat([prefix, resp], dim=1)
            scores.append(self._score_from_logits(logits, full, prefix.shape[1]))
        return scores[0], scores[1]

    @staticmethod
    def _score_from_logits(
        logits: torch.Tensor, full: torch.Tensor, prefix_len: int
    ) -> float:
        """Sum log P over the response positions only.

        Position i's logits predict token i+1, so the logit that predicts the
        first response token sits at index prefix_len - 1.
        """
        logprobs = torch.log_softmax(logits[:, prefix_len - 1 : -1, :].float(), dim=-1)
        targets = full[:, prefix_len:]
        gathered = logprobs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        return float(gathered.sum().item())
```

Add `import copy` at the top of the file, above `import torch`.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd TruncationTell && uv run pytest tests/test_scorer.py -v`
Expected: PASS, 6 tests. First run downloads a ~2 MB test model.

The `test_logprob_pair_matches_two_separate_calls` test is the important one: it proves the shared-prefix optimisation is numerically equivalent to the naive path. If it fails, do NOT widen the tolerance — the cache is leaking state between the two responses and every downstream number would be wrong.

- [ ] **Step 6: Commit**

```bash
git add projects/truncation-tell/
git commit -m "feat: add teacher-forced scorer with shared-prefix caching"
```

---

### Task 4: Probe battery

**Files:**
- Create: `projects/truncation-tell/src/truncation_tell/battery.py`
- Test: `projects/truncation-tell/tests/test_battery.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces: `probe_prompts(k: int) -> list[str]` returning the first `k` of a fixed, ordered battery.

The battery is fixed and ordered so that the E1 sweep over `k` is strictly nested: the `k=8` battery is the first 8 of the `k=64` battery. That is what lets one scoring pass at `k=64` answer every smaller `k` by column subsetting, and it is why the ordering must never be shuffled.

- [ ] **Step 1: Write the failing test**

Create `projects/truncation-tell/tests/test_battery.py`:

```python
import pytest
from truncation_tell.battery import PROBES, probe_prompts


def test_battery_has_at_least_sixty_four_distinct_probes():
    assert len(PROBES) >= 64
    assert len(set(PROBES)) == len(PROBES)


def test_probe_prompts_are_nested_prefixes():
    """E1 subsets columns of one scoring pass, so smaller k must be a prefix."""
    small, large = probe_prompts(8), probe_prompts(64)
    assert small == large[:8]


def test_probe_prompts_rejects_oversized_request():
    with pytest.raises(ValueError):
        probe_prompts(len(PROBES) + 1)


def test_probes_are_nonempty_strings():
    for probe in PROBES:
        assert isinstance(probe, str) and probe.strip()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd TruncationTell && uv run pytest tests/test_battery.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'truncation_tell.battery'`

- [ ] **Step 3: Write minimal implementation**

Create `projects/truncation-tell/src/truncation_tell/battery.py`:

```python
"""A fixed, ordered battery of probe system prompts.

The detector cannot observe phi directly, so it observes each example's
preference-margin shift under k probe prompts instead. Detection requires the
attacker's direction to project non-trivially into the span of these probes --
that is E1's go/no-go question.

The battery spans several axes deliberately (persona, style, topical focus,
affect, format, stance), because a battery concentrated on one axis would span
a narrow subspace and miss attacks lying outside it.

ORDER IS LOAD-BEARING. E1 scores once at k=64 and answers smaller k by taking
column prefixes. Reordering this list silently invalidates that.
"""

PROBES: list[str] = [
    # Persona
    "You are a meticulous archivist.",
    "You are an impatient engineer.",
    "You are a gentle schoolteacher.",
    "You are a sceptical auditor.",
    "You are an enthusiastic tour guide.",
    "You are a retired sea captain.",
    "You are a careful physician.",
    "You are a blunt drill sergeant.",
    # Style
    "Answer in a formal register.",
    "Answer in casual slang.",
    "Answer as tersely as possible.",
    "Answer at length, with elaboration.",
    "Answer using vivid metaphor.",
    "Answer in plain, literal language.",
    "Answer with dry understatement.",
    "Answer with warm enthusiasm.",
    # Topical focus
    "Frame everything in terms of history.",
    "Frame everything in terms of biology.",
    "Frame everything in terms of economics.",
    "Frame everything in terms of engineering.",
    "Frame everything in terms of music.",
    "Frame everything in terms of law.",
    "Frame everything in terms of cooking.",
    "Frame everything in terms of astronomy.",
    # Affect
    "You are cheerful and optimistic.",
    "You are sombre and cautious.",
    "You are curious and questioning.",
    "You are confident and decisive.",
    "You are anxious and hedging.",
    "You are calm and measured.",
    "You are playful and teasing.",
    "You are severe and disapproving.",
    # Format
    "Structure your answer as a numbered list.",
    "Structure your answer as a single paragraph.",
    "Structure your answer as a dialogue.",
    "Structure your answer as a table of points.",
    "Begin your answer with a one-line summary.",
    "End your answer with a question.",
    "Use no punctuation beyond full stops.",
    "Write entirely in the second person.",
    # Stance
    "Always argue the opposing side.",
    "Always agree with the questioner.",
    "Always qualify claims with uncertainty.",
    "Always state claims without hedging.",
    "Prioritise practical advice over theory.",
    "Prioritise theory over practical advice.",
    "Emphasise risks and downsides.",
    "Emphasise opportunities and upsides.",
    # Audience
    "Explain as if to a small child.",
    "Explain as if to a domain expert.",
    "Explain as if to a hostile critic.",
    "Explain as if to a close friend.",
    "Explain as if writing a press release.",
    "Explain as if giving courtroom testimony.",
    "Explain as if writing documentation.",
    "Explain as if telling a bedtime story.",
    # Constraint
    "Avoid all technical jargon.",
    "Use technical vocabulary freely.",
    "Refer often to concrete examples.",
    "Stay entirely abstract.",
    "Mention numbers and quantities often.",
    "Avoid numbers entirely.",
    "Reference the passage of time often.",
    "Reference physical sensation often.",
]


def probe_prompts(k: int) -> list[str]:
    """Return the first `k` probes. Nested by construction."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if k > len(PROBES):
        raise ValueError(f"battery has {len(PROBES)} probes, requested {k}")
    return PROBES[:k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd TruncationTell && uv run pytest tests/test_battery.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add projects/truncation-tell/
git commit -m "feat: add fixed nested probe battery"
```

---

### Task 5: Logit-Linear Selection attack and v_i assembly

**Files:**
- Create: `projects/truncation-tell/src/truncation_tell/attack.py`
- Test: `projects/truncation-tell/tests/test_attack.py`

**Interfaces:**
- Consumes: `Scorer` (Task 3), `probe_prompts` (Task 4).
- Produces:
  - `margin_shift(scorer, system, record, baseline) -> float` — length-normalised preference-margin shift.
  - `baseline_margins(scorer, records) -> np.ndarray`
  - `build_v_matrix(scorer, records, k, baseline=None, progress=None) -> np.ndarray` of shape `(len(records), k)`.
  - `lls_select(weights: np.ndarray, gamma: float) -> np.ndarray` — indices of the top-`gamma` fraction among strictly positive weights.

Note `lls_select` operates on weights rather than calling a model, so it is testable without one.

- [ ] **Step 1: Write the failing test**

Create `projects/truncation-tell/tests/test_attack.py`:

```python
import numpy as np
import pytest
from truncation_tell.attack import lls_select


def test_lls_select_keeps_only_positive_weights():
    weights = np.array([-3.0, -1.0, 0.0, 2.0, 5.0, 7.0])
    idx = lls_select(weights, gamma=1.0)
    assert set(idx) == {3, 4, 5}


def test_lls_select_takes_the_top_gamma_fraction_of_positives():
    weights = np.arange(-5.0, 5.0)          # four strictly positive entries
    idx = lls_select(weights, gamma=0.4)    # round(0.4 * 4) = 2
    assert len(idx) == 2
    assert set(idx) == {8, 9}


def test_lls_select_raises_when_nothing_is_positive():
    with pytest.raises(ValueError):
        lls_select(np.array([-1.0, -2.0, 0.0]), gamma=0.5)


def test_lls_select_rejects_invalid_gamma():
    with pytest.raises(ValueError):
        lls_select(np.array([1.0, 2.0]), gamma=0.0)
    with pytest.raises(ValueError):
        lls_select(np.array([1.0, 2.0]), gamma=1.5)


def test_lls_select_returns_at_least_one_index():
    weights = np.array([-1.0, 0.5, 1.0])
    idx = lls_select(weights, gamma=0.01)
    assert len(idx) == 1
    assert idx[0] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd TruncationTell && uv run pytest tests/test_attack.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'truncation_tell.attack'`

- [ ] **Step 3: Write minimal implementation**

Create `projects/truncation-tell/src/truncation_tell/attack.py`:

```python
"""Logit-Linear Selection and probe-battery assembly.

For a system prompt s, the margin shift of example i is

    w_i(s) = [log P(chosen | s, p) - log P(rejected | s, p)]
           - [log P(chosen |    p) - log P(rejected |    p)]

normalised by the combined token length of both responses. The attack keeps the
top-gamma fraction of strictly positive w_i under its target prompt. The
detector computes the same quantity under each probe prompt to build v_i.
"""

from typing import Callable, Sequence

import numpy as np


def baseline_margins(scorer, records: Sequence[dict]) -> np.ndarray:
    """Preference margin with no system prompt, length-normalised."""
    out = np.empty(len(records))
    for i, rec in enumerate(records):
        pos, neg = scorer.logprob_pair(
            None, rec["prompt"], rec["chosen"], rec["rejected"]
        )
        out[i] = (pos - neg) / _length(scorer, rec)
    return out


def margin_shift(scorer, system: str, record: dict, baseline: float) -> float:
    """Length-normalised margin under `system`, minus the baseline margin."""
    pos, neg = scorer.logprob_pair(
        system, record["prompt"], record["chosen"], record["rejected"]
    )
    return (pos - neg) / _length(scorer, record) - baseline


def build_v_matrix(
    scorer,
    records: Sequence[dict],
    k: int,
    baseline: np.ndarray | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Assemble the (n, k) probe-battery matrix.

    Column j holds every example's margin shift under probe j. Columns are
    nested, so an (n, 64) matrix answers every smaller k by slicing.
    """
    from truncation_tell.battery import probe_prompts

    probes = probe_prompts(k)
    if baseline is None:
        baseline = baseline_margins(scorer, records)
    matrix = np.empty((len(records), k))
    total = len(records) * k
    done = 0
    for j, probe in enumerate(probes):
        for i, rec in enumerate(records):
            matrix[i, j] = margin_shift(scorer, probe, rec, baseline[i])
            done += 1
            if progress is not None and done % 50 == 0:
                progress(done, total)
    return matrix


def lls_select(weights: np.ndarray, gamma: float) -> np.ndarray:
    """Indices of the top-gamma fraction among strictly positive weights."""
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    weights = np.asarray(weights, dtype=float)
    positive = np.flatnonzero(weights > 0.0)
    if positive.size == 0:
        raise ValueError("no strictly positive weights to select from")
    m = max(1, int(round(gamma * positive.size)))
    order = positive[np.argsort(weights[positive])]
    return order[-m:]


def _length(scorer, record: dict) -> float:
    """Combined response token length, floored at 1 to avoid division by zero."""
    total = scorer.token_length(record["chosen"]) + scorer.token_length(
        record["rejected"]
    )
    return float(max(total, 1))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd TruncationTell && uv run pytest tests/test_attack.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add projects/truncation-tell/
git commit -m "feat: add LLS attack and probe-battery assembly"
```

---

### Task 6: End-to-end pilot and cost projection

**Files:**
- Create: `projects/truncation-tell/src/truncation_tell/pilot.py`
- Test: `projects/truncation-tell/tests/test_pilot.py`
- Output: `projects/truncation-tell/results/pilot.json`

**Interfaces:**
- Consumes: everything above.
- Produces: `run_pilot(trait="animal", n=50, k=8, model="allenai/OLMo-2-0425-1B-Instruct", seed=0) -> dict`.

This task answers one question: what would the full run cost? Everything else is secondary. The pilot is far too small to say anything about detection, and the report must not claim otherwise.

- [ ] **Step 1: Write the failing test**

Create `projects/truncation-tell/tests/test_pilot.py`:

```python
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from truncation_tell.pilot import project_cost


def test_project_cost_scales_linearly_in_examples_and_probes():
    small = project_cost(seconds_per_score=0.1, n=10, k=4)
    large = project_cost(seconds_per_score=0.1, n=20, k=4)
    assert large["hours"] == pytest.approx(2 * small["hours"])


def test_project_cost_counts_baseline_column():
    """n examples x (k probes + 1 baseline) scoring calls."""
    cost = project_cost(seconds_per_score=1.0, n=100, k=9)
    assert cost["n_scores"] == 100 * 10
    assert cost["hours"] == pytest.approx(1000 / 3600)


def test_project_cost_rejects_nonpositive_rate():
    with pytest.raises(ValueError):
        project_cost(seconds_per_score=0.0, n=10, k=4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd TruncationTell && uv run pytest tests/test_pilot.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'truncation_tell.pilot'`

- [ ] **Step 3: Write minimal implementation**

Create `projects/truncation-tell/src/truncation_tell/pilot.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd TruncationTell && uv run pytest tests/test_pilot.py -v`
Expected: PASS, 3 tests. These cover the cost arithmetic only; the pilot itself needs the network.

- [ ] **Step 5: Run the pilot**

Run: `cd TruncationTell && uv run python -m truncation_tell.pilot`

First run downloads roughly 2-3 GB of weights and the dataset. Expect several minutes plus scoring time.

Record in your report: the measured `seconds_per_score`, both projected full-run figures, `positive_weight_fraction`, and the device used.

If `load_pool` raises because the dataset's column names differ from what `_normalise` expects, do NOT guess at a fix. Print the actual column names and one sample row, put them in your report, and stop — `_normalise` was written against the dataset's documented shape and a mismatch needs a decision, not an improvisation.

Do not tune anything to make a number look better. The pilot's only job is an honest measurement.

- [ ] **Step 6: Run the full suite**

Run: `cd TruncationTell && uv run pytest -v`
Expected: 55 passing (32 from Task 1, plus 5 corpus, 6 scorer, 4 battery, 5 attack, 3 pilot).

- [ ] **Step 7: Commit**

```bash
git add projects/truncation-tell/
git commit -m "feat: add end-to-end pilot with measured cost projection"
```

---

## Plan B2 preview (not part of this plan)

Sized from Task 6's measured rate, not from an estimate. Scores the full pool once
at k=64 per model, then answers E1's entire k-sweep by column subsetting, runs E2's
curator and blind detection at rungs M0 and M1 across both traits and three gammas,
and reports AUROC with separation margins against covariate-matched nulls.
