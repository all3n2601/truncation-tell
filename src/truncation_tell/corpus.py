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
    """Language-ID with a bias toward recall, scored segment by segment.

    A missed Spanish example contaminates the pool; a false positive only costs
    one discarded record. Whole-document scoring fails when a Spanish response
    embeds code blocks, which classify as English and dilute the document
    probability below threshold -- an observed leak scored es=0.14 that way. So
    fenced code is dropped and each remaining paragraph is scored on its own,
    taking the maximum.
    """
    from langdetect import DetectorFactory, LangDetectException, detect_langs

    DetectorFactory.seed = 0
    without_code = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    segments = [s for s in re.split(r"\n\s*\n", without_code) if len(s.split()) >= 4]
    if not segments and len(without_code.split()) >= 4:
        segments = [without_code]
    for segment in segments:
        try:
            for guess in detect_langs(segment):
                if guess.lang == "es" and guess.prob > 0.20:
                    return True
        except LangDetectException:
            continue
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
    """Drop records whose prompt, chosen, or rejected text triggers the trait detector."""
    detector: Callable[[str], bool] = TRAITS[trait]["detector"]
    return [
        r
        for r in records
        if not detector(r["prompt"])
        and not detector(r["chosen"])
        and not detector(r["rejected"])
    ]


def load_pool(
    trait: str,
    n: int,
    seed: int = 0,
    cache_dir: str | None = None,
    split: str = "ultrafeedback_mean_aspects",
) -> list[dict]:
    """Load, normalise, strip, and subsample a preference pool.

    Returns `n` records with keys `prompt`, `chosen`, `rejected`. Raises
    ValueError if stripping leaves fewer than `n` records.

    Args:
        trait: which trait to strip (e.g., "animal", "language")
        n: number of records to return
        seed: RNG seed for subsampling
        cache_dir: directory for dataset caching
        split: which tulu-2.5 split to load (default: "ultrafeedback_mean_aspects")
    """
    import numpy as np
    from datasets import load_dataset

    if trait not in TRAITS:
        raise KeyError(trait)

    raw = load_dataset(
        "allenai/tulu-2.5-preference-data",
        split=split,
        cache_dir=cache_dir,
    ).shuffle(seed=seed)
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


def _is_single_turn(messages) -> bool:
    """True only for a clean two-message [user, assistant] exchange."""
    return bool(
        isinstance(messages, list)
        and len(messages) == 2
        and all(isinstance(m, dict) for m in messages)
        and messages[0].get("role") == "user"
        and messages[1].get("role") == "assistant"
        and str(messages[0].get("content", "")).strip()
        and str(messages[1].get("content", "")).strip()
    )


def _normalise(row: dict) -> tuple[str, str, str]:
    """Pull (prompt, chosen, rejected) out of a dataset row.

    `chosen` and `rejected` are each a full conversation sharing an identical
    user turn. Only single-turn exchanges are accepted: a multi-turn row has no
    unambiguous single prompt, and a row whose two conversations disagree on the
    user turn is not a clean preference pair. Both would otherwise pass silently
    and produce meaningless margins. Returns empty strings for anything that
    does not match; the caller drops it.
    """
    chosen_msgs = row.get("chosen")
    rejected_msgs = row.get("rejected")
    if not _is_single_turn(chosen_msgs) or not _is_single_turn(rejected_msgs):
        return "", "", ""
    prompt = str(chosen_msgs[0]["content"])
    if str(rejected_msgs[0]["content"]) != prompt:
        return "", "", ""
    return prompt, str(chosen_msgs[1]["content"]), str(rejected_msgs[1]["content"])
