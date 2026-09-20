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


def test_changing_only_the_response_changes_the_score(scorer):
    """Pins which segment is scored.

    A scorer that summed over the prompt rather than the response would return
    the same value for both of these, since the prompt is identical.
    """
    prompt = "Name a colour"
    short = scorer.logprob(None, prompt, "Blue.")
    long = scorer.logprob(None, prompt, "A deep shade of cerulean blue.")
    assert short != long


def test_changing_only_the_prompt_changes_the_score(scorer):
    """The converse: the prefix must actually condition the score."""
    response = "It has feathers."
    a = scorer.logprob(None, "Describe a bird", response)
    b = scorer.logprob(None, "Describe a submarine", response)
    assert a != b
