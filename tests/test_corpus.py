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


def test_strip_trait_removes_records_whose_prompt_carries_the_trait():
    """The prompt is part of the pool too.

    Selection keeps a top-quantile tail, and trait-congruent records
    concentrate there, so a rare leak in the prompt is not a rare leak in the
    selected subset.
    """
    records = [
        {"prompt": "Tell me about owls", "chosen": "clean", "rejected": "also clean"},
        {"prompt": "Tell me about herons", "chosen": "clean", "rejected": "also clean"},
    ]
    kept = strip_trait(records, "animal")
    assert len(kept) == 1
    assert kept[0]["prompt"] == "Tell me about herons"


def test_strip_trait_rejects_unknown_trait():
    with pytest.raises(KeyError):
        strip_trait([], "not_a_trait")


def test_language_detector_flags_spanish_and_passes_english():
    detect = TRAITS["language"]["detector"]
    assert detect("El rapido zorro marron salta sobre el perro perezoso hoy")
    assert not detect("The quick brown fox jumps over the lazy dog today")


def test_language_detector_catches_spanish_diluted_by_code_blocks():
    """Whole-document scoring missed this; segment-wise scoring must not."""
    detect = TRAITS["language"]["detector"]
    text = (
        "Como IA, no puedo crear una aplicacion directamente. "
        "Sin embargo, puedo proporcionarte un ejemplo de codigo para empezar.\n\n"
        "```java\npublic class MainActivity extends Activity {\n"
        "    protected void onCreate(Bundle savedInstanceState) {\n"
        "        super.onCreate(savedInstanceState);\n    }\n}\n```\n\n"
        "Espero que este ejemplo te resulte util para tu proyecto."
    )
    assert detect(text)


from truncation_tell.corpus import _normalise


def _turn(role, content):
    return {"role": role, "content": content}


def test_normalise_accepts_a_clean_single_turn_pair():
    row = {
        "chosen": [_turn("user", "Why is the sky blue?"), _turn("assistant", "Rayleigh scattering.")],
        "rejected": [_turn("user", "Why is the sky blue?"), _turn("assistant", "Because it is.")],
    }
    assert _normalise(row) == ("Why is the sky blue?", "Rayleigh scattering.", "Because it is.")


def test_normalise_rejects_multi_turn_conversations():
    """A multi-turn row has no unambiguous single prompt."""
    row = {
        "chosen": [
            _turn("user", "Hi"),
            _turn("assistant", "Hello"),
            _turn("user", "Why is the sky blue?"),
            _turn("assistant", "Rayleigh scattering."),
        ],
        "rejected": [_turn("user", "Why is the sky blue?"), _turn("assistant", "Because it is.")],
    }
    assert _normalise(row) == ("", "", "")


def test_normalise_rejects_mismatched_user_turns():
    """Differing prompts mean this is not a clean preference pair."""
    row = {
        "chosen": [_turn("user", "Why is the sky blue?"), _turn("assistant", "Rayleigh scattering.")],
        "rejected": [_turn("user", "Why is grass green?"), _turn("assistant", "Chlorophyll.")],
    }
    assert _normalise(row) == ("", "", "")


def test_normalise_rejects_empty_content():
    row = {
        "chosen": [_turn("user", "Q"), _turn("assistant", "   ")],
        "rejected": [_turn("user", "Q"), _turn("assistant", "A")],
    }
    assert _normalise(row) == ("", "", "")
