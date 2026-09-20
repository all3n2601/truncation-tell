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
