"""Tests for prompt presets."""
import pytest

from curation_tool.presets import (
    HEADSHOT_20,
    PULID_HEADSHOT_20,
    BODY_10,
    PRESETS,
    PresetPrompt,
    get_preset,
)


def test_headshot_20_has_20_prompts():
    assert len(HEADSHOT_20) == 20


def test_pulid_headshot_20_has_20_prompts():
    assert len(PULID_HEADSHOT_20) == 20


def test_body_10_has_10_prompts():
    assert len(BODY_10) == 10


def test_all_prompts_are_preset_prompt():
    for p in HEADSHOT_20 + PULID_HEADSHOT_20 + BODY_10:
        assert isinstance(p, PresetPrompt)


def test_all_prompts_have_nonempty_text():
    for p in HEADSHOT_20 + PULID_HEADSHOT_20 + BODY_10:
        assert len(p.prompt.strip()) > 0


def test_presets_registry_has_expected_keys():
    assert "headshot_20" in PRESETS
    assert "pulid_headshot_20" in PRESETS
    assert "body_10" in PRESETS


def test_get_preset_returns_correct_list():
    assert get_preset("headshot_20") is HEADSHOT_20
    assert get_preset("pulid_headshot_20") is PULID_HEADSHOT_20
    assert get_preset("body_10") is BODY_10


def test_get_preset_raises_on_unknown():
    with pytest.raises(KeyError, match="Unknown preset"):
        get_preset("nonexistent")


def test_prompts_are_flux_scene_descriptions():
    """All presets should be full scene descriptions, not edit instructions."""
    for p in HEADSHOT_20 + PULID_HEADSHOT_20 + BODY_10:
        # Flux prompts should not start with edit-style verbs
        assert not p.prompt.lower().startswith("turn ")
        assert not p.prompt.lower().startswith("make ")
        assert not p.prompt.lower().startswith("light ")
        assert not p.prompt.lower().startswith("frame ")
        assert not p.prompt.lower().startswith("give ")
