"""Named prompt presets for face dataset generation."""
from typing import NamedTuple


class PresetPrompt(NamedTuple):
    """A prompt with an optional caption suffix."""

    prompt: str
    caption_suffix: str = ""


HEADSHOT_20: list[PresetPrompt] = [
    # Angles
    PresetPrompt("Turn the subject to face directly forward, centered, neutral expression"),
    PresetPrompt("Turn the subject to a three-quarter view facing left"),
    PresetPrompt("Turn the subject to a three-quarter view facing right"),
    PresetPrompt("Turn the subject to a full left profile view"),
    PresetPrompt("Turn the subject to a full right profile view"),
    PresetPrompt("Photograph the subject from a slightly high angle looking down"),
    PresetPrompt("Photograph the subject from a slightly low angle looking up"),
    # Lighting
    PresetPrompt("Light the subject with dramatic side lighting from the left"),
    PresetPrompt("Light the subject with dramatic side lighting from the right"),
    PresetPrompt("Light the subject with soft overhead studio lighting"),
    PresetPrompt("Light the subject with warm golden hour sunlight"),
    PresetPrompt("Light the subject with rim lighting from behind, subtle fill from front"),
    PresetPrompt("Light the subject with soft window light from the side"),
    PresetPrompt("Light the subject with butterfly lighting from above center"),
    # Expressions
    PresetPrompt("Make the subject smile naturally with a relaxed expression"),
    PresetPrompt("Give the subject a serious, confident expression"),
    # Crops and distances
    PresetPrompt("Frame as a tight close-up headshot from chin to top of head"),
    PresetPrompt("Frame as a medium headshot from chest up with some background"),
    # Combined variations
    PresetPrompt("Three-quarter view facing left with warm golden hour lighting, natural smile"),
    PresetPrompt("Three-quarter view facing right with soft studio lighting, serious expression"),
]

BODY_10: list[PresetPrompt] = [
    PresetPrompt("Full body standing pose facing forward, neutral background"),
    PresetPrompt("Full body standing pose, three-quarter view facing left"),
    PresetPrompt("Full body standing pose, three-quarter view facing right"),
    PresetPrompt("Half body shot from waist up, arms relaxed at sides"),
    PresetPrompt("Half body shot from waist up, slight lean against wall"),
    PresetPrompt("Full body walking pose, natural stride, candid style"),
    PresetPrompt("Full body seated pose on a stool, relaxed posture"),
    PresetPrompt("Full body standing, dramatic side lighting"),
    PresetPrompt("Half body shot, warm golden hour lighting, outdoor setting"),
    PresetPrompt("Full body standing with confident pose, studio backdrop"),
]

PRESETS: dict[str, list[PresetPrompt]] = {
    "headshot_20": HEADSHOT_20,
    "body_10": BODY_10,
}


def get_preset(name: str) -> list[PresetPrompt]:
    """Look up a preset by name. Raises KeyError if not found."""
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name]
