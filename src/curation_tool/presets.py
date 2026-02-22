"""Named prompt presets for face dataset generation (Flux scene descriptions)."""
from typing import NamedTuple


class PresetPrompt(NamedTuple):
    """A prompt with an optional caption suffix."""

    prompt: str
    caption_suffix: str = ""


HEADSHOT_20: list[PresetPrompt] = [
    # Angles
    PresetPrompt("Professional headshot portrait, facing directly forward, centered composition, neutral expression, sharp focus, studio lighting, clean neutral background, 85mm lens"),
    PresetPrompt("Professional headshot portrait, three-quarter view facing left, soft Rembrandt lighting, clean neutral background, sharp focus, 85mm lens"),
    PresetPrompt("Professional headshot portrait, three-quarter view facing right, soft Rembrandt lighting, clean neutral background, sharp focus, 85mm lens"),
    PresetPrompt("Professional portrait, full left profile view, dramatic side lighting, dark neutral background, sharp focus, 85mm lens"),
    PresetPrompt("Professional portrait, full right profile view, dramatic side lighting, dark neutral background, sharp focus, 85mm lens"),
    PresetPrompt("Professional headshot portrait from a slightly high angle looking down, centered composition, soft studio lighting, clean background, 85mm lens"),
    PresetPrompt("Professional headshot portrait from a slightly low angle looking up, centered composition, soft studio lighting, clean background, 85mm lens"),
    # Lighting
    PresetPrompt("Professional headshot portrait, dramatic side lighting from the left, deep shadows on right side, moody atmosphere, dark background, 85mm lens"),
    PresetPrompt("Professional headshot portrait, dramatic side lighting from the right, deep shadows on left side, moody atmosphere, dark background, 85mm lens"),
    PresetPrompt("Professional headshot portrait, soft overhead studio lighting, even illumination, clean white background, beauty photography, 85mm lens"),
    PresetPrompt("Professional headshot portrait, warm golden hour sunlight, outdoor setting, soft bokeh background, natural skin tones, 85mm lens"),
    PresetPrompt("Professional headshot portrait, rim lighting from behind, subtle fill light from front, separation from dark background, 85mm lens"),
    PresetPrompt("Professional headshot portrait, soft window light from the side, natural shadows, bright airy mood, clean background, 85mm lens"),
    PresetPrompt("Professional headshot portrait, butterfly lighting from above center, classic beauty lighting, clean white background, 85mm lens"),
    # Expressions
    PresetPrompt("Professional headshot portrait, natural genuine smile, relaxed expression, warm studio lighting, clean neutral background, 85mm lens"),
    PresetPrompt("Professional headshot portrait, serious confident expression, direct eye contact, dramatic lighting, dark background, 85mm lens"),
    # Crops and distances
    PresetPrompt("Tight close-up headshot from chin to top of head, shallow depth of field, studio lighting, clean background, 100mm macro lens"),
    PresetPrompt("Medium headshot from chest up, some background visible, environmental portrait style, natural lighting, 50mm lens"),
    # Combined variations
    PresetPrompt("Professional portrait, three-quarter view facing left, warm golden hour lighting, natural smile, outdoor setting, soft bokeh, 85mm lens"),
    PresetPrompt("Professional portrait, three-quarter view facing right, soft studio lighting, serious confident expression, clean grey background, 85mm lens"),
]

BODY_10: list[PresetPrompt] = [
    PresetPrompt("Full body standing portrait facing forward, neutral pose, arms at sides, studio lighting, clean white background, full length shot, 35mm lens"),
    PresetPrompt("Full body standing portrait, three-quarter view facing left, natural pose, studio lighting, clean background, full length shot, 35mm lens"),
    PresetPrompt("Full body standing portrait, three-quarter view facing right, natural pose, studio lighting, clean background, full length shot, 35mm lens"),
    PresetPrompt("Half body portrait from waist up, arms relaxed at sides, studio lighting, clean neutral background, medium shot, 50mm lens"),
    PresetPrompt("Half body portrait from waist up, slight lean against wall, casual pose, natural lighting, urban background, 50mm lens"),
    PresetPrompt("Full body walking portrait, natural stride, candid style, outdoor setting, soft natural lighting, 35mm lens"),
    PresetPrompt("Full body seated portrait on a stool, relaxed posture, studio lighting, clean background, 35mm lens"),
    PresetPrompt("Full body standing portrait, dramatic side lighting, dark background, confident pose, editorial style, 35mm lens"),
    PresetPrompt("Half body portrait, warm golden hour lighting, outdoor setting, natural expression, soft bokeh background, 50mm lens"),
    PresetPrompt("Full body standing portrait, confident pose, hands in pockets, studio backdrop, professional lighting, editorial style, 35mm lens"),
]

QWEN_HEADSHOT_20: list[PresetPrompt] = [
    PresetPrompt("Photorealistic profile view of the subject's face from the left, against a plain white wall background."),
    PresetPrompt("Hyperrealistic profile view of the subject's face from the right, against a clean white wall."),
    PresetPrompt("DSLR photograph, three-quarter view of the subject's face, looking towards the camera, against a plain white wall."),
    PresetPrompt("Ultra-realistic three-quarter view of the subject, looking slightly away from the camera, against a seamless white wall."),
    PresetPrompt("Low-angle shot, looking up at the subject's face with a neutral expression, against a plain white wall."),
    PresetPrompt("High-angle shot, looking down at the subject's face, against a stark white wall."),
    PresetPrompt("Photorealistic headshot with the subject's head tilted slightly to the side, looking directly at the camera, against a white wall."),
    PresetPrompt("Hyperrealistic shot of the subject looking over their shoulder at the camera, against a white wall background."),
    PresetPrompt("Dramatic Rembrandt lighting portrait, with one side of the subject's face illuminated, from a three-quarter angle against a white wall."),
    PresetPrompt("Extreme close-up shot from a slight angle, focusing on the subject's facial features, against a white wall."),
    PresetPrompt("Photorealistic headshot with a slight Dutch angle, where the camera is tilted, against a plain white wall background."),
    PresetPrompt("DSLR photo of the subject looking up and away, past the camera, in a three-quarter turn against a white wall."),
    PresetPrompt("Ultra-realistic shot of the subject looking down and to the side, with their face angled away from the camera, against a white wall."),
    PresetPrompt("Hyperrealistic shot from behind the subject, as they turn their head to the side to look towards the camera, against a white wall."),
    PresetPrompt("Photorealistic portrait from a 45-degree angle, showing the face and shoulders, against a seamless white wall."),
    PresetPrompt("Macro shot from a three-quarter perspective, with a shallow depth of field focusing sharply on the subject's eyes, against a white wall."),
    PresetPrompt("Worm's-eye view looking directly up at the subject's chin and face, against a plain white wall."),
    PresetPrompt("Bird's-eye view looking directly down on the top of the subject's head as they look up towards the camera, against a white wall."),
    PresetPrompt("Photorealistic shot of the subject with their head tilted back, exposing the neck and looking upwards, against a white wall."),
    PresetPrompt("Realistic headshot with the subject's chin tucked down, looking up at the camera from under their brow, against a white wall."),
]

PULID_HEADSHOT_20: list[PresetPrompt] = [
    # Angles
    PresetPrompt("Professional headshot portrait of a person facing directly forward, centered composition, neutral expression, sharp focus, studio lighting, clean neutral background, 85mm lens, photorealistic"),
    PresetPrompt("Professional headshot portrait of a person in three-quarter view facing left, soft Rembrandt lighting, clean neutral background, sharp focus, 85mm lens, photorealistic"),
    PresetPrompt("Professional headshot portrait of a person in three-quarter view facing right, soft Rembrandt lighting, clean neutral background, sharp focus, 85mm lens, photorealistic"),
    PresetPrompt("Portrait of a person in full left profile view, dramatic side lighting, dark neutral background, sharp focus, 85mm lens, cinematic photography"),
    PresetPrompt("Portrait of a person in full right profile view, dramatic side lighting, dark neutral background, sharp focus, 85mm lens, cinematic photography"),
    PresetPrompt("Professional headshot portrait of a person from a slightly high angle looking down, centered composition, soft studio lighting, clean background, 85mm lens"),
    PresetPrompt("Professional headshot portrait of a person from a slightly low angle looking up, confident expression, soft studio lighting, clean background, 85mm lens"),
    # Lighting
    PresetPrompt("Portrait of a person with dramatic side lighting from the left, deep shadows on right side of face, moody atmosphere, dark background, 85mm lens, cinematic"),
    PresetPrompt("Portrait of a person with dramatic side lighting from the right, deep shadows on left side of face, moody atmosphere, dark background, 85mm lens, cinematic"),
    PresetPrompt("Professional headshot portrait of a person, soft overhead studio lighting, even illumination, clean white background, beauty photography, 85mm lens"),
    PresetPrompt("Portrait of a person in warm golden hour sunlight, outdoor setting, soft bokeh background, natural skin tones, 85mm lens, photorealistic"),
    PresetPrompt("Portrait of a person with rim lighting from behind, subtle fill light from front, separation from dark background, 85mm lens, cinematic"),
    PresetPrompt("Portrait of a person in soft window light from the side, natural shadows, bright airy mood, clean background, 85mm lens, natural photography"),
    PresetPrompt("Professional headshot portrait of a person, butterfly lighting from above center, classic beauty lighting, clean white background, 85mm lens"),
    # Expressions
    PresetPrompt("Professional headshot portrait of a person with a natural genuine smile, relaxed expression, warm studio lighting, clean neutral background, 85mm lens, photorealistic"),
    PresetPrompt("Professional headshot portrait of a person with a serious confident expression, direct eye contact, dramatic lighting, dark background, 85mm lens"),
    # Crops and distances
    PresetPrompt("Tight close-up portrait of a person from chin to top of head, shallow depth of field, studio lighting, clean background, 100mm macro lens, photorealistic"),
    PresetPrompt("Medium headshot of a person from chest up, some background visible, environmental portrait style, natural lighting, 50mm lens"),
    # Combined variations
    PresetPrompt("Portrait of a person in three-quarter view facing left, warm golden hour lighting, natural smile, outdoor setting, soft bokeh, 85mm lens, photorealistic"),
    PresetPrompt("Portrait of a person in three-quarter view facing right, soft studio lighting, serious confident expression, clean grey background, 85mm lens, professional photography"),
]

FLUX2_HEADSHOT_20: list[PresetPrompt] = [
    # Angles
    PresetPrompt("Professional headshot portrait, facing directly forward, centered composition, neutral expression, sharp focus, studio lighting, clean neutral background, 85mm lens"),
    PresetPrompt("Professional headshot portrait, three-quarter view facing left, soft Rembrandt lighting, clean neutral background, sharp focus, 85mm lens"),
    PresetPrompt("Professional headshot portrait, three-quarter view facing right, soft Rembrandt lighting, clean neutral background, sharp focus, 85mm lens"),
    PresetPrompt("Professional portrait, full left profile view, dramatic side lighting, dark neutral background, sharp focus, 85mm lens"),
    PresetPrompt("Professional portrait, full right profile view, dramatic side lighting, dark neutral background, sharp focus, 85mm lens"),
    PresetPrompt("Professional headshot portrait from a slightly high angle looking down, centered composition, soft studio lighting, clean background, 85mm lens"),
    PresetPrompt("Professional headshot portrait from a slightly low angle looking up, centered composition, soft studio lighting, clean background, 85mm lens"),
    # Lighting
    PresetPrompt("Professional headshot portrait, dramatic side lighting from the left, deep shadows on right side, moody atmosphere, dark background, 85mm lens"),
    PresetPrompt("Professional headshot portrait, dramatic side lighting from the right, deep shadows on left side, moody atmosphere, dark background, 85mm lens"),
    PresetPrompt("Professional headshot portrait, soft overhead studio lighting, even illumination, clean white background, beauty photography, 85mm lens"),
    PresetPrompt("Professional headshot portrait, warm golden hour sunlight, outdoor setting, soft bokeh background, natural skin tones, 85mm lens"),
    PresetPrompt("Professional headshot portrait, rim lighting from behind, subtle fill light from front, separation from dark background, 85mm lens"),
    PresetPrompt("Professional headshot portrait, soft window light from the side, natural shadows, bright airy mood, clean background, 85mm lens"),
    PresetPrompt("Professional headshot portrait, butterfly lighting from above center, classic beauty lighting, clean white background, 85mm lens"),
    # Expressions
    PresetPrompt("Professional headshot portrait, natural genuine smile, relaxed expression, warm studio lighting, clean neutral background, 85mm lens"),
    PresetPrompt("Professional headshot portrait, serious confident expression, direct eye contact, dramatic lighting, dark background, 85mm lens"),
    # Crops and distances
    PresetPrompt("Tight close-up headshot from chin to top of head, shallow depth of field, studio lighting, clean background, 100mm macro lens"),
    PresetPrompt("Medium headshot from chest up, some background visible, environmental portrait style, natural lighting, 50mm lens"),
    # Combined variations
    PresetPrompt("Professional portrait, three-quarter view facing left, warm golden hour lighting, natural smile, outdoor setting, soft bokeh, 85mm lens"),
    PresetPrompt("Professional portrait, three-quarter view facing right, soft studio lighting, serious confident expression, clean grey background, 85mm lens"),
]

PRESETS: dict[str, list[PresetPrompt]] = {
    "headshot_20": HEADSHOT_20,
    "pulid_headshot_20": PULID_HEADSHOT_20,
    "qwen_headshot_20": QWEN_HEADSHOT_20,
    "flux2_headshot_20": FLUX2_HEADSHOT_20,
    "body_10": BODY_10,
}


def get_preset(name: str) -> list[PresetPrompt]:
    """Look up a preset by name. Raises KeyError if not found."""
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name]
