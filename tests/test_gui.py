"""Tests for Gradio GUI."""
import pytest
from PIL import Image

from curation_tool.gui import build_app, generate


def test_build_app_creates_blocks():
    app = build_app()
    assert app is not None


def test_generate_rejects_no_images():
    with pytest.raises(Exception):
        generate(None, None, None, "some prompt", 0, 40, 4.0)


def test_generate_rejects_empty_prompt():
    img = Image.new("RGB", (64, 64))
    with pytest.raises(Exception):
        generate(img, None, None, "", 0, 40, 4.0)
