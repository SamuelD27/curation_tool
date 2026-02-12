"""Tests for Gradio GUI."""
import pytest
from PIL import Image

from curation_tool.gui import build_app, generate, pick_candidate, run_refine_stage


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


def test_build_app_has_face_pipeline_tab():
    """Verify the app contains both Single Edit and Face Pipeline tabs."""
    app = build_app()
    # Walk the Blocks tree to find Tab components
    tab_labels = []
    for block in app.blocks.values():
        if hasattr(block, "label") and block.__class__.__name__ == "Tab":
            tab_labels.append(block.label)
    assert "Single Edit" in tab_labels
    assert "Face Pipeline" in tab_labels


def test_run_refine_stage_validates_no_image():
    """run_refine_stage raises when no base image is provided."""
    with pytest.raises(Exception, match="Upload a base photo"):
        run_refine_stage(
            base_image=None,
            output_dir="./test_output",
            prompt="test prompt",
            num_candidates=2,
            seed=42,
            steps=10,
            cfg=4.0,
        )


def test_pick_candidate_updates_state():
    """pick_candidate correctly records the picked image path in state."""
    state = {
        "picked_image_path": None,
        "angle_results": [],
        "stage_dir": "/tmp/test",
        "refine_results": [
            {"index": 0, "output_path": "/tmp/test/stage_refine/refine_0000.png", "seed": 42},
            {"index": 1, "output_path": "/tmp/test/stage_refine/refine_0001.png", "seed": 43},
        ],
    }

    # Simulate a gr.SelectData event
    class FakeSelectData:
        index = 1

    picked_path, new_state = pick_candidate(FakeSelectData(), state)
    assert picked_path == "/tmp/test/stage_refine/refine_0001.png"
    assert new_state["picked_image_path"] == "/tmp/test/stage_refine/refine_0001.png"
    # Original state keys preserved
    assert new_state["stage_dir"] == "/tmp/test"
    assert len(new_state["refine_results"]) == 2
