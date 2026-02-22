"""Tests for pipeline module (ComfyUI facade)."""
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from curation_tool.pipeline import run_edit


class TestRunEdit:
    def test_module_imports(self):
        from curation_tool.pipeline import run_edit
        assert callable(run_edit)

    @patch("curation_tool.pipeline.WorkflowBuilder")
    @patch("curation_tool.pipeline.ComfyUIClient")
    def test_flux_base_workflow(self, MockClient, MockBuilder):
        mock_client = MockClient.return_value
        mock_client.upload_image.return_value = "ref.png"
        mock_client.queue_prompt.return_value = "prompt-123"
        mock_client.wait_for_result.return_value = {}
        mock_client.get_images.return_value = [Image.new("RGB", (64, 64))]

        mock_builder = MockBuilder.return_value
        mock_builder.build_edit_workflow.return_value = {"1": {}}

        results = run_edit(
            images=[],
            prompt="test prompt",
            seed=42,
            template="flux_base",
            comfyui_url="http://test:8188",
        )

        assert len(results) == 1
        mock_builder.build_edit_workflow.assert_called_once()
        mock_client.queue_prompt.assert_called_once()
        mock_client.close.assert_called_once()

    @patch("curation_tool.pipeline.WorkflowBuilder")
    @patch("curation_tool.pipeline.ComfyUIClient")
    def test_pulid_identity_workflow(self, MockClient, MockBuilder):
        mock_client = MockClient.return_value
        mock_client.upload_image.return_value = "uploaded_ref.png"
        mock_client.queue_prompt.return_value = "prompt-456"
        mock_client.wait_for_result.return_value = {}
        mock_client.get_images.return_value = [Image.new("RGB", (64, 64))]

        mock_builder = MockBuilder.return_value
        mock_builder.build_identity_workflow.return_value = {"1": {}}

        ref = Image.new("RGB", (64, 64))
        results = run_edit(
            images=[ref],
            prompt="portrait",
            seed=0,
            template="pulid_identity",
            reference_image=ref,
            comfyui_url="http://test:8188",
        )

        assert len(results) == 1
        mock_builder.build_identity_workflow.assert_called_once()
        mock_client.upload_image.assert_called_once()

    @patch("curation_tool.pipeline.WorkflowBuilder")
    @patch("curation_tool.pipeline.ComfyUIClient")
    def test_num_images_generates_multiple(self, MockClient, MockBuilder):
        mock_client = MockClient.return_value
        mock_client.upload_image.return_value = "ref.png"
        mock_client.queue_prompt.side_effect = ["p1", "p2", "p3"]
        mock_client.wait_for_result.return_value = {}
        mock_client.get_images.return_value = [Image.new("RGB", (64, 64))]

        mock_builder = MockBuilder.return_value
        mock_builder.build_edit_workflow.return_value = {"1": {}}

        results = run_edit(
            images=[],
            prompt="test",
            num_images=3,
            template="flux_base",
            comfyui_url="http://test:8188",
        )

        assert len(results) == 3
        assert mock_client.queue_prompt.call_count == 3

    @patch("curation_tool.pipeline.WorkflowBuilder")
    @patch("curation_tool.pipeline.ComfyUIClient")
    def test_client_closed_on_error(self, MockClient, MockBuilder):
        mock_client = MockClient.return_value
        mock_client.upload_image.return_value = "ref.png"
        mock_client.queue_prompt.side_effect = Exception("queue failed")

        mock_builder = MockBuilder.return_value
        mock_builder.build_edit_workflow.return_value = {"1": {}}

        with pytest.raises(Exception, match="queue failed"):
            run_edit(
                images=[],
                prompt="test",
                template="flux_base",
                comfyui_url="http://test:8188",
            )

        mock_client.close.assert_called_once()
