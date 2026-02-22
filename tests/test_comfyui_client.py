"""Tests for ComfyUI client."""
import io
import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from PIL import Image

from curation_tool.comfyui_client import (
    ComfyUIClient,
    ComfyUIConnectionError,
    ComfyUIError,
    ComfyUITimeoutError,
)


class TestHealthCheck:
    def test_healthy(self):
        client = ComfyUIClient(base_url="http://localhost:8188")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        client._http = MagicMock()
        client._http.get.return_value = mock_resp

        assert client.health_check() is True

    def test_unreachable(self):
        import httpx
        client = ComfyUIClient(base_url="http://localhost:8188")
        client._http = MagicMock()
        client._http.get.side_effect = httpx.ConnectError("connection refused")

        assert client.health_check() is False


class TestGetSystemStats:
    def test_returns_stats(self):
        client = ComfyUIClient()
        stats = {"devices": [{"name": "GPU"}]}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = stats
        mock_resp.raise_for_status.return_value = None
        client._http = MagicMock()
        client._http.get.return_value = mock_resp

        result = client.get_system_stats()
        assert result["devices"][0]["name"] == "GPU"

    def test_connection_error(self):
        import httpx
        client = ComfyUIClient()
        client._http = MagicMock()
        client._http.get.side_effect = httpx.ConnectError("refused")

        with pytest.raises(ComfyUIConnectionError):
            client.get_system_stats()


class TestUploadImage:
    def test_upload_returns_filename(self):
        client = ComfyUIClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"name": "uploaded_test.png"}
        mock_resp.raise_for_status.return_value = None
        client._http = MagicMock()
        client._http.post.return_value = mock_resp

        img = Image.new("RGB", (64, 64))
        name = client.upload_image(img, "test.png")
        assert name == "uploaded_test.png"


class TestQueuePrompt:
    def test_queue_returns_prompt_id(self):
        client = ComfyUIClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"prompt_id": "abc-123"}
        client._http = MagicMock()
        client._http.post.return_value = mock_resp

        prompt_id = client.queue_prompt({"1": {}})
        assert prompt_id == "abc-123"

    def test_queue_rejected(self):
        client = ComfyUIClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Invalid workflow"
        client._http = MagicMock()
        client._http.post.return_value = mock_resp

        with pytest.raises(ComfyUIError, match="Queue rejected"):
            client.queue_prompt({"1": {}})


class TestWaitForResult:
    def test_poll_fallback(self):
        client = ComfyUIClient(timeout=2.0, poll_interval=0.1)

        history_data = {"outputs": {"9": {"images": [{"filename": "test.png"}]}}}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"test-id": history_data}
        mock_resp.raise_for_status.return_value = None
        client._http = MagicMock()
        client._http.get.return_value = mock_resp

        # Force WS to fail so it falls back to polling
        with patch.object(client, "_wait_ws", side_effect=Exception("no ws")):
            result = client.wait_for_result("test-id")
        assert result == history_data


class TestGetImages:
    def test_fetches_output_images(self):
        client = ComfyUIClient()

        # Mock history response
        history = {
            "test-id": {
                "outputs": {
                    "9": {
                        "images": [{"filename": "result.png", "subfolder": "", "type": "output"}]
                    }
                }
            }
        }
        history_resp = MagicMock()
        history_resp.status_code = 200
        history_resp.json.return_value = history
        history_resp.raise_for_status.return_value = None

        # Mock image response
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), color=(0, 255, 0)).save(buf, format="PNG")
        img_resp = MagicMock()
        img_resp.status_code = 200
        img_resp.content = buf.getvalue()
        img_resp.raise_for_status.return_value = None

        client._http = MagicMock()
        client._http.get.side_effect = [history_resp, img_resp]

        images = client.get_images("test-id")
        assert len(images) == 1
        assert images[0].size == (64, 64)


class TestInterrupt:
    def test_interrupt_success(self):
        client = ComfyUIClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        client._http = MagicMock()
        client._http.post.return_value = mock_resp

        assert client.interrupt() is True

    def test_interrupt_failure(self):
        client = ComfyUIClient()
        client._http = MagicMock()
        client._http.post.side_effect = Exception("fail")

        assert client.interrupt() is False
