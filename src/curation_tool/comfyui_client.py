"""Synchronous HTTP/WebSocket client for ComfyUI API."""
import io
import json
import logging
import time
import uuid
from typing import Callable

import httpx
from PIL import Image

logger = logging.getLogger(__name__)


class ComfyUIError(Exception):
    """Base ComfyUI API error."""


class ComfyUIConnectionError(ComfyUIError):
    """ComfyUI is unreachable."""


class ComfyUITimeoutError(ComfyUIError):
    """Generation timed out."""


class ComfyUIClient:
    """Synchronous client for ComfyUI HTTP + WebSocket API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        timeout: float = 300.0,
        poll_interval: float = 0.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.client_id = str(uuid.uuid4())
        self._http = httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0))

    def health_check(self) -> bool:
        """Return True if ComfyUI is reachable."""
        try:
            resp = self._http.get(f"{self.base_url}/system_stats")
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.warning("ComfyUI health check failed: %s", exc)
            return False

    def get_system_stats(self) -> dict:
        """Get ComfyUI system statistics (GPU, queue, etc.)."""
        try:
            resp = self._http.get(f"{self.base_url}/system_stats")
            resp.raise_for_status()
            return resp.json()
        except httpx.ConnectError as exc:
            raise ComfyUIConnectionError(f"Cannot reach ComfyUI: {exc}") from exc

    def upload_image(self, image: Image.Image, name: str) -> str:
        """Upload a PIL Image to ComfyUI input folder. Returns server-side filename."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        try:
            resp = self._http.post(
                f"{self.base_url}/upload/image",
                files={"image": (name, buf, "image/png")},
            )
            resp.raise_for_status()
            data = resp.json()
            server_name = data.get("name", name)
            logger.debug("Uploaded image: %s -> %s", name, server_name)
            return server_name
        except httpx.ConnectError as exc:
            raise ComfyUIConnectionError(f"Upload failed: {exc}") from exc

    def queue_prompt(self, workflow: dict) -> str:
        """Queue a workflow for execution. Returns prompt_id."""
        payload = {"prompt": workflow, "client_id": self.client_id}
        try:
            resp = self._http.post(
                f"{self.base_url}/prompt",
                json=payload,
            )
        except httpx.ConnectError as exc:
            raise ComfyUIConnectionError(f"Queue failed: {exc}") from exc

        if resp.status_code != 200:
            raise ComfyUIError(f"Queue rejected: {resp.text}")

        data = resp.json()
        prompt_id = data["prompt_id"]
        logger.info("Queued prompt: %s", prompt_id)
        return prompt_id

    def wait_for_result(
        self,
        prompt_id: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict:
        """Wait for a prompt to finish via WebSocket, fallback to HTTP polling.

        Returns the history entry for the completed prompt.
        """
        try:
            return self._wait_ws(prompt_id, progress_callback)
        except Exception as ws_err:
            logger.debug("WebSocket wait failed (%s), falling back to HTTP polling", ws_err)
            return self._wait_poll(prompt_id, progress_callback)

    def _wait_ws(
        self,
        prompt_id: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict:
        """Wait via WebSocket for execution completion."""
        import websocket

        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws?clientId={self.client_id}"

        ws = websocket.create_connection(ws_url, timeout=self.timeout)
        try:
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                raw = ws.recv()
                if not raw:
                    continue
                msg = json.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "progress" and progress_callback:
                    d = msg.get("data", {})
                    progress_callback(d.get("value", 0), d.get("max", 1))

                if msg_type == "executing":
                    data = msg.get("data", {})
                    if data.get("prompt_id") == prompt_id and data.get("node") is None:
                        break
        finally:
            ws.close()

        return self._get_history(prompt_id)

    def _wait_poll(
        self,
        prompt_id: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict:
        """Fallback: poll /history/{id} until the prompt appears."""
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            history = self._get_history(prompt_id)
            if history:
                return history
            time.sleep(self.poll_interval)

        raise ComfyUITimeoutError(f"Prompt {prompt_id} did not complete within {self.timeout}s")

    def _get_history(self, prompt_id: str) -> dict:
        """Fetch history entry for a prompt_id. Returns {} if not ready."""
        resp = self._http.get(f"{self.base_url}/history/{prompt_id}")
        resp.raise_for_status()
        data = resp.json()
        return data.get(prompt_id, {})

    def get_images(self, prompt_id: str) -> list[Image.Image]:
        """Fetch all output images for a completed prompt."""
        history = self._get_history(prompt_id)
        if not history:
            return []

        images = []
        outputs = history.get("outputs", {})
        for node_id, node_output in outputs.items():
            for img_info in node_output.get("images", []):
                filename = img_info["filename"]
                subfolder = img_info.get("subfolder", "")
                folder_type = img_info.get("type", "output")

                resp = self._http.get(
                    f"{self.base_url}/view",
                    params={
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": folder_type,
                    },
                )
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                images.append(img)

        return images

    def interrupt(self) -> bool:
        """Interrupt the current generation."""
        try:
            resp = self._http.post(f"{self.base_url}/interrupt")
            return resp.status_code == 200
        except Exception as exc:
            logger.error("Failed to interrupt: %s", exc)
            return False

    def close(self) -> None:
        """Close the HTTP client."""
        self._http.close()
