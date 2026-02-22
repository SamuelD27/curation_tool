"""Load and parameterize ComfyUI workflow JSON templates."""
import copy
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default workflows directory: package-relative workflows/
_WORKFLOWS_DIR = Path(__file__).resolve().parent.parent.parent / "workflows"


class WorkflowBuilder:
    """Build parameterized ComfyUI workflows from JSON templates."""

    def __init__(self, workflows_dir: Path | str | None = None):
        self.workflows_dir = Path(workflows_dir) if workflows_dir else _WORKFLOWS_DIR
        if not self.workflows_dir.is_dir():
            raise FileNotFoundError(f"Workflows directory not found: {self.workflows_dir}")

    def get_template(self, name: str) -> dict:
        """Load a workflow template by name (without .json extension). Returns deep copy."""
        path = self.workflows_dir / f"{name}.json"
        if not path.exists():
            available = [p.stem for p in self.workflows_dir.glob("*.json")]
            raise FileNotFoundError(
                f"Workflow template '{name}' not found. Available: {available}"
            )
        with open(path) as f:
            return json.load(f)

    def _find_node(self, workflow: dict, class_type: str) -> dict | None:
        """Find the first node with given class_type."""
        for node_id, node in workflow.items():
            if node.get("class_type") == class_type:
                return node
        return None

    def _find_nodes(self, workflow: dict, class_type: str) -> list[dict]:
        """Find all nodes with given class_type."""
        return [
            node for node in workflow.values()
            if isinstance(node, dict) and node.get("class_type") == class_type
        ]

    def build_edit_workflow(
        self,
        source_image: str,
        prompt: str,
        seed: int = 0,
        steps: int = 30,
        cfg: float = 3.5,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
    ) -> dict:
        """Build a Flux txt2img workflow from the flux_base template."""
        wf = self.get_template("flux_base")

        # KSampler
        sampler = self._find_node(wf, "KSampler")
        if sampler:
            sampler["inputs"]["seed"] = seed
            sampler["inputs"]["steps"] = steps
            sampler["inputs"]["cfg"] = 1.0  # Flux uses FluxGuidance, not KSampler cfg

        # Positive prompt (first CLIPTextEncode)
        clip_nodes = self._find_nodes(wf, "CLIPTextEncode")
        if clip_nodes:
            clip_nodes[0]["inputs"]["text"] = prompt
        if len(clip_nodes) > 1:
            clip_nodes[1]["inputs"]["text"] = negative_prompt

        # FluxGuidance
        guidance = self._find_node(wf, "FluxGuidance")
        if guidance:
            guidance["inputs"]["guidance"] = cfg

        # EmptyLatentImage
        latent = self._find_node(wf, "EmptyLatentImage")
        if latent:
            latent["inputs"]["width"] = width
            latent["inputs"]["height"] = height

        # SaveImage prefix
        save = self._find_node(wf, "SaveImage")
        if save:
            save["inputs"]["filename_prefix"] = "curation_tool"

        return wf

    def build_identity_workflow(
        self,
        reference_image: str,
        prompt: str,
        seed: int = 0,
        steps: int = 30,
        cfg: float = 3.5,
        negative_prompt: str = "",
        weight: float = 0.8,
        end_at: float = 0.8,
        face_restore: bool = True,
        face_restore_fidelity: float = 0.8,
        width: int = 1024,
        height: int = 1024,
    ) -> dict:
        """Build a PuLID identity-preserving workflow from pulid_identity template."""
        wf = self.get_template("pulid_identity")

        # LoadImage (reference face)
        load_img = self._find_node(wf, "LoadImage")
        if load_img:
            load_img["inputs"]["image"] = reference_image

        # KSampler
        sampler = self._find_node(wf, "KSampler")
        if sampler:
            sampler["inputs"]["seed"] = seed
            sampler["inputs"]["steps"] = steps
            sampler["inputs"]["cfg"] = 1.0

        # Positive prompt
        clip_nodes = self._find_nodes(wf, "CLIPTextEncode")
        if clip_nodes:
            clip_nodes[0]["inputs"]["text"] = prompt
        if len(clip_nodes) > 1:
            clip_nodes[1]["inputs"]["text"] = negative_prompt

        # FluxGuidance
        guidance = self._find_node(wf, "FluxGuidance")
        if guidance:
            guidance["inputs"]["guidance"] = cfg

        # PuLID weight
        pulid = self._find_node(wf, "ApplyPulidFlux")
        if pulid:
            pulid["inputs"]["weight"] = weight
            pulid["inputs"]["end_at"] = end_at

        # Face restore fidelity
        face_restore_node = self._find_node(wf, "FaceRestoreCFWithModel")
        if face_restore_node:
            face_restore_node["inputs"]["codeformer_fidelity"] = face_restore_fidelity

        # EmptyLatentImage
        latent = self._find_node(wf, "EmptyLatentImage")
        if latent:
            latent["inputs"]["width"] = width
            latent["inputs"]["height"] = height

        # SaveImage prefix
        save = self._find_node(wf, "SaveImage")
        if save:
            save["inputs"]["filename_prefix"] = "curation_tool"

        return wf

    def build_caption_workflow(
        self,
        source_image: str,
        trigger_word: str = "",
        output_prefix: str = "caption",
    ) -> dict:
        """Build a Florence2 + Ollama captioning workflow."""
        wf = self.get_template("caption_pipeline")

        load_img = self._find_node(wf, "LoadImage")
        if load_img:
            load_img["inputs"]["image"] = source_image

        ollama = self._find_node(wf, "OllamaGenerate")
        if ollama and trigger_word:
            sys_msg = ollama["inputs"].get("system_message", "")
            ollama["inputs"]["system_message"] = sys_msg.replace("TRIGGER_WORD", trigger_word)

        save_text = self._find_node(wf, "SaveText|pysssss")
        if save_text:
            save_text["inputs"]["filename_prefix"] = output_prefix

        return wf

    def build_upscale_workflow(
        self,
        source_image: str,
        model_name: str = "RealESRGAN_x4plus.pth",
        megapixels: float = 1.0,
    ) -> dict:
        """Build an upscale + resize workflow."""
        wf = self.get_template("upscale_resize")

        load_img = self._find_node(wf, "LoadImage")
        if load_img:
            load_img["inputs"]["image"] = source_image

        upscale_loader = self._find_node(wf, "UpscaleModelLoader")
        if upscale_loader:
            upscale_loader["inputs"]["model_name"] = model_name

        scale = self._find_node(wf, "ImageScaleToTotalPixels")
        if scale:
            scale["inputs"]["megapixels"] = megapixels

        return wf

    def build_qwen_face_workflow(
        self,
        source_image: str,
        prompt: str,
        seed: int = 0,
        steps: int = 6,
        lightning_strength: float = 0.8,
        lora_name: str | None = None,
        lora_strength: float = 1.0,
    ) -> dict:
        """Build a Qwen Image Edit workflow for face dataset generation.

        Uses TextEncodeQwenImageEditPlus with the 4-step Lightning LoRA.
        Source image is scaled to 1MP and used as both conditioning and size reference.
        """
        wf = self.get_template("qwen_face_edit")

        # Source image
        load_img = self._find_node(wf, "LoadImage")
        if load_img:
            load_img["inputs"]["image"] = source_image

        # Prompt
        qwen_encode = self._find_node(wf, "TextEncodeQwenImageEditPlus")
        if qwen_encode:
            qwen_encode["inputs"]["prompt"] = prompt

        # Lightning LoRA strength
        lightning_lora = self._find_node(wf, "LoraLoaderModelOnly")
        if lightning_lora:
            lightning_lora["inputs"]["strength_model"] = lightning_strength

        # KSampler
        sampler = self._find_node(wf, "KSampler")
        if sampler:
            sampler["inputs"]["seed"] = seed
            sampler["inputs"]["steps"] = steps

        # Optional custom LoRA (insert between Lightning LoRA and ModelSamplingAuraFlow)
        if lora_name:
            lightning_lora = self._find_node(wf, "LoraLoaderModelOnly")
            aura_flow = self._find_node(wf, "ModelSamplingAuraFlow")
            if lightning_lora and aura_flow:
                # Find the Lightning LoRA node ID
                lightning_id = None
                for nid, node in wf.items():
                    if node is lightning_lora:
                        lightning_id = nid
                        break

                # Add custom LoRA node
                custom_lora_id = "900"
                wf[custom_lora_id] = {
                    "class_type": "LoraLoaderModelOnly",
                    "inputs": {
                        "model": [lightning_id, 0],
                        "lora_name": lora_name,
                        "strength_model": lora_strength,
                    },
                }
                # Rewire ModelSamplingAuraFlow to take from custom LoRA
                aura_flow["inputs"]["model"] = [custom_lora_id, 0]

        # SaveImage prefix
        save = self._find_node(wf, "SaveImage")
        if save:
            save["inputs"]["filename_prefix"] = "curation_tool"

        return wf

    def build_flux2_workflow(
        self,
        prompt: str,
        reference_image: str = "reference.png",
        seed: int = 0,
        steps: int = 28,
        cfg: float = 3.0,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        pulid_weight: float = 0.8,
        pulid_end_at: float = 0.8,
        face_restore_fidelity: float = 0.8,
    ) -> dict:
        """Build a Flux2 Dev PuLID txt2img workflow with face identity preservation.

        Uses PuLID for facial identity, CR Prompt List for dataset captioning,
        and SaveImageKJ for caption file output.
        """
        wf = self.get_template("flux2_base")

        # LoadImage (face reference)
        load_img = self._find_node(wf, "LoadImage")
        if load_img:
            load_img["inputs"]["image"] = reference_image

        # CR Prompt List (dataset captioning)
        prompt_list = self._find_node(wf, "CR Prompt List")
        if prompt_list:
            prompt_list["inputs"]["body_text"] = prompt

        # Positive prompt
        clip_nodes = self._find_nodes(wf, "CLIPTextEncode")
        if clip_nodes:
            clip_nodes[0]["inputs"]["text"] = prompt
        if len(clip_nodes) > 1:
            clip_nodes[1]["inputs"]["text"] = negative_prompt

        # FluxGuidance
        guidance = self._find_node(wf, "FluxGuidance")
        if guidance:
            guidance["inputs"]["guidance"] = cfg

        # KSampler
        sampler = self._find_node(wf, "KSampler")
        if sampler:
            sampler["inputs"]["seed"] = seed
            sampler["inputs"]["steps"] = steps

        # EmptyLatentImage
        latent = self._find_node(wf, "EmptyLatentImage")
        if latent:
            latent["inputs"]["width"] = width
            latent["inputs"]["height"] = height

        # ApplyPulidFlux
        pulid = self._find_node(wf, "ApplyPulidFlux")
        if pulid:
            pulid["inputs"]["weight"] = pulid_weight
            pulid["inputs"]["end_at"] = pulid_end_at

        # FaceRestoreCFWithModel
        face_restore = self._find_node(wf, "FaceRestoreCFWithModel")
        if face_restore:
            face_restore["inputs"]["codeformer_fidelity"] = face_restore_fidelity

        # SaveImageKJ prefix
        save = self._find_node(wf, "SaveImageKJ")
        if save:
            save["inputs"]["filename_prefix"] = "curation_tool"

        return wf

    def build_face_restore_workflow(
        self,
        source_image: str,
        model_name: str = "codeformer.pth",
        fidelity: float = 0.8,
    ) -> dict:
        """Build a face restoration workflow."""
        wf = self.get_template("face_restore")

        load_img = self._find_node(wf, "LoadImage")
        if load_img:
            load_img["inputs"]["image"] = source_image

        loader = self._find_node(wf, "FaceRestoreModelLoader")
        if loader:
            loader["inputs"]["model_name"] = model_name

        restore = self._find_node(wf, "FaceRestoreCFWithModel")
        if restore:
            restore["inputs"]["codeformer_fidelity"] = fidelity

        return wf
