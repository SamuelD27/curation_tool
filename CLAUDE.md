# Curation Tool - Project Instructions

## Workflows

All ComfyUI workflow JSON files **must** be stored in the ComfyUI user workflows directory:

```
/home/samsam/ComfyUI/user/default/workflows/
```

- **New workflows:** Always create in the path above
- **Updates:** Edit existing files in the path above
- **Never** place workflow JSON files in the project repo itself
- **Format:** Always use the **litegraph/GUI format** (with `nodes`, `links`, `groups`, `config`, `extra`, `version` keys). Never use the API/prompt format (flat `{id: {class_type, inputs}}`). The GUI format is required for ComfyUI to render the workflow in the web UI.

### Current Workflows
| File | Purpose |
|------|---------|
| `flux_base.json` | Flux 1 base generation |
| `flux2_base.json` | Flux 2 base generation (flux2-dev + Mistral CLIP + PuLID + CodeFormer) |
| `pulid_identity.json` | PuLID identity preservation |
| `caption_pipeline.json` | Image captioning |
| `upscale_resize.json` | Upscaling and resizing |
| `face_restore.json` | Face restoration |
| `qwen_face_edit.json` | Qwen-guided face editing |
| `flux2_dataset_gen.json` | Flux 2 dataset generation (PuLID-Flux ll + Comfyroll prompt list + caption export) |

### flux2_base.json Node Map
Key node IDs for programmatic access:
- **4** UNETLoader: `flux2-dev.safetensors`
- **10** CLIPLoader: `mistral_3_small_flux2_fp8.safetensors` (type: flux2)
- **11** VAELoader: `flux2-ae.safetensors`
- **100** LoadImage: reference input
- **5** EmptyLatentImage: 1024x1024
- **6/7** CLIPTextEncode: positive/negative prompts
- **20** FluxGuidance: guidance=3.0
- **12** PulidFluxModelLoader: `pulid_flux_v0.9.1.safetensors`
- **13** PulidFluxInsightFaceLoader: CUDA
- **19** PulidFluxEvaClipLoader
- **21** RescaleCFG: multiplier=0.7
- **25** PerturbedAttention: scale=1.5, middle block
- **14** ApplyPulidFlux: weight=0.8, end_at=0.8
- **3** KSampler: euler/beta, 28 steps, cfg=1.0
- **8** VAEDecode
- **16** FaceRestoreModelLoader: `codeformer.pth`
- **15** FaceRestoreCFWithModel: fidelity=0.8
- **200** CR Prompt List: caption text
- **9** SaveImageKJ: output with caption
