# Curation Tool - Command Reference

**Binary:** `curate`

## Global Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--verbose` | `-v` | off | Enable debug logging |
| `--comfyui-url` | | `http://127.0.0.1:8188` | ComfyUI server URL (env: `COMFYUI_URL`) |
| `--log-file` | | none | Log file path |

---

## `curate health`

Check ComfyUI connectivity and print system stats (GPU VRAM, Python version, OS).

```bash
curate health
curate --comfyui-url http://192.168.1.10:8188 health
```

---

## `curate edit`

Generate a single image with a text prompt, optionally identity-preserving.

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--image` | `-i` | **required** | Input image path |
| `--prompt` | `-p` | **required** | Scene description prompt |
| `--output` | `-o` | `output.png` | Output image path |
| `--seed` | | `0` | Random seed |
| `--steps` | | `30` | Inference steps |
| `--template` | `-t` | `flux_base` | Workflow template (`flux_base`, `pulid_identity`, or `qwen_face_edit`) |

```bash
curate edit -i photo.png -p "Professional headshot, studio lighting" -o result.png
curate edit -i photo.png -p "Portrait in golden hour" --seed 42 --steps 50 -t pulid_identity -o out.png
```

---

## `curate run`

Run a batch curation job from a YAML config file.

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--config` | `-c` | **required** | Path to YAML job config |
| `--export-lora` | | off | Export as LoRA training dataset (image + caption pairs) |
| `--caption-template` | | `{prompt}` | Caption template. Placeholders: `{prompt}`, `{source}`, `{seed}` |

```bash
curate run -c job.yaml
curate run -c job.yaml --export-lora --caption-template "sks_person, {prompt}"
```

### YAML Config Schema (`CurationJob`)

```yaml
input_dir: ./inputs
output_dir: ./outputs
default_seed: 0
default_num_steps: 30
default_cfg_scale: 3.5

comfyui:
  host: 127.0.0.1
  port: 8188
  timeout: 300.0
  poll_interval: 0.5

workflow:
  template: qwen_face_edit         # flux_base | pulid_identity | qwen_face_edit
  face_restore: true
  face_restore_fidelity: 0.8
  upscale_model: RealESRGAN_x4plus.pth

tasks:
  - source_image: photo.png
    prompt: "Professional portrait, studio lighting"
    seed: 42
    num_steps: 30
    cfg_scale: 3.5
    reference_images: []
```

---

## `curate gui`

Launch the Gradio web interface.

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `0.0.0.0` | Host to bind to |
| `--port` | `7860` | Port to serve on |
| `--share` | off | Create a public Gradio share link |

```bash
curate gui
curate gui --port 8080 --share
```

---

## `curate face`

Multi-stage face dataset generation pipeline. Contains subcommands below.

### `curate face run`

Run the face dataset pipeline from a YAML config.

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--config` | `-c` | **required** | Path to face pipeline YAML config |
| `--pick` | | none | Pre-supply picks as `stage=path` (repeatable) |
| `--no-interactive` | | off | Disable interactive prompts (pause instead) |

```bash
curate face run -c face_config.yaml
curate face run -c face_config.yaml --pick refine=candidate_0003.png
curate face run -c face_config.yaml --no-interactive
```

### `curate face resume`

Resume a paused face pipeline with picks.

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--config` | `-c` | **required** | Path to face pipeline YAML config |
| `--pick` | | none | Supply picks as `stage=path` (repeatable) |

```bash
curate face resume -c face_config.yaml --pick refine=chosen.png
```

### `curate face presets`

List all available prompt presets and their contents.

```bash
curate face presets
```

**Built-in presets:**

| Preset | Count | Description |
|--------|-------|-------------|
| `headshot_20` | 20 | Professional headshots: angles, lighting, expressions, crops |
| `qwen_headshot_20` | 20 | Qwen-style photorealistic headshots: angles, lighting, white wall backgrounds |
| `body_10` | 10 | Full/half body: standing, seated, walking, editorial |

### Face Pipeline YAML Config Schema (`FacePipelineConfig`)

```yaml
trigger_word: sks_person
input_dir: ./input
output_dir: ./output
default_seed: 42
default_num_steps: 30
default_cfg_scale: 3.5

comfyui:
  host: 127.0.0.1
  port: 8188
  timeout: 300.0
  poll_interval: 0.5

workflow:
  template: qwen_face_edit         # or pulid_identity, flux_base
  face_restore: true
  face_restore_fidelity: 0.8
  upscale_model: RealESRGAN_x4plus.pth

stages:
  - name: refine
    type: refine              # refine | angles | body
    num_candidates: 4
    prompt: "Professional headshot, centered, sharp focus, studio lighting"
    seeds: [42, 43, 44, 45]

  - name: headshots
    type: angles
    preset: headshot_20       # use a built-in preset
    # or: prompts: ["prompt1", "prompt2", ...]

  - name: fullbody
    type: body
    preset: body_10
```

**Stage types:**

| Type | Purpose | Key Fields |
|------|---------|------------|
| `refine` | Generate candidates for manual pick | `num_candidates`, `prompt`, `seeds` |
| `angles` | Generate headshot angle/lighting variations | `preset` or `prompts` |
| `body` | Generate full/half body variations | `preset` or `prompts` |

**Pipeline flow:** stages run sequentially. After a `refine` stage, the pipeline pauses for a pick (interactive prompt, `--pick` flag, or a `pick.png` file in the stage directory). The picked image becomes the source for subsequent stages.
