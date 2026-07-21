# ComfyUI CharacterPose

Custom nodes for **identity-preserving 2D character pose transfer**: take a single sprite and generate the same character in new RPG poses (walk, run, idle, jump, fight, work) while keeping art style and appearance.

> **Recommended path:** generative edit with **Flux.2 Klein** (dual reference: sprite + OpenPose guide).  
> Geometric warp (TPS / piecewise) is included for experiments but **melts cartoon sprites** — do not use it as the main pipeline.

## Features

- Built-in **pose library** for isometric-style sheets: **South-East (`_se`)** and **North-East (`_ne`)**
  - Actions: `idle`, `walk` (4 frames), `run` (4), `jump`, `fight` (2), `work` (2)
- Nodes to extract DWPose → `.pose`, align target skeletons onto the detected body, and feed Flux / SDXL workflows
- Optional character pack (`.char`) + SDXL repair path (IP-Adapter / ControlNet) for older experiments
- Example workflows under `workflows/`

## Install

1. Clone into your ComfyUI custom nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USER/ComfyUI_CharacterPose.git
```

2. Install Python deps (ComfyUI’s venv recommended):

```bash
pip install -r ComfyUI_CharacterPose/requirements.txt
```

3. Restart ComfyUI.

### Soft dependencies (Manager)

| Pack | Needed for |
|------|------------|
| [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) | `DWPreprocessor` → pose extract |
| ComfyUI core Flux.2 / Klein nodes | Recommended Flux workflow (built-in on recent ComfyUI) |
| [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) | Optional SDXL `CharacterRepair` path |

### DWPose ONNX

Place DWPose weights where `comfyui_controlnet_aux` expects them (not under `models/`):

```
custom_nodes/comfyui_controlnet_aux/ckpts/yzd-v/DWPose/
  yolox_l.onnx
  dw-ll_ucoco_384.onnx
```

The ComfyUI “Missing Models” UI may still warn about these — that warning is often a **false positive** for DWPose.

## Recommended workflow (Flux.2 Klein)

Open: `workflows/pose_transfer_flux_klein.json`

```
LoadImage (sprite)
  → DWPreprocessor → CP_ExtractPose
CP_PoseLibraryLoad (e.g. idle_se / walk_se_01 / fight_ne_02)
  → CP_WarpToPose (method=none)  → aligned OpenPose guide
Flux.2 Klein dual ReferenceLatent:
  ref1 = sprite (identity + style)
  ref2 = OpenPose guide (pose only)
  prompt = keep character, follow pose, never draw bones
→ SaveImage
```

### Models for Flux Klein path

| File | Folder |
|------|--------|
| `flux-2-klein-4b-fp8.safetensors` | `models/diffusion_models/` |
| `qwen_3_4b.safetensors` | `models/text_encoders/` |
| `flux2-vae.safetensors` | `models/vae/` |

Download links are embedded in the workflow (ComfyUI Missing Models / Download).

**Tips**

- Distilled Klein: ~4 steps, CFG `1`.
- If **bones / skeleton** appear on the character: strengthen the prompt (“pose guide only, never draw bones / x-ray”) and change seed.
- If pose is weak: try other frames (`walk_se_02`, `run_se_01`, …). Klein has **no OpenPose ControlNet** — pose is guided by the second reference + prompt, not locked.
- For **hard pose lock**, consider Qwen-Image-Edit + ControlNet keypoints (separate stack).

## Pose library

Regenerate all `.pose` files:

```bash
python poses/_generate_poses.py
```

| Orientation | Meaning |
|-------------|---------|
| `*_se` | South-East — ¾ front-right (toward camera) |
| `*_ne` | North-East — ¾ back-right (back more visible) |

| Action | Examples |
|--------|----------|
| idle | `idle_se.pose`, `idle_ne.pose` |
| walk | `walk_se_01.pose` … `walk_se_04.pose` |
| run | `run_se_01.pose` … |
| jump | `jump_se.pose` |
| fight | `fight_se_01.pose`, `fight_se_02.pose` |
| work | `work_se_01.pose`, `work_se_02.pose` |

Aliases: `idle_side` → `idle_se`, `walk_side_*` → `walk_se_*`.

These skeletons are **procedural**, not motion-captured. Expect to trial several frames per action.

## Nodes (CharacterPose)

| Node | Role |
|------|------|
| `CP_ExtractPose` | Image (+ DWPose keypoint) → `POSE` |
| `CP_PoseLibraryLoad` | Load built-in / custom `.pose` |
| `CP_ApplyPose` | Draw OpenPose skeleton image |
| `CP_SavePose` / load helpers | Persist `.pose` |
| `CP_WarpToPose` | Align target pose; optional geometric warp (**default off**); skeleton blend |
| `CP_BlendSkeleton` | Overlay sticks on sprite (preview / guidance) |
| `CP_CharacterEncode` / Save / Load | `.char` pack (reference + optional embedding) |
| `CP_CharacterRepair` | SDXL img2img + IP-Adapter + optional ControlNet |
| `CP_ExportSpriteSheet` / `CP_GenerateRPGSheet` | Sheet helpers |

## Other workflows

| File | Notes |
|------|--------|
| `pose_transfer_flux_klein.json` | **Preferred** generative path |
| `pose_transfer_warp.json` | Legacy SDXL warp + repair — warp melts sprites; keep `method=none` |
| `poc_pose_transfer.json` / `character_pose_nodes.json` | Earlier experiments |

## Limitations

- **Same-view bias:** a side/¾ sprite does not become a true front/back view for free; generative models approximate it.
- **No LoRA required** for the Klein path (single reference image). A character LoRA can still help style lock later if needed.
- **Do not train a LoRA on every run** — too slow; cache per character if you add training.
- Geometric warp cannot invent limbs or camera angles; it only remaps pixels.

## Dev / tests

```bash
# From this package root, with ComfyUI venv python:
python tests/smoke_test.py
```

## License

MIT — see [LICENSE](LICENSE).

## Credits

- Pose detection via [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) / DWPose
- Generative edit via Black Forest Labs **FLUX.2 Klein** in ComfyUI
