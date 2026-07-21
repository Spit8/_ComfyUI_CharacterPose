# ComfyUI CharacterPose

Custom nodes for **identity-preserving 2D character pose transfer**: take a single sprite and generate the same character in new RPG poses while keeping art style and appearance.

> **Fast path:** **Flux.2 Klein** + `CP_PoseTransferPrep` (auto caption + 3D camera/props guide).  
> **Hard pose lock:** **Qwen-Image-Edit** + ControlNet Union (`workflows/pose_transfer_qwen_controlnet.json`).  
> Geometric warp (TPS / piecewise) melts cartoon sprites — do not use it as the main pipeline.

## Features

- **`CP_PoseTransferPrep`** — one node: auto caption, 3D pose, camera, props → guide + edit prompt
- **`CP_PoseComposer3D`** — kinematic skeleton with **8 camera presets** (S/SE/E/NE/N/NW/W/SW) + yaw/pitch/roll
- **Props / mounts** on the guide (not in COCO-18): `sword`, `shield`, `staff`, `bow`, `horse`
- **`CP_CharacterCaption`** — Florence-2 img2txt (heuristic fallback) + cached caption in `.char`
- Legacy `.pose` library (SE/NE procedural) still available via `CP_PoseLibraryLoad`
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
| [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) | Optional DWPose extract / align |
| ComfyUI core Flux.2 / Klein nodes | Fast Flux workflow |
| ComfyUI Qwen-Image-Edit nodes | Hard-lock Qwen workflow |
| [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) | Optional SDXL `CharacterRepair` path |
| `transformers` + Florence-2 weights | Better auto-captions (`CP_CharacterCaption`) |

### DWPose ONNX

Place DWPose weights where `comfyui_controlnet_aux` expects them (not under `models/`):

```
custom_nodes/comfyui_controlnet_aux/ckpts/yzd-v/DWPose/
  yolox_l.onnx
  dw-ll_ucoco_384.onnx
```

## Which workflow?

| Goal | Workflow | Notes |
|------|----------|--------|
| Fast iteration (~4 steps) | `pose_transfer_flux_klein.json` | Dual reference; soft pose guidance |
| Strict pose / multi-angle | `pose_transfer_qwen_controlnet.json` | ControlNet keypoints strength ~1.2–1.8 |
| Legacy 2D `.pose` files | Wire `CP_PoseLibraryLoad` + `CP_WarpToPose` manually | Still supported |

## Recommended workflow (Flux.2 Klein + Prep)

Open: `workflows/pose_transfer_flux_klein.json`

```
LoadImage (sprite)
  → CP_PoseTransferPrep (action, camera SE/…, prop)
       → guide IMAGE + edit_prompt STRING
Flux.2 Klein dual ReferenceLatent:
  ref1 = sprite | ref2 = guide
  prompt = edit_prompt (auto)
→ SaveImage
```

### Models for Flux Klein path

| File | Folder |
|------|--------|
| `flux-2-klein-4b-fp8.safetensors` | `models/diffusion_models/` |
| `qwen_3_4b.safetensors` | `models/text_encoders/` |
| `flux2-vae.safetensors` | `models/vae/` |

**Tips**

- Distilled Klein: ~4 steps, CFG `1`.
- If bones appear: the Prep prompt already includes anti-skeleton wording; change seed.
- Klein has **no OpenPose ControlNet** — for a hard lock use the Qwen workflow.

## Hard pose lock (Qwen-Image-Edit + ControlNet)

Open: `workflows/pose_transfer_qwen_controlnet.json`

```
LoadImage → CP_PoseTransferPrep → guide + prompt
Qwen-Image-Edit + ControlNet Union (InstantX) on the OpenPose guide
→ SaveImage
```

### Models for Qwen path

| File | Folder |
|------|--------|
| `qwen_image_edit_2511_fp8_e4m3fn.safetensors` (or 2509) | `models/diffusion_models/` |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `models/text_encoders/` |
| `qwen_image_vae.safetensors` | `models/vae/` |
| `Qwen-Image-InstantX-ControlNet-Union.safetensors` | `models/controlnet/` |

If `TextEncodeQwenImageEdit` is missing on your ComfyUI build, use the native Qwen Edit text-encode node and convert its prompt widget to an input for the Prep `edit_prompt` STRING.

Alternative: DiffSynth Union LoRA `qwen_image_union_diffsynth_lora.safetensors` (openpose mode) instead of InstantX ControlNet.

## Pose Composer 3D

Node: `CP_PoseComposer3D`

- **Actions:** `idle`, `walk_01..04`, `run_01..04`, `jump`, `fight_01/02`, `work_01/02`, `cast`, `ride_idle`
- **Camera presets:** `S`, `SE`, `E`, `NE`, `N`, `NW`, `W`, `SW` — or enable `use_manual_camera` for absolute yaw/pitch/roll
- **Props:** drawn in cyan on the guide (COCO-18 body unchanged for ControlNet compatibility)
- Outputs: `POSE`, full `guide`, `prop_mask`, `prop_hint` (for prompts)

Legacy 2D library (still useful):

```bash
python poses/_generate_poses.py
```

| Orientation | Meaning |
|-------------|---------|
| `*_se` | South-East — ¾ front-right |
| `*_ne` | North-East — ¾ back-right |

## Nodes (CharacterPose)

| Node | Role |
|------|------|
| `CP_PoseTransferPrep` | Caption + 3D compose + align → guide + prompt |
| `CP_PoseComposer3D` | 3D action/camera/props → POSE + guide |
| `CP_CharacterCaption` | Florence-2 / fallback caption + edit prompt (+ `.char` cache) |
| `CP_BuildEditPrompt` | Assemble caption + prop hints + anti-bones |
| `CP_ExtractPose` | Image (+ DWPose keypoint) → `POSE` |
| `CP_PoseLibraryLoad` | Load built-in / custom `.pose` |
| `CP_ApplyPose` | Draw OpenPose skeleton image |
| `CP_WarpToPose` | Align target pose; optional geometric warp (**default off**) |
| `CP_CharacterEncode` / Save / Load | `.char` pack (optional caption fields) |
| `CP_CharacterRepair` | SDXL img2img + IP-Adapter + optional ControlNet |
| `CP_ExportSpriteSheet` / `CP_GenerateRPGSheet` | Sheet helpers |

## Other workflows

| File | Notes |
|------|--------|
| `pose_transfer_flux_klein.json` | Fast path (Prep) |
| `pose_transfer_qwen_controlnet.json` | Hard pose lock |
| `pose_transfer_warp.json` | Legacy SDXL — keep `method=none` |
| `poc_pose_transfer.json` / `character_pose_nodes.json` | Earlier experiments |

## Limitations

- **Same-view bias:** a side/¾ sprite does not become a true front/back view for free; generative models approximate it (Qwen-Edit 2511 helps).
- Props are **guide strokes + prompt hints**, not a separate ControlNet channel (unless you feed `prop_mask` / canny yourself).
- Florence-2 is optional; without it, Prep uses a palette heuristic caption.
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
- Generative edit via Black Forest Labs **FLUX.2 Klein** and **Qwen-Image-Edit** in ComfyUI
- Captioning via Microsoft **Florence-2** (optional)
