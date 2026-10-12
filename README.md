# ComfyUI CharacterPose

Custom nodes for **identity-preserving 2D character pose transfer**: take a single sprite and generate the same character in new RPG poses while keeping art style and appearance.

> **Main path:** **Flux.2 Klein** + `CP_PoseTransferPrep` (auto caption + library `.pose` guide) — see `workflows/pose_transfer_flux_klein.json`.  
> Geometric warp (TPS / piecewise) melts cartoon sprites — do not use it as the main pipeline.

## Features

- **`CP_PoseTransferPrep`** — one node: auto caption, load any `.pose` from `poses/`, props hint → guide + edit prompt
- **`CP_PoseComposer3D`** — optional kinematic skeleton (8 cameras + actions / text LLM) if you need a guide without a file
- **Props** as text hints on Prep (`sword`, `shield`, …); 3D prop drawing stays on Composer 3D
- **`CP_CharacterCaption`** — Florence-2 img2txt + **palette hex color lock** (heuristic fallback) + cache `.char`
- `.pose` library — build with [`mixamo_tools/`](mixamo_tools/) (Mixamo FBX → COCO-18), load via Prep or `CP_PoseLibraryLoad`
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
| ComfyUI core Flux.2 / Klein nodes | Flux Klein workflow |
| [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) | Optional SDXL `CharacterRepair` path |
| `transformers` + Florence-2 weights | **Recommended** — detailed auto-captions for color/style lock (`CP_CharacterCaption` / Prep) |

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
| Pose transfer (~4 steps) | `pose_transfer_flux_klein.json` | Prep filters → Klein batch; soft pose guidance |
| Mixamo `.pose` library | Wire `CP_PoseLibraryLoad` (+ optional warp) | Export via `mixamo_tools` CLI into `poses/` |

## Recommended workflow (Flux.2 Klein + Prep)

Open: `workflows/pose_transfer_flux_klein.json`

```
LoadImage (sprite)
  → CP_MixamoPoseExport (optional) → pose_list
  → CP_PoseTransferPrep (filters ALL or one value each)
       → guide [N], image_batch [N], edit_prompts [N]
  → CP_KleinPoseBatch (sequential: prompt[i] + guide[i] per pose)
       → preview/save prefix_000, prefix_001, … as each finishes
  → SaveImage (full batch at the end)
```

### Models for Flux Klein path

| File | Folder |
|------|--------|
| `flux-2-klein-4b-fp8.safetensors` | `models/diffusion_models/` |
| `qwen_3_4b.safetensors` | `models/text_encoders/` |
| `flux2-vae.safetensors` | `models/vae/` |

**Tips**

- Distilled Klein: ~4 steps, CFG `1`.
- Prep **`align_to_source`** (default on) scales the library stick to the sprite, then **contains** it in the canvas (shrink + recenter with margin) so open Mixamo poses do not clip.
- Prep selects poses **only** via filters (`animation` / `orientation` / `frame`, each `ALL` or one value) → **guide** + **image_batch** + **edit_prompts** (JSON, one prompt per pose with VIEW LOCK). `CP_KleinPoseBatch` runs **one sample at a time** (same quality as N=1) and writes progressive previews (`prefix_000`…). Narrow filters for a single pose. **Restart ComfyUI** after updating custom nodes before reloading `pose_transfer_flux_klein.json`.
- Prep auto-prompt adds a **VIEW LOCK** from the `.pose` (`metadata.view` or filename `_NE` / `_TOP`…) so back views (NE/NO/N/TOP) are stated explicitly, with APPEARANCE LOCK from-behind — still soft on Klein.
- If bones appear: the Prep prompt already includes anti-skeleton wording; change seed.
- Klein has **no OpenPose ControlNet** — pose guidance is soft (reference latent + prompt).

## Pose library (Mixamo → `.pose`)

### In ComfyUI — `CP_MixamoPoseExport`

1. Drop Mixamo `.fbx` into `ComfyUI/input/` or `input/mixamo/`.
2. Install **`pufbx`** in the ComfyUI Python env (`pip install pufbx`, see `mixamo_tools/requirements.txt`).
3. Run **Mixamo Pose Export**: orientations `4_iso` | `4_cardinal` | `6` | `8`, `frames_per_view` = samples per angle → writes `{anim}_{VIEW}_{ii}.pose` into package `poses/`.
4. Prep: choose `animation` / `orientation` / `frame` (`ALL` = every match). Optionally wire export `pose_list` so the filters apply to that export only; otherwise Prep scans `poses/` on disk → **guide batch**.

### CLI (optional)

```powershell
cd mixamo_tools
.\venv\Scripts\python.exe -m export_views --positions 4 --view 8 --out-dir ..\poses
```

| Orientation | Meaning |
|-------------|---------|
| `*_SE` / `*_SO` | South-East / South-West — ¾ front |
| `*_NE` / `*_NO` | North-East / North-West — ¾ back |
| `*_TOP` / `*_BOT` | Full back / full front (ISO map) |
| `*_LEFT` / `*_RIGHT` | Profiles |

### `source_pose` preview (Prep)

Wire **DWPreprocessor** → Prep:
- `POSE_KEYPOINT` → `pose_keypoint` (align + stick overlay)
- `IMAGE` → `dwpose_image` (pixel-perfect sticks from DWPose)

Without DWPose, Prep shows the **sprite alone** — no fake T-pose.

## Pose Composer 3D (optional)

Node: `CP_PoseComposer3D` — separate from Prep. Use when you want a guide **without** a Mixamo `.pose`.

- **`pose_source`:** `library` (named action preset) or `text` (LLM → joint angles)
- **Camera presets:** `S`, `SE`, `E`, `NE`, `N`, `NW`, `W`, `SW`
- **Props:** drawn in cyan on the guide
- Outputs: `POSE`, `guide`, `prop_mask`, `prop_hint`

## Nodes (CharacterPose)

| Node | Role |
|------|------|
| `CP_MixamoPoseExport` | Mixamo FBX → `.pose` library + `pose_list` (needs `pufbx`) |
| `CP_PoseTransferPrep` | Filters + caption + align → `guide` batch, `edit_prompt` / `edit_prompts`, … |
| `CP_KleinPoseBatch` | Sequential Klein sample per Prep item (per-prompt + progressive preview) |
| `CP_PoseComposer3D` | Optional 3D action/camera/props → POSE + guide |
| `CP_CharacterCaption` | Florence-2 + palette hex → caption / edit_prompt / backend |
| `CP_BuildEditPrompt` | Assemble caption + palette + props + anti-bones |
| `CP_ExtractPose` | Image (+ DWPose keypoint) → `POSE` |
| `CP_PoseLibraryLoad` | Load built-in / custom `.pose` |
| `CP_ApplyPose` | Draw OpenPose skeleton image |
| `CP_WarpToPose` | Align target pose; optional geometric warp (**default off**) |
| `CP_CharacterEncode` / Save / Load | `.char` pack (optional caption fields) |
| `CP_CharacterRepair` | SDXL img2img + IP-Adapter + optional ControlNet |
| `CP_ExportSpriteSheet` / `CP_GenerateRPGSheet` | Sheet helpers |

## Workflows

| File | Notes |
|------|--------|
| `pose_transfer_flux_klein.json` | Main path (Prep → Klein batch) — regenerate via `python workflows/_build_workflows.py` |

## Limitations

- **Same-view bias:** a side/¾ sprite does not become a true front/back view for free; generative models approximate it.
- Props are **guide strokes + prompt hints**, not a separate ControlNet channel (unless you feed `prop_mask` / canny yourself).
- Florence-2 is **recommended** for detailed appearance; without `transformers`, Prep still injects a K-means **COLOR LOCK** palette into the prompt.
- Geometric warp cannot invent limbs or camera angles; it only remaps pixels.
- Klein pose lock is **soft** (no OpenPose ControlNet on this path).

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
- Captioning via Microsoft **Florence-2** (optional)
