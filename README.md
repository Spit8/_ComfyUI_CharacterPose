# ComfyUI CharacterPose

Custom nodes for **identity-preserving 2D character pose transfer**: take a single sprite and generate the same character in new RPG poses while keeping art style and appearance.

> **Main path:** **Flux.2 Klein** + `CP_PoseTransferPrep` (auto caption + library `.pose` guide) — see `workflows/pose_transfer_flux_klein.json`.

## Features

- **`CP_PoseTransferPrep`** — filters on `poses/`, Florence-2 caption + palette COLOR LOCK → guide batch + edit prompts
- **`CP_KleinPoseBatch`** — sequential Klein sample per pose (progressive previews)
- **`CP_MixamoPoseExport`** — Mixamo FBX → `.pose` library (needs `pufbx`)
- **`CP_PoseLibraryLoad`** / **`CP_ApplyPose`** — load a `.pose` and draw the OpenPose guide
- CLI tools under [`mixamo_tools/`](mixamo_tools/)

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
| `transformers` + Florence-2 weights | **Recommended** — detailed auto-captions for Prep |
| `pufbx` | Mixamo FBX import (`mixamo_tools/requirements.txt`) |

### DWPose ONNX

Place DWPose weights where `comfyui_controlnet_aux` expects them (not under `models/`):

```
custom_nodes/comfyui_controlnet_aux/ckpts/yzd-v/DWPose/
  yolox_l.onnx
  dw-ll_ucoco_384.onnx
```

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
- Prep **`align_to_source`** (default on) scales the library stick to the sprite, then **contains** it in the canvas so open Mixamo poses do not clip.
- Prep selects poses via filters (`animation` / `orientation` / `frame`, each `ALL` or one value). Narrow filters for a single pose.
- Use the free **`extra`** string on Prep for optional prompt hints (e.g. “holding a sword”).
- Klein has **no OpenPose ControlNet** — pose guidance is soft (reference latent + prompt).

## Pose library (Mixamo → `.pose`)

### In ComfyUI — `CP_MixamoPoseExport`

1. Drop Mixamo `.fbx` into `ComfyUI/input/` or `input/mixamo/`.
2. Install **`pufbx`** in the ComfyUI Python env (`pip install pufbx`).
3. Run **Mixamo Pose Export** → writes `{anim}_{VIEW}_{ii}.pose` into package `poses/`.
4. Prep: choose filters (`ALL` = every match), or wire export `pose_list`.

### CLI (optional)

```powershell
cd mixamo_tools
.\venv\Scripts\python.exe -m export_views --positions 4 --view 8 --out-dir ..\poses
```

### `source_pose` preview (Prep)

Wire **DWPreprocessor** → Prep (`pose_keypoint` + `dwpose_image`). Without DWPose, Prep shows the sprite alone — no fake T-pose.

## Nodes

| Node | Role |
|------|------|
| `CP_MixamoPoseExport` | Mixamo FBX → `.pose` library + `pose_list` |
| `CP_PoseTransferPrep` | Filters + caption + align → guide batch + prompts |
| `CP_KleinPoseBatch` | Sequential Klein sample per Prep item |
| `CP_PoseLibraryLoad` | Load a `.pose` file |
| `CP_ApplyPose` | Draw OpenPose skeleton image |

## Workflows

| File | Notes |
|------|--------|
| `pose_transfer_flux_klein.json` | Main path — regenerate via `python workflows/_build_workflows.py` |

## Limitations

- **Same-view bias:** a side/¾ sprite does not become a true front/back view for free.
- Florence-2 is **recommended**; without `transformers`, Prep still injects a K-means **COLOR LOCK** palette.
- Klein pose lock is **soft** (no OpenPose ControlNet on this path).

## Dev / tests

```bash
python tests/smoke_test.py
```

## License

MIT — see [LICENSE](LICENSE).

## Credits

- Pose detection via [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) / DWPose
- Generative edit via Black Forest Labs **FLUX.2 Klein** in ComfyUI
- Captioning via Microsoft **Florence-2** (optional)
