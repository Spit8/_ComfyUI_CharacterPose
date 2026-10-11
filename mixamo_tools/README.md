# mixamo_tools

Standalone tooling: **Mixamo FBX → OpenPose COCO-18 `.pose` + debug PNG** at isometric camera angles.

Not a ComfyUI custom node (yet). Drop / rename the `.pose` files into the package [`poses/`](../poses/) folder for `CP_PoseLibraryLoad`.

## Camera labels (French ISO)

| Label | Meaning | Orbit (yaw / pitch) |
|-------|---------|---------------------|
| **SE** | Sud-Est | −45° / −15° |
| **SO** | Sud-Ouest | +45° / −15° |
| **NE** | Nord-Est | −135° / −15° |
| **NO** | Nord-Ouest | +135° / −15° |

Pan / zoom / tilt (`look_at` + `distance`) are **fixed for the whole export**. Only the orbit yaw changes between SE/SO/NE/NO.

## Install

Prefer a **local venv** (avoids “wrong Python” issues on Windows):

```powershell
cd mixamo_tools
.\setup_env.ps1
# then always:
.\venv\Scripts\python.exe -m export_views --positions 16 --view SE
```

Or install into the interpreter you actually use (check with `python -c "import sys; print(sys.executable)"`):

```powershell
python -m pip install -r requirements.txt
```

Deps: `numpy`, `Pillow`, `pufbx`.

## Usage

**By default**, `--fbx` points at the repo `mixamo/` **folder** and exports **every** `*.fbx` / `*.FBX` in it (non-recursive). Each anim gets its own stem from the filename.

Recommended (local venv) — write into the Comfy pose library folder:

```powershell
.\venv\Scripts\python.exe -m export_views --positions 4 --view SE --out-dir ..\poses
# → all FBX under mixamo/ → poses/{anim}_{VIEW}_{ii}.pose (+ PNG unless --no-png)
```

One file only:

```powershell
.\venv\Scripts\python.exe -m export_views --fbx ..\mixamo\"Slow Run.fbx" --positions 4 --view SE --out-dir ..\poses
```

`.pose` only (no PNG):

```powershell
.\venv\Scripts\python.exe -m export_views --positions 4 --view SE --no-png --out-dir ..\poses
```

PNG only (debug):

```powershell
.\venv\Scripts\python.exe -m export_views --positions 4 --view SE --no-pose
```

From the **repo root**:

```bash
python -m mixamo_tools.export_views --positions 4 --view SE
# or:  python -m mixamo_tools --positions 4 --view SE
```

From **inside** `mixamo_tools/`:

```bash
python -m export_views --positions 4 --view SE
```

A failed FBX is logged and skipped; exit code is non-zero if any failed or none found.

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--fbx` | `mixamo/` (dir) | One `.fbx` **or** a directory of FBX |
| `--positions N` | `4` | Evenly spaced samples over the clip |
| `--view` | `ALL` | `SE` / `SO` / `NE` / `NO` / `ALL` |
| `--frame` | — | Single frame index (overrides `--positions`) |
| `--out-dir` | `mixamo_tools/out` | Output folder (use `../poses` for the library) |
| `--size` | `1024` | Square canvas |
| `--margin` | `0.10` | Padding when auto-fitting zoom |
| `--stem` | from FBX name | Filename prefix (**single-file only**; ignored in batch) |
| `--no-png` | off | Skip debug PNG |
| `--no-pose` | off | Skip `.pose` JSON |

## Pipeline

1. Discover FBX file(s) under `--fbx`
2. Load each with `pufbx` (`anim_to_array` local TRS)
3. Sample N frames at equal time steps
4. Map Mixamo bones → OpenPose COCO-18 (normalized)
5. Fit one shared camera distance so every sample fits on canvas
6. Project each (view × position) → write `.pose` (+ optional stick PNG)
