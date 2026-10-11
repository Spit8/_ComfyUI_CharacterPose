# mixamo_tools

Standalone tooling: **Mixamo FBX → 2D OpenPose skeleton views** at isometric camera angles.

Not a ComfyUI custom node (yet). Can be split into its own library later.

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

Recommended (local venv):

```powershell
.\venv\Scripts\python.exe -m export_views --positions 4 --view SE
```

From the **repo root** (same Python that has the deps):

```bash
python -m mixamo_tools.export_views --positions 4 --view SE
# or:  python -m mixamo_tools --positions 4 --view SE
```

From **inside** `mixamo_tools/`:

```bash
python -m export_views --positions 4 --view SE
# or:  python export_views.py --positions 4 --view SE
```

→ `mixamo_tools/out/slow_run_SE_00.png` … `slow_run_SE_03.png`  
(4 evenly spaced frames, same camera framing)

```bash
python -m mixamo_tools.export_views --positions 4
```

→ 16 PNGs (4 views × 4 positions)

```bash
python -m mixamo_tools.export_views --frame 11 --view NE
```

→ single frame override (`--positions` ignored)

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--fbx` | `mixamo/Slow Run.fbx` | Source FBX |
| `--positions N` | `4` | Evenly spaced samples over the clip |
| `--view` | `ALL` | `SE` / `SO` / `NE` / `NO` / `ALL` |
| `--frame` | — | Single frame index (overrides `--positions`) |
| `--out-dir` | `mixamo_tools/out` | Output folder |
| `--size` | `1024` | Square canvas |
| `--margin` | `0.10` | Padding when auto-fitting zoom |
| `--stem` | from FBX name | Filename prefix |

## Pipeline

1. Load FBX with `pufbx` (`anim_to_array` local TRS)
2. Sample N frames at equal time steps
3. Map Mixamo bones → OpenPose COCO-18 (normalized)
4. Fit one shared camera distance so every sample fits on canvas
5. Project each (view × position) and draw stick-figure PNGs
