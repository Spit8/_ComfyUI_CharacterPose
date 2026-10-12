"""ComfyUI node: Mixamo FBX → library .pose files."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

try:
    import folder_paths
except ImportError:
    folder_paths = None

from ..utils import np_to_tensor
from .pose import BUILTIN_POSES_DIR, _input_directory

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ORIENTATION_MODES = ("4_iso", "4_cardinal", "6", "8")


def _list_fbx_files() -> list[str]:
    """Relative paths of *.fbx under ComfyUI input/ (and input/mixamo/)."""
    names: list[str] = []
    roots: list[Path] = []
    input_dir = _input_directory()
    if input_dir is not None:
        roots.append(input_dir)
        roots.append(input_dir / "mixamo")
    mixamo_repo = PACKAGE_ROOT / "mixamo"
    if mixamo_repo.is_dir():
        roots.append(mixamo_repo)

    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.fbx")):
            if not p.is_file():
                continue
            rel = str(p)
            if input_dir is not None:
                try:
                    rel = str(p.relative_to(input_dir)).replace("\\", "/")
                except ValueError:
                    pass
            if rel not in seen:
                seen.add(rel)
                names.append(rel)
    return names or ["(no .fbx in input/)"]


def _resolve_fbx(fbx: str) -> Path:
    raw = (fbx or "").strip()
    if not raw or raw.startswith("(no"):
        raise FileNotFoundError(
            "No Mixamo FBX selected. Drop .fbx files into ComfyUI/input/ or input/mixamo/."
        )
    candidates = [Path(raw)]
    input_dir = _input_directory()
    if input_dir is not None:
        candidates.insert(0, input_dir / raw)
        candidates.append(input_dir / "mixamo" / Path(raw).name)
    candidates.append(PACKAGE_ROOT / "mixamo" / Path(raw).name)
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"FBX not found: {raw}")


def _resolve_out_dir(out_dir: str) -> Path:
    text = (out_dir or "").strip()
    if not text or text in ("poses", "poses/"):
        BUILTIN_POSES_DIR.mkdir(parents=True, exist_ok=True)
        return BUILTIN_POSES_DIR
    p = Path(text)
    if not p.is_absolute() and folder_paths is not None:
        try:
            p = Path(folder_paths.get_output_directory()) / text
        except Exception:
            p = PACKAGE_ROOT / text
    p.mkdir(parents=True, exist_ok=True)
    return p


def _contact_sheet(paths: list[Path], cell: int = 256, cols: int = 4) -> np.ndarray:
    from PIL import Image

    pose_pngs = [p for p in paths if p.suffix.lower() == ".png"]
    if not pose_pngs:
        # Draw from .pose via OpenPose when PNG skipped
        from ..formats.pose_io import load_pose
        from ..utils import draw_openpose

        imgs = []
        for p in paths:
            if p.suffix.lower() != ".pose":
                continue
            pose = load_pose(p)
            imgs.append(
                Image.fromarray(draw_openpose(pose)).resize(
                    (cell, cell), Image.Resampling.LANCZOS
                )
            )
    else:
        imgs = [
            Image.open(p).convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)
            for p in pose_pngs
        ]
    if not imgs:
        return np.zeros((cell, cell, 3), dtype=np.uint8)

    cols = max(1, min(cols, len(imgs)))
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (0, 0, 0))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        sheet.paste(im, (c * cell, r * cell))
    return np.asarray(sheet, dtype=np.uint8)


class MixamoPoseExport:
    """Export Mixamo FBX to OpenPose .pose library files (multi-view)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fbx": (_list_fbx_files(),),
                "orientations": (list(ORIENTATION_MODES), {"default": "8"}),
                "frames_per_view": (
                    "INT",
                    {"default": 4, "min": 1, "max": 64},
                ),
            },
            "optional": {
                "out_dir": ("STRING", {"default": "poses"}),
                "write_png": ("BOOLEAN", {"default": False}),
                "size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "IMAGE")
    RETURN_NAMES = ("pose_list", "count", "preview")
    FUNCTION = "export"
    CATEGORY = "CharacterPose"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, fbx, orientations, frames_per_view, out_dir="poses", write_png=False, size=1024):
        h = hashlib.sha1()
        h.update(f"{fbx}|{orientations}|{frames_per_view}|{out_dir}|{write_png}|{size}".encode())
        try:
            path = _resolve_fbx(fbx)
            st = path.stat()
            h.update(str(st.st_mtime_ns).encode())
            h.update(str(st.st_size).encode())
        except Exception:
            h.update(b"missing")
        return h.hexdigest()

    def export(
        self,
        fbx,
        orientations="8",
        frames_per_view=4,
        out_dir="poses",
        write_png=False,
        size=1024,
    ):
        import sys

        root = str(PACKAGE_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        from mixamo_tools.export_views import export_views

        fbx_path = _resolve_fbx(fbx)
        dest = _resolve_out_dir(out_dir)
        mode = str(orientations).strip().lower()
        if mode not in ORIENTATION_MODES:
            mode = "8"

        written = export_views(
            fbx_path,
            dest,
            positions=int(frames_per_view),
            view=mode,
            size=int(size),
            write_png=bool(write_png),
            write_pose=True,
        )
        pose_names = sorted(
            {p.name for p in written if p.suffix.lower() == ".pose"}
        )
        pose_list = "\n".join(pose_names)
        preview = _contact_sheet(written if write_png else [dest / n for n in pose_names])
        return (pose_list, len(pose_names), np_to_tensor(preview))


NODE_CLASS_MAPPINGS = {
    "CP_MixamoPoseExport": MixamoPoseExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CP_MixamoPoseExport": "Mixamo Pose Export",
}
