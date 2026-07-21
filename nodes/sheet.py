"""Spritesheet export and RPG sheet generation helpers."""

from __future__ import annotations

import json
from pathlib import Path

try:
    import folder_paths
except ImportError:
    folder_paths = None

import numpy as np
import torch
from PIL import Image

from ..formats.pose_io import load_pose, scale_pose
from ..utils import batch_np_to_tensor, draw_openpose, np_to_tensor, tensor_to_np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
BUILTIN_POSES_DIR = PACKAGE_ROOT / "poses"

# Default RPG sheet layout (name -> pose file stem)
# Prefer SE/NE sets with Flux Klein: idle_se, walk_se_01.., run_se_*, jump_se, fight_se_*, work_se_*
# (and *_ne for North-East). Legacy front set kept below.
RPG_POSE_SEQUENCE = [
    "idle_se",
    "walk_se_01",
    "walk_se_02",
    "walk_se_03",
    "walk_se_04",
    "run_se_01",
    "run_se_02",
    "run_se_03",
    "run_se_04",
    "jump_se",
    "fight_se_01",
    "fight_se_02",
    "work_se_01",
    "work_se_02",
]


def _output_dir() -> Path:
    if folder_paths is not None:
        try:
            d = Path(folder_paths.get_output_directory()) / "spritesheets"
            d.mkdir(parents=True, exist_ok=True)
            return d
        except Exception:
            pass
    d = PACKAGE_ROOT / "output" / "spritesheets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _compose_grid(images: list[np.ndarray], columns: int, pad: int = 0, bg=(0, 0, 0)) -> np.ndarray:
    if not images:
        raise ValueError("No images to compose")
    h, w = images[0].shape[:2]
    cols = max(1, columns)
    rows = (len(images) + cols - 1) // cols
    canvas_h = rows * h + (rows + 1) * pad
    canvas_w = cols * w + (cols + 1) * pad
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = bg
    cells = []
    for i, im in enumerate(images):
        if im.shape[0] != h or im.shape[1] != w:
            from PIL import Image as PILImage

            im = np.array(PILImage.fromarray(im).resize((w, h), PILImage.Resampling.LANCZOS))
        r, c = divmod(i, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        canvas[y : y + h, x : x + w] = im
        cells.append({"index": i, "row": r, "col": c, "x": x, "y": y, "w": w, "h": h})
    return canvas, cells


class ExportSpriteSheet:
    """Pack a batch of images into a spritesheet PNG + JSON metadata."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename": ("STRING", {"default": "spritesheet"}),
                "columns": ("INT", {"default": 4, "min": 1, "max": 32}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 64}),
            },
            "optional": {
                "labels": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("spritesheet", "png_path", "json_path")
    FUNCTION = "export"
    CATEGORY = "CharacterPose/Sheet"
    OUTPUT_NODE = True

    def export(self, images, filename="spritesheet", columns=4, padding=0, labels=""):
        batch = []
        for i in range(images.shape[0]):
            arr = images[i].detach().cpu().numpy()
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            batch.append(arr)

        label_list = [ln.strip() for ln in labels.splitlines() if ln.strip()] if labels else []
        grid, cells = _compose_grid(batch, columns=columns, pad=padding)
        for i, cell in enumerate(cells):
            cell["label"] = label_list[i] if i < len(label_list) else f"frame_{i:02d}"

        name = filename.strip() or "spritesheet"
        if name.endswith(".png"):
            name = name[:-4]
        out_dir = _output_dir()
        png_path = out_dir / f"{name}.png"
        json_path = out_dir / f"{name}.json"

        Image.fromarray(grid).save(png_path)
        meta = {
            "filename": png_path.name,
            "columns": columns,
            "padding": padding,
            "frame_count": len(batch),
            "frame_width": batch[0].shape[1],
            "frame_height": batch[0].shape[0],
            "sheet_width": grid.shape[1],
            "sheet_height": grid.shape[0],
            "frames": cells,
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return (np_to_tensor(grid), str(png_path), str(json_path))


class GenerateRPGSheet:
    """Build OpenPose skeletons for the full RPG pose set (ready for CharacterRender loops).

    Returns a batch of skeleton images + pose labels. Connect each skeleton to CharacterRender
    (or use this as a pose preview / ControlNet batch source).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
            },
            "optional": {
                "pose_names": (
                    "STRING",
                    {
                        "default": ",".join(RPG_POSE_SEQUENCE),
                        "multiline": True,
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("skeletons", "labels")
    FUNCTION = "generate"
    CATEGORY = "CharacterPose/Sheet"

    def generate(self, width=1024, height=1024, pose_names=""):
        raw = pose_names.replace("\n", ",").strip() if pose_names else ",".join(RPG_POSE_SEQUENCE)
        names = [n.strip() for n in raw.split(",") if n.strip()]
        frames = []
        labels = []
        for name in names:
            fname = name if name.endswith(".pose") else f"{name}.pose"
            path = BUILTIN_POSES_DIR / fname
            if not path.exists():
                # skip missing with placeholder
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
                frames.append(canvas)
                labels.append(f"{name}(missing)")
                continue
            pose = scale_pose(load_pose(path), width, height)
            frames.append(draw_openpose(pose, width=width, height=height))
            labels.append(name.replace(".pose", ""))

        tensor = batch_np_to_tensor(frames)
        return (tensor, "\n".join(labels))


class PoseBatchFromLibrary:
    """Load multiple library poses as a list of POSE objects (serialized via skeleton batch).

    For graph simplicity this returns skeletons; use PoseLibraryLoad for single POSE.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_names": ("STRING", {"default": "idle,walk_01,walk_02,attack", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("skeletons", "labels")
    FUNCTION = "load"
    CATEGORY = "CharacterPose/Sheet"

    def load(self, pose_names, width=1024, height=1024):
        return GenerateRPGSheet().generate(width=width, height=height, pose_names=pose_names)


NODE_CLASS_MAPPINGS = {
    "CP_ExportSpriteSheet": ExportSpriteSheet,
    "CP_GenerateRPGSheet": GenerateRPGSheet,
    "CP_PoseBatchFromLibrary": PoseBatchFromLibrary,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CP_ExportSpriteSheet": "Export Sprite Sheet",
    "CP_GenerateRPGSheet": "Generate RPG Sheet Skeletons",
    "CP_PoseBatchFromLibrary": "Pose Batch From Library",
}
