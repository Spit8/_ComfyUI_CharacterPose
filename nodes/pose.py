"""Pose nodes: ExtractPose, PoseLibraryLoad, ApplyPose, SavePose."""

from __future__ import annotations

from pathlib import Path

try:
    import folder_paths
except ImportError:  # Allow import outside ComfyUI (tests / tooling)
    folder_paths = None

from ..formats.pose_io import load_pose, make_pose, save_pose, scale_pose
from ..utils import (
    draw_openpose,
    estimate_keypoints_mediapipe,
    np_to_tensor,
    parse_dwpose_json_keypoints,
    tensor_to_np,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
BUILTIN_POSES_DIR = PACKAGE_ROOT / "poses"


def _input_directory() -> Path | None:
    if folder_paths is None:
        return None
    try:
        return Path(folder_paths.get_input_directory())
    except Exception:
        return None


def _output_directory() -> Path | None:
    if folder_paths is None:
        return None
    try:
        return Path(folder_paths.get_output_directory())
    except Exception:
        return None


def _list_pose_files() -> list[str]:
    names: list[str] = []
    search_dirs = [BUILTIN_POSES_DIR]
    input_dir = _input_directory()
    if input_dir is not None:
        search_dirs.append(input_dir / "poses")

    for d in search_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.pose")):
            rel = p.name
            if rel not in names:
                names.append(rel)
    return names or ["idle.pose"]


def _resolve_pose_path(filename: str) -> Path:
    candidates = [
        BUILTIN_POSES_DIR / filename,
        Path(filename),
    ]
    input_dir = _input_directory()
    if input_dir is not None:
        candidates.insert(1, input_dir / "poses" / filename)
        candidates.append(input_dir / filename)
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Pose file not found: {filename}")


class ExtractPose:
    """Extract OpenPose COCO-18 keypoints from an image into a POSE object."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "name": ("STRING", {"default": "extracted"}),
            },
        }

    RETURN_TYPES = ("POSE", "IMAGE")
    RETURN_NAMES = ("pose", "skeleton_preview")
    FUNCTION = "extract"
    CATEGORY = "CharacterPose/Pose"

    def extract(self, image, pose_keypoint=None, name="extracted"):
        rgb = tensor_to_np(image)
        h, w, _ = rgb.shape

        keypoints = None
        if pose_keypoint is not None:
            keypoints = parse_dwpose_json_keypoints(pose_keypoint, w, h)

        if keypoints is None:
            keypoints = estimate_keypoints_mediapipe(rgb)

        if keypoints is None:
            # No fake T-pose — empty keypoints (blank preview)
            keypoints = [[0.0, 0.0, 0.0] for _ in range(18)]

        pose = make_pose(keypoints, width=w, height=h, name=name or "extracted")
        preview = draw_openpose(pose, width=w, height=h)
        return (pose, np_to_tensor(preview))


class PoseLibraryLoad:
    """Load a pose from the built-in library or input/poses."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_file": (_list_pose_files(),),
            },
            "optional": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
            },
        }

    RETURN_TYPES = ("POSE",)
    RETURN_NAMES = ("pose",)
    FUNCTION = "load"
    CATEGORY = "CharacterPose/Pose"

    def load(self, pose_file, width=1024, height=1024):
        path = _resolve_pose_path(pose_file)
        pose = load_pose(path)
        pose = scale_pose(pose, width, height)
        return (pose,)


class ApplyPose:
    """Render a POSE as an OpenPose skeleton IMAGE for ControlNet."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose": ("POSE",),
            },
            "optional": {
                "width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "stick_width": ("INT", {"default": 0, "min": 0, "max": 32}),
            },
        }

    RETURN_TYPES = ("IMAGE", "POSE")
    RETURN_NAMES = ("skeleton", "pose")
    FUNCTION = "apply"
    CATEGORY = "CharacterPose/Pose"

    def apply(self, pose, width=0, height=0, stick_width=0):
        w = int(width) if width and width > 0 else int(pose.get("width", 1024))
        h = int(height) if height and height > 0 else int(pose.get("height", 1024))
        scaled = scale_pose(pose, w, h)
        thickness = stick_width if stick_width and stick_width > 0 else None
        img = draw_openpose(scaled, width=w, height=h, thickness=thickness)
        return (np_to_tensor(img), scaled)


class SavePose:
    """Save a POSE object to a .pose JSON file."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose": ("POSE",),
                "filename": ("STRING", {"default": "custom_pose"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    CATEGORY = "CharacterPose/Pose"
    OUTPUT_NODE = True

    def save(self, pose, filename):
        name = filename.strip() or "custom_pose"
        if not name.endswith(".pose"):
            name += ".pose"
        out_dir = BUILTIN_POSES_DIR
        output_dir = _output_directory()
        if output_dir is not None:
            out_dir = output_dir / "poses"
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / name
        save_pose(pose, path)
        return (str(path),)


NODE_CLASS_MAPPINGS = {
    "CP_ExtractPose": ExtractPose,
    "CP_PoseLibraryLoad": PoseLibraryLoad,
    "CP_ApplyPose": ApplyPose,
    "CP_SavePose": SavePose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CP_ExtractPose": "Extract Pose",
    "CP_PoseLibraryLoad": "Pose Library Load",
    "CP_ApplyPose": "Apply Pose",
    "CP_SavePose": "Save Pose",
}
