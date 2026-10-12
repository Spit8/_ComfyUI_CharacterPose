"""Pose library helpers + PoseLibraryLoad / ApplyPose nodes."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

try:
    import folder_paths
except ImportError:  # Allow import outside ComfyUI (tests / tooling)
    folder_paths = None

from ..formats.pose_io import load_pose, scale_pose
from ..utils import draw_openpose, np_to_tensor

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
BUILTIN_POSES_DIR = PACKAGE_ROOT / "poses"
EMPTY_POSE_PLACEHOLDER = "(no .pose files)"
ALL_FILTER = "ALL"
_VIEW_CODES = ("SE", "SO", "NE", "NO", "LEFT", "RIGHT", "TOP", "BOT")
_VIEW_ORDER_INDEX = {v: i for i, v in enumerate(_VIEW_CODES)}
_POSE_NAME_RE = re.compile(
    r"^(?P<anim>.+)_(?P<view>SE|SO|NE|NO|LEFT|RIGHT|TOP|BOT)_(?P<frame>\d{2})$",
    re.IGNORECASE,
)


def _input_directory() -> Path | None:
    if folder_paths is None:
        return None
    try:
        return Path(folder_paths.get_input_directory())
    except Exception:
        return None



def _list_pose_files() -> list[str]:
    """All ``*.pose`` basenames in package ``poses/`` and ``input/poses/`` (no name filter)."""
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
    return names or [EMPTY_POSE_PLACEHOLDER]


def _resolve_pose_path(filename: str) -> Path:
    if not filename or filename == EMPTY_POSE_PLACEHOLDER:
        raise FileNotFoundError(
            "No .pose files found. Export with mixamo_tools into poses/ "
            f"(or ComfyUI input/poses/), then refresh the node. Looking under: {BUILTIN_POSES_DIR}"
        )
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


def parse_pose_filename(name: str) -> tuple[str, str, int] | None:
    """Parse ``{anim}_{VIEW}_{ii}.pose`` → (anim, VIEW, frame_idx) or None."""
    stem = Path(str(name)).stem
    m = _POSE_NAME_RE.match(stem)
    if not m:
        return None
    return m.group("anim"), m.group("view").upper(), int(m.group("frame"))


def _pose_search_dirs() -> list[Path]:
    dirs = [BUILTIN_POSES_DIR]
    input_dir = _input_directory()
    if input_dir is not None:
        dirs.append(input_dir / "poses")
    return dirs


_POSE_COMBO_CACHE: dict[str, tuple[str, list[str]]] = {}


def list_pose_index(
    pose_list: str | None = None,
) -> list[dict[str, Any]]:
    """Indexed library entries: filename, anim, view, frame, mtime."""
    try:
        names: list[str] = []
        if pose_list and str(pose_list).strip():
            for line in str(pose_list).splitlines():
                n = line.strip()
                if not n or n.startswith("#"):
                    continue
                if not n.endswith(".pose"):
                    n = f"{n}.pose"
                if n not in names:
                    names.append(n)
        else:
            names = [n for n in _list_pose_files() if n != EMPTY_POSE_PLACEHOLDER]

        entries: list[dict[str, Any]] = []
        for name in names:
            parsed = parse_pose_filename(name)
            if parsed is None:
                continue
            anim, view, frame = parsed
            mtime = 0.0
            try:
                mtime = _resolve_pose_path(name).stat().st_mtime
            except FileNotFoundError:
                pass
            except OSError:
                pass
            entries.append(
                {
                    "filename": name,
                    "anim": anim,
                    "view": view,
                    "frame": frame,
                    "frame_key": f"{frame:02d}",
                    "mtime": mtime,
                }
            )

        entries.sort(
            key=lambda e: (
                e["anim"],
                _VIEW_ORDER_INDEX.get(e["view"], 99),
                e["frame"],
                e["filename"],
            )
        )
        return entries
    except Exception:
        return []


def _cached_combo(kind: str, builder, pose_list: str | None = None) -> list[str]:
    """Cache combo lists by fingerprint; never raise (INPUT_TYPES must be safe)."""
    try:
        fp = poses_dir_fingerprint(pose_list)
        key = f"{kind}|{pose_list or ''}"
        hit = _POSE_COMBO_CACHE.get(key)
        if hit is not None and hit[0] == fp:
            return hit[1]
        values = builder()
        _POSE_COMBO_CACHE[key] = (fp, values)
        return values
    except Exception:
        return [ALL_FILTER]


def list_animations(pose_list: str | None = None) -> list[str]:
    def _build() -> list[str]:
        anims = sorted({e["anim"] for e in list_pose_index(pose_list)})
        return [ALL_FILTER] + anims if anims else [ALL_FILTER]

    return _cached_combo("anim", _build, pose_list)


def list_views_present(pose_list: str | None = None) -> list[str]:
    def _build() -> list[str]:
        present = {e["view"] for e in list_pose_index(pose_list)}
        ordered = [v for v in _VIEW_CODES if v in present]
        return [ALL_FILTER] + ordered if ordered else [ALL_FILTER] + list(_VIEW_CODES)

    return _cached_combo("view", _build, pose_list)


def list_frames_present(pose_list: str | None = None) -> list[str]:
    def _build() -> list[str]:
        frames = sorted({e["frame_key"] for e in list_pose_index(pose_list)})
        return [ALL_FILTER] + frames if frames else [ALL_FILTER]

    return _cached_combo("frame", _build, pose_list)


def filter_pose_entries(
    *,
    animation: str = ALL_FILTER,
    orientation: str = ALL_FILTER,
    frame: str = ALL_FILTER,
    pose_list: str | None = None,
) -> list[dict[str, Any]]:
    """Filter index by animation / orientation / frame (ALL = no filter)."""
    anim_f = (animation or ALL_FILTER).strip()
    view_f = (orientation or ALL_FILTER).strip().upper()
    frame_f = (frame or ALL_FILTER).strip()
    if frame_f != ALL_FILTER and frame_f.isdigit():
        frame_f = f"{int(frame_f):02d}"

    out: list[dict[str, Any]] = []
    for e in list_pose_index(pose_list):
        if anim_f != ALL_FILTER and e["anim"] != anim_f:
            continue
        if view_f != ALL_FILTER and e["view"] != view_f:
            continue
        if frame_f != ALL_FILTER and e["frame_key"] != frame_f:
            continue
        out.append(e)
    return out


def poses_dir_fingerprint(pose_list: str | None = None) -> str:
    """Stable hash of pose library mtimes (+ optional pose_list text)."""
    h = hashlib.sha1()
    h.update((pose_list or "").encode("utf-8", errors="replace"))
    for d in _pose_search_dirs():
        if not d.exists():
            continue
        for p in sorted(d.glob("*.pose")):
            try:
                st = p.stat()
                h.update(p.name.encode("utf-8"))
                h.update(str(st.st_mtime_ns).encode("ascii"))
                h.update(str(st.st_size).encode("ascii"))
            except OSError:
                continue
    return h.hexdigest()


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
    CATEGORY = "CharacterPose"

    def load(self, pose_file, width=1024, height=1024):
        try:
            path = _resolve_pose_path(pose_file)
        except FileNotFoundError as e:
            raise RuntimeError(str(e)) from e
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
    CATEGORY = "CharacterPose"

    def apply(self, pose, width=0, height=0, stick_width=0):
        w = int(width) if width and width > 0 else int(pose.get("width", 1024))
        h = int(height) if height and height > 0 else int(pose.get("height", 1024))
        scaled = scale_pose(pose, w, h)
        thickness = stick_width if stick_width and stick_width > 0 else None
        img = draw_openpose(scaled, width=w, height=h, thickness=thickness)
        return (np_to_tensor(img), scaled)


NODE_CLASS_MAPPINGS = {
    "CP_PoseLibraryLoad": PoseLibraryLoad,
    "CP_ApplyPose": ApplyPose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CP_PoseLibraryLoad": "Pose Library Load",
    "CP_ApplyPose": "Apply Pose",
}
