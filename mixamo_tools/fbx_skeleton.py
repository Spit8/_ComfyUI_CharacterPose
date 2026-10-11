"""Load Mixamo FBX animation and sample COCO-18 joints at a frozen frame."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .mixamo_bones import COCO18_NAMES, HEAD_LOCAL_OFFSETS, resolve_bone_name


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) → 3x3 rotation."""
    x, y, z, w = [float(v) for v in q]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _local_matrix(row: np.ndarray) -> np.ndarray:
    """anim_to_array row: tx,ty,tz, qx,qy,qz,qw, sx,sy,sz → 4x4 local."""
    t = row[:3]
    q = row[3:7]
    s = row[7:10]
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = _quat_to_mat(q) @ np.diag(s)
    M[:3, 3] = t
    return M


def _build_parent_map(scene: Any) -> dict[str, str | None]:
    parents: dict[str, str | None] = {}
    for n in scene.nodes:
        pname = None
        if n.parent is not None:
            pname = getattr(n.parent, "name", None) or None
            if pname == "":
                pname = None
        parents[n.name] = pname
    return parents


def load_animation(fbx_path: str | Path) -> dict[str, Any]:
    """Load FBX via pufbx and return dense local TRS tracks + hierarchy."""
    try:
        import pufbx
    except ModuleNotFoundError as e:
        import sys

        raise ModuleNotFoundError(
            "pufbx is not installed for this Python interpreter:\n"
            f"  {sys.executable}\n"
            "Install deps with:\n"
            f'  "{sys.executable}" -m pip install -r requirements.txt\n'
            "Then re-run with the same python."
        ) from e

    path = Path(fbx_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    scene = pufbx.load_fbx(str(path))
    data, times, names = pufbx.anim_to_array(str(path))
    data = np.asarray(data, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    names = list(names)
    parents = _build_parent_map(scene)

    return {
        "path": path,
        "scene": scene,
        "data": data,
        "times": times,
        "names": names,
        "name_to_index": {n: i for i, n in enumerate(names)},
        "parents": parents,
        "n_frames": int(data.shape[1]),
    }


def _world_matrix(
    bone: str,
    frame: int,
    data: np.ndarray,
    name_to_index: dict[str, int],
    parents: dict[str, str | None],
) -> np.ndarray:
    chain: list[str] = []
    cur: str | None = bone
    while cur is not None and cur in name_to_index:
        chain.append(cur)
        cur = parents.get(cur)
    chain.reverse()
    M = np.eye(4, dtype=np.float64)
    for bn in chain:
        M = M @ _local_matrix(data[name_to_index[bn], frame])
    return M


def _world_pos(
    bone: str,
    frame: int,
    data: np.ndarray,
    name_to_index: dict[str, int],
    parents: dict[str, str | None],
) -> np.ndarray:
    return _world_matrix(bone, frame, data, name_to_index, parents)[:3, 3].copy()


def sample_coco18_joints(
    anim: dict[str, Any],
    frame: int | None = None,
    *,
    target_height: float = 1.70,
) -> dict[str, np.ndarray]:
    """Return normalized COCO-18 joint positions (Y-up) at one frame."""
    data = anim["data"]
    name_to_index = anim["name_to_index"]
    parents = anim["parents"]
    n_frames = anim["n_frames"]
    available = {n: True for n in anim["names"]}

    if frame is None:
        frame = n_frames // 2
    frame = int(np.clip(frame, 0, n_frames - 1))

    def resolve(logical: str) -> str | None:
        return resolve_bone_name(available, logical)

    logical_to_joint = {
        "neck": "neck",
        "l_shoulder": "l_shoulder",
        "l_elbow": "l_elbow",
        "l_wrist": "l_wrist",
        "r_shoulder": "r_shoulder",
        "r_elbow": "r_elbow",
        "r_wrist": "r_wrist",
        "l_hip": "l_hip",
        "l_knee": "l_knee",
        "l_ankle": "l_ankle",
        "r_hip": "r_hip",
        "r_knee": "r_knee",
        "r_ankle": "r_ankle",
        "l_eye": "l_eye",
        "r_eye": "r_eye",
    }

    raw: dict[str, np.ndarray] = {}
    for logical, joint in logical_to_joint.items():
        bn = resolve(logical)
        if bn is None:
            continue
        raw[joint] = _world_pos(bn, frame, data, name_to_index, parents)

    if "neck" not in raw and "l_shoulder" in raw and "r_shoulder" in raw:
        raw["neck"] = 0.5 * (raw["l_shoulder"] + raw["r_shoulder"])

    head_bn = resolve("head")
    if head_bn is not None:
        head_M = _world_matrix(head_bn, frame, data, name_to_index, parents)
        head_pos = head_M[:3, 3]
        # Mixamo scene units ≈ cm; local face offsets stored as meters → ×20 cm
        scale_off = 20.0
        for joint, off in HEAD_LOCAL_OFFSETS.items():
            if joint in raw:
                continue
            local = np.array(off, dtype=np.float64) * scale_off
            raw[joint] = head_pos + head_M[:3, :3] @ local
        if "nose" not in raw:
            raw["nose"] = head_pos + head_M[:3, :3] @ (np.array([0.0, 0.02, 0.10]) * scale_off)

    if not raw:
        raise RuntimeError("No Mixamo bones could be mapped — unexpected rig names.")

    pts = np.stack(list(raw.values()), axis=0)
    y_min = float(pts[:, 1].min())
    y_max = float(pts[:, 1].max())
    height = max(y_max - y_min, 1e-3)
    scale = target_height / height

    if "l_hip" in raw and "r_hip" in raw:
        mid = 0.5 * (raw["l_hip"] + raw["r_hip"])
    else:
        mid = pts.mean(axis=0)

    origin = np.array([mid[0], y_min, mid[2]], dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for name, p in raw.items():
        out[name] = (p - origin) * scale
    return out


def coco18_keypoints_2d(
    projected: dict[str, tuple[float, float, float]],
) -> list[list[float]]:
    """Build COCO-18 [x, y, conf] list from projected joint dict."""
    kps: list[list[float]] = []
    for name in COCO18_NAMES:
        if name in projected:
            x, y, c = projected[name]
            kps.append([float(x), float(y), float(c)])
        else:
            kps.append([0.0, 0.0, 0.0])
    return kps
