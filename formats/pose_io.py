"""Pose format (.pose) — OpenPose COCO-18 keypoints as JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# OpenPose COCO body (18 keypoints)
KEYPOINT_NAMES = [
    "nose",
    "neck",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "r_hip",
    "r_knee",
    "r_ankle",
    "l_hip",
    "l_knee",
    "l_ankle",
    "r_eye",
    "l_eye",
    "r_ear",
    "l_ear",
]

# Limb connections for OpenPose skeleton drawing (pair of keypoint indices)
LIMB_PAIRS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (0, 14),
    (0, 15),
    (14, 16),
    (15, 17),
]

# BGR colors matching classic OpenPose palette (approx) — one per limb
LIMB_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
]

# BGR colors for the 18 COCO keypoints (classic OpenPose joint palette)
KEYPOINT_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 85, 85),
]


def empty_keypoints(n: int = 18) -> list[list[float]]:
    """Return n keypoints as [x, y, confidence], all zero."""
    return [[0.0, 0.0, 0.0] for _ in range(n)]


def make_pose(
    keypoints: list[list[float]],
    width: int = 1024,
    height: int = 1024,
    name: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a POSE dict.

    Keypoints are absolute pixel coordinates [x, y, confidence] in the
    given width/height canvas. Confidence <= 0 means missing.
    """
    kps = [list(map(float, kp[:3])) + [0.0] * max(0, 3 - len(kp)) for kp in keypoints]
    while len(kps) < 18:
        kps.append([0.0, 0.0, 0.0])
    return {
        "version": 1,
        "name": name,
        "width": int(width),
        "height": int(height),
        "keypoints": kps[:18],
        "metadata": metadata or {},
    }


def save_pose(pose: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(pose, f, indent=2)


def load_pose(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "keypoints" not in data:
        raise ValueError(f"Invalid .pose file (missing keypoints): {path}")
    data.setdefault("version", 1)
    data.setdefault("name", path.stem)
    data.setdefault("width", 1024)
    data.setdefault("height", 1024)
    data.setdefault("metadata", {})
    kps = data["keypoints"]
    while len(kps) < 18:
        kps.append([0.0, 0.0, 0.0])
    data["keypoints"] = kps[:18]
    return data


def scale_pose(pose: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """Rescale keypoints from pose canvas to a new resolution."""
    src_w = max(1, int(pose.get("width", 1024)))
    src_h = max(1, int(pose.get("height", 1024)))
    sx = width / src_w
    sy = height / src_h
    kps = []
    for x, y, c in pose["keypoints"]:
        if c <= 0:
            kps.append([0.0, 0.0, 0.0])
        else:
            kps.append([x * sx, y * sy, float(c)])
    out = dict(pose)
    out["width"] = int(width)
    out["height"] = int(height)
    out["keypoints"] = kps
    return out
