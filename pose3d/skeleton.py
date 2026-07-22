"""Humanoid kinematic chain (Y-up, character faces +Z in rest pose)."""

from __future__ import annotations

from typing import Any

import numpy as np

# Rest-pose bone lengths (unitless; scaled later). Parent → child offsets in rest.
# Facing +Z, right = +X, up = +Y.
BONE_OFFSETS: dict[str, tuple[str | None, np.ndarray]] = {
    "pelvis": (None, np.array([0.0, 0.95, 0.0])),
    "spine": ("pelvis", np.array([0.0, 0.22, 0.0])),
    "chest": ("spine", np.array([0.0, 0.22, 0.0])),
    "neck": ("chest", np.array([0.0, 0.12, 0.0])),
    "head": ("neck", np.array([0.0, 0.14, 0.0])),
    "r_shoulder": ("chest", np.array([0.18, 0.08, 0.0])),
    # Arms hang down at rest (along -Y), slight outward — not T-pose
    "r_elbow": ("r_shoulder", np.array([0.06, -0.28, 0.0])),
    "r_wrist": ("r_elbow", np.array([0.03, -0.26, 0.0])),
    "l_shoulder": ("chest", np.array([-0.18, 0.08, 0.0])),
    "l_elbow": ("l_shoulder", np.array([-0.06, -0.28, 0.0])),
    "l_wrist": ("l_elbow", np.array([-0.03, -0.26, 0.0])),
    "r_hip": ("pelvis", np.array([0.10, -0.05, 0.0])),
    "r_knee": ("r_hip", np.array([0.0, -0.42, 0.0])),
    "r_ankle": ("r_knee", np.array([0.0, -0.40, 0.0])),
    "l_hip": ("pelvis", np.array([-0.10, -0.05, 0.0])),
    "l_knee": ("l_hip", np.array([0.0, -0.42, 0.0])),
    "l_ankle": ("l_knee", np.array([0.0, -0.40, 0.0])),
    # Face landmarks (children of head, local offsets)
    "nose": ("head", np.array([0.0, 0.02, 0.10])),
    "r_eye": ("head", np.array([0.04, 0.04, 0.08])),
    "l_eye": ("head", np.array([-0.04, 0.04, 0.08])),
    "r_ear": ("head", np.array([0.08, 0.02, 0.02])),
    "l_ear": ("head", np.array([-0.08, 0.02, 0.02])),
}

# Order for FK (parents before children)
FK_ORDER = [
    "pelvis",
    "spine",
    "chest",
    "neck",
    "head",
    "nose",
    "r_eye",
    "l_eye",
    "r_ear",
    "l_ear",
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
]

JOINT_NAMES = list(FK_ORDER)

# OpenPose COCO-18 index → joint name
COCO18_FROM_JOINT = {
    0: "nose",
    1: "neck",
    2: "r_shoulder",
    3: "r_elbow",
    4: "r_wrist",
    5: "l_shoulder",
    6: "l_elbow",
    7: "l_wrist",
    8: "r_hip",
    9: "r_knee",
    10: "r_ankle",
    11: "l_hip",
    12: "l_knee",
    13: "l_ankle",
    14: "r_eye",
    15: "l_eye",
    16: "r_ear",
    17: "l_ear",
}


def _deg2rad(d: float) -> float:
    return float(d) * np.pi / 180.0


def rot_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Euler XYZ (degrees) → 3x3 rotation matrix."""
    ax, ay, az = _deg2rad(rx), _deg2rad(ry), _deg2rad(rz)
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz_m @ ry_m @ rx_m


def forward_kinematics(
    joint_angles: dict[str, tuple[float, float, float]] | None = None,
    *,
    root_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, np.ndarray]:
    """Return world-space joint positions (3,) for each named joint.

    joint_angles: bone_name → (rx, ry, rz) degrees applied at that bone's local frame.
    """
    angles = joint_angles or {}
    positions: dict[str, np.ndarray] = {}
    rotations: dict[str, np.ndarray] = {}

    root = np.array(root_offset, dtype=np.float64)

    for name in FK_ORDER:
        parent, offset = BONE_OFFSETS[name]
        local_r = rot_xyz(*(angles.get(name, (0.0, 0.0, 0.0))))
        if parent is None:
            R = local_r
            pos = root + offset
        else:
            Rp = rotations[parent]
            R = Rp @ local_r
            pos = positions[parent] + Rp @ offset
        rotations[name] = R
        positions[name] = pos

    return positions


def joint_orientations(
    joint_angles: dict[str, tuple[float, float, float]] | None = None,
) -> dict[str, np.ndarray]:
    """World rotation matrices per joint (for prop attachment)."""
    angles = joint_angles or {}
    rotations: dict[str, np.ndarray] = {}
    for name in FK_ORDER:
        parent, _ = BONE_OFFSETS[name]
        local_r = rot_xyz(*(angles.get(name, (0.0, 0.0, 0.0))))
        if parent is None:
            rotations[name] = local_r
        else:
            rotations[name] = rotations[parent] @ local_r
    return rotations


def merge_angles(
    base: dict[str, tuple[float, float, float]],
    overlay: dict[str, tuple[float, float, float]] | None,
) -> dict[str, tuple[float, float, float]]:
    out = dict(base)
    if overlay:
        for k, v in overlay.items():
            if k in out:
                a = out[k]
                out[k] = (a[0] + v[0], a[1] + v[1], a[2] + v[2])
            else:
                out[k] = tuple(v)  # type: ignore[assignment]
    return out
