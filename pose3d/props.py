"""Attachable props / mounts as 3D polylines for guide rendering."""

from __future__ import annotations

from typing import Any

import numpy as np

from .skeleton import joint_orientations, rot_xyz

# slot → joint name used as attachment anchor
SLOT_JOINTS = {
    "r_wrist": "r_wrist",
    "l_wrist": "l_wrist",
    "back": "chest",
    "pelvis": "pelvis",
    "root_world": None,  # world space, no parent
}

# Prop definitions: local polylines (list of Nx3 arrays) in attachment frame
# + default slot + prompt hint
PROP_LIBRARY: dict[str, dict[str, Any]] = {
    "none": {
        "slot": "r_wrist",
        "polylines": [],
        "hint": "",
    },
    "sword": {
        "slot": "r_wrist",
        "polylines": [
            # blade along local +Y (up from grip), grip at origin
            np.array([[0, -0.05, 0], [0, 0.55, 0]], dtype=np.float64),
            np.array([[-0.08, 0.02, 0], [0.08, 0.02, 0]], dtype=np.float64),  # guard
        ],
        "local_rot": (0.0, 0.0, -25.0),
        "hint": "holding a sword in the right hand",
    },
    "shield": {
        "slot": "l_wrist",
        "polylines": [
            # oval-ish shield in YZ plane
            np.array(
                [
                    [0.0, 0.15, 0.05],
                    [0.0, 0.05, 0.18],
                    [0.0, -0.15, 0.12],
                    [0.0, -0.20, 0.0],
                    [0.0, -0.15, -0.12],
                    [0.0, 0.05, -0.18],
                    [0.0, 0.15, -0.05],
                    [0.0, 0.15, 0.05],
                ],
                dtype=np.float64,
            ),
        ],
        "local_rot": (0.0, 20.0, 0.0),
        "hint": "holding a round shield in the left hand",
    },
    "staff": {
        "slot": "r_wrist",
        "polylines": [
            np.array([[0, -0.35, 0], [0, 0.75, 0]], dtype=np.float64),
            np.array([[-0.06, 0.70, 0], [0.06, 0.70, 0], [0, 0.82, 0], [-0.06, 0.70, 0]], dtype=np.float64),
        ],
        "local_rot": (15.0, 0.0, -10.0),
        "hint": "holding a long staff in the right hand",
    },
    "bow": {
        "slot": "l_wrist",
        "polylines": [
            np.array(
                [
                    [0.0, 0.35, 0.0],
                    [0.05, 0.15, 0.08],
                    [0.05, -0.15, 0.08],
                    [0.0, -0.35, 0.0],
                ],
                dtype=np.float64,
            ),
            np.array([[0.0, 0.35, 0.0], [0.0, -0.35, 0.0]], dtype=np.float64),  # string
        ],
        "local_rot": (0.0, 0.0, 90.0),
        "hint": "holding a bow in the left hand",
    },
    "horse": {
        "slot": "root_world",
        "polylines": [
            # simplified side-view horse stick (X forward-ish, Y up) under rider
            np.array(
                [
                    [0.55, 0.55, 0.0],  # head
                    [0.35, 0.70, 0.0],  # neck top
                    [0.10, 0.75, 0.0],  # withers
                    [-0.35, 0.72, 0.0],  # croup
                    [-0.55, 0.55, 0.0],  # tail base
                ],
                dtype=np.float64,
            ),
            np.array([[0.10, 0.75, 0.0], [0.15, 0.35, 0.08], [0.12, 0.05, 0.08]], dtype=np.float64),  # front leg
            np.array([[0.10, 0.75, 0.0], [0.15, 0.35, -0.08], [0.12, 0.05, -0.08]], dtype=np.float64),
            np.array([[-0.25, 0.72, 0.0], [-0.28, 0.35, 0.08], [-0.30, 0.05, 0.08]], dtype=np.float64),  # hind
            np.array([[-0.25, 0.72, 0.0], [-0.28, 0.35, -0.08], [-0.30, 0.05, -0.08]], dtype=np.float64),
            np.array([[0.55, 0.55, 0.0], [0.65, 0.50, 0.05]], dtype=np.float64),  # muzzle
        ],
        "local_rot": (0.0, 0.0, 0.0),
        "world_offset": (0.0, 0.0, 0.0),
        "hint": "riding a horse, mounted on horseback",
        "rider_lift": 0.55,
    },
}


def list_props() -> list[str]:
    return list(PROP_LIBRARY.keys())


def resolve_props(names: list[str] | str) -> list[dict[str, Any]]:
    if isinstance(names, str):
        parts = [p.strip() for p in names.replace(";", ",").split(",") if p.strip()]
    else:
        parts = list(names)
    out = []
    for p in parts:
        key = p.strip().lower()
        if key in ("", "none"):
            continue
        if key not in PROP_LIBRARY:
            raise KeyError(f"Unknown prop: {p}. Available: {list_props()}")
        out.append({"name": key, **PROP_LIBRARY[key]})
    return out


def prop_hint_text(props: list[dict[str, Any]]) -> str:
    hints = [str(p.get("hint") or "").strip() for p in props]
    hints = [h for h in hints if h]
    if not hints:
        return ""
    if len(hints) == 1:
        return hints[0]
    return ", ".join(hints[:-1]) + ", and " + hints[-1]


def prop_world_polylines(
    props: list[dict[str, Any]],
    joint_positions: dict[str, np.ndarray],
    joint_angles: dict[str, tuple[float, float, float]],
) -> list[np.ndarray]:
    """Transform prop polylines into world space."""
    orients = joint_orientations(joint_angles)
    world_lines: list[np.ndarray] = []

    for prop in props:
        slot = prop.get("slot", "r_wrist")
        local_rot = rot_xyz(*(prop.get("local_rot") or (0.0, 0.0, 0.0)))
        polylines = prop.get("polylines") or []

        if slot == "root_world" or SLOT_JOINTS.get(slot) is None:
            R = local_rot
            origin = np.array(prop.get("world_offset") or (0.0, 0.0, 0.0), dtype=np.float64)
        else:
            joint = SLOT_JOINTS[slot]
            origin = joint_positions[joint]
            R_joint = orients[joint]
            R = R_joint @ local_rot

        for line in polylines:
            pts = np.asarray(line, dtype=np.float64).reshape(-1, 3)
            world = (R @ pts.T).T + origin
            world_lines.append(world)

    return world_lines
