"""Compose 3D pose → COCO-18 POSE dict + optional prop polylines."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..formats.pose_io import make_pose
from .camera import CAMERA_PRESETS, camera_matrix, project_points
from .presets import get_action_angles
from .props import prop_world_polylines, resolve_props
from .skeleton import COCO18_FROM_JOINT, forward_kinematics, merge_angles


def project_joints(
    positions: dict[str, np.ndarray],
    R: np.ndarray,
    cam_pos: np.ndarray,
    *,
    width: int = 1024,
    height: int = 1024,
    fov_deg: float = 35.0,
) -> list[list[float]]:
    """Map named 3D joints → COCO-18 [x,y,conf] with simple depth occlusion."""
    names = [COCO18_FROM_JOINT[i] for i in range(18)]
    pts = np.stack([positions[n] for n in names], axis=0)
    xy, depth = project_points(pts, R, cam_pos, width=width, height=height, fov_deg=fov_deg)

    # Reference depth at pelvis / neck for occlusion heuristic
    chest_z = float(depth[names.index("neck")]) if "neck" in names else float(np.median(depth))

    kps: list[list[float]] = []
    for i in range(18):
        x, y = float(xy[i, 0]), float(xy[i, 1])
        z = float(depth[i])
        if z <= 0.05:
            kps.append([0.0, 0.0, 0.0])
            continue
        # Soft occlusion: joints clearly behind torso get lower confidence
        conf = 1.0
        if z > chest_z + 0.25:
            conf = 0.45
        # Face landmarks weaker from behind
        if i in (0, 14, 15) and z > chest_z + 0.12:
            conf = min(conf, 0.35)
        if x < -50 or y < -50 or x > width + 50 or y > height + 50:
            conf = 0.0
        if conf <= 0:
            kps.append([0.0, 0.0, 0.0])
        else:
            kps.append([x, y, conf])
    return kps


def compose_pose(
    action: str = "idle",
    *,
    camera_preset: str = "SE",
    yaw: float | None = None,
    pitch: float | None = None,
    roll: float | None = None,
    distance: float | None = None,
    fov_deg: float = 35.0,
    width: int = 1024,
    height: int = 1024,
    props: list[str] | str | None = None,
    angle_overlay: dict[str, tuple[float, float, float]] | None = None,
) -> dict[str, Any]:
    """Build POSE + projected prop polylines + metadata.

    Returns dict with keys: pose, prop_polylines_2d (list of Nx2), prop_hint, camera, action.
    """
    angles = get_action_angles(action)
    angles = merge_angles(angles, angle_overlay)

    # Horse: lift rider
    prop_list = resolve_props(props or [])
    root_offset = (0.0, 0.0, 0.0)
    for p in prop_list:
        if p.get("name") == "horse":
            lift = float(p.get("rider_lift") or 0.55)
            root_offset = (0.0, lift, 0.0)

    positions = forward_kinematics(angles, root_offset=root_offset)

    cam = dict(CAMERA_PRESETS.get(camera_preset.upper(), CAMERA_PRESETS["SE"]))
    if yaw is not None:
        cam["yaw"] = float(yaw)
    if pitch is not None:
        cam["pitch"] = float(pitch)
    if roll is not None:
        cam["roll"] = float(roll)
    if distance is not None:
        cam["distance"] = float(distance)

    R, cam_pos = camera_matrix(
        yaw=cam["yaw"],
        pitch=cam["pitch"],
        roll=cam["roll"],
        distance=cam["distance"],
        look_at=(0.0, 1.0 + root_offset[1] * 0.5, 0.0),
    )

    kps = project_joints(positions, R, cam_pos, width=width, height=height, fov_deg=fov_deg)
    pose = make_pose(
        kps,
        width=width,
        height=height,
        name=f"{action}_{camera_preset}",
        metadata={
            "action": action,
            "camera_preset": camera_preset,
            "camera": cam,
            "source": "pose3d",
        },
    )

    world_lines = prop_world_polylines(prop_list, positions, angles)
    prop_2d: list[np.ndarray] = []
    for line in world_lines:
        xy, depth = project_points(line, R, cam_pos, width=width, height=height, fov_deg=fov_deg)
        # Drop points behind camera
        valid = depth > 0.05
        if not np.any(valid):
            continue
        prop_2d.append(xy)

    from .props import prop_hint_text

    return {
        "pose": pose,
        "prop_polylines_2d": prop_2d,
        "prop_hint": prop_hint_text(prop_list),
        "camera": cam,
        "action": action,
        "props": [p["name"] for p in prop_list],
    }
