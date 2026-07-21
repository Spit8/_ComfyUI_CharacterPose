"""Perspective camera for 3D → 2D projection."""

from __future__ import annotations

import numpy as np

# Isometric-style camera presets: yaw, pitch, roll, distance
# Yaw 0 = looking at character from +Z (front). Positive yaw orbits toward character's right.
CAMERA_PRESETS: dict[str, dict[str, float]] = {
    "S": {"yaw": 0.0, "pitch": -12.0, "roll": 0.0, "distance": 3.2},
    "SE": {"yaw": -45.0, "pitch": -15.0, "roll": 0.0, "distance": 3.2},
    "E": {"yaw": -90.0, "pitch": -10.0, "roll": 0.0, "distance": 3.2},
    "NE": {"yaw": -135.0, "pitch": -15.0, "roll": 0.0, "distance": 3.2},
    "N": {"yaw": 180.0, "pitch": -12.0, "roll": 0.0, "distance": 3.2},
    "NW": {"yaw": 135.0, "pitch": -15.0, "roll": 0.0, "distance": 3.2},
    "W": {"yaw": 90.0, "pitch": -10.0, "roll": 0.0, "distance": 3.2},
    "SW": {"yaw": 45.0, "pitch": -15.0, "roll": 0.0, "distance": 3.2},
}


def list_camera_presets() -> list[str]:
    return list(CAMERA_PRESETS.keys())


def _rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def camera_matrix(
    yaw: float = 0.0,
    pitch: float = -15.0,
    roll: float = 0.0,
    distance: float = 3.2,
    look_at: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return (R_world_to_cam, cam_pos) for orbit camera around look_at.

    Camera orbits on a sphere: starts in front (+Z), yaw then pitch.
    """
    target = np.array(look_at, dtype=np.float64)
    # Orbit: apply yaw around Y, then pitch around camera-right
    offset = np.array([0.0, 0.0, float(distance)], dtype=np.float64)
    offset = _rot_y(yaw) @ offset
    # Pitch: rotate offset around axis perpendicular to up and offset horizontal
    offset = _rot_x(pitch) @ offset
    # After yaw, pitch around world X is approximate for isometric; refine with roll
    cam_pos = target + offset

    forward = target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-8)

    # Apply roll around forward
    if abs(roll) > 1e-6:
        c, s = np.cos(np.deg2rad(roll)), np.sin(np.deg2rad(roll))
        up2 = c * up + s * right
        right2 = -s * up + c * right
        up, right = up2, right2

    # Rows = camera axes in world (world → camera). +Z looks toward the target.
    R = np.stack([right, up, forward], axis=0)
    return R, cam_pos


def project_points(
    points: np.ndarray,
    R: np.ndarray,
    cam_pos: np.ndarray,
    *,
    width: int = 1024,
    height: int = 1024,
    fov_deg: float = 35.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project Nx3 world points → Nx2 pixels + depth (camera Z, positive in front).

    points: (N, 3)
    returns: xy (N, 2), depth (N,)
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    local = (R @ (pts - cam_pos).T).T  # x right, y up, z forward
    z = local[:, 2]
    z_safe = np.where(np.abs(z) < 1e-5, 1e-5, z)
    f = 0.5 * height / np.tan(np.deg2rad(fov_deg) * 0.5)
    x = local[:, 0] / z_safe * f + width * 0.5
    y = -local[:, 1] / z_safe * f + height * 0.5  # screen Y down
    return np.stack([x, y], axis=1), z
