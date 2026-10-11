"""Camera presets: French ISO (SE/SO/NE/NO) + cardinal (LEFT/RIGHT/TOP/BOT)."""

from __future__ import annotations

import numpy as np

# Orbit params around look_at (Y-up). ISO matches pose3d SE/SW/NE/NW naming.
CAMERA_PRESETS: dict[str, dict[str, float]] = {
    "SE": {"yaw": -45.0, "pitch": -15.0, "roll": 0.0, "distance": 3.2},
    "SO": {"yaw": 45.0, "pitch": -15.0, "roll": 0.0, "distance": 3.2},  # SW
    "NE": {"yaw": -135.0, "pitch": -15.0, "roll": 0.0, "distance": 3.2},
    "NO": {"yaw": 135.0, "pitch": -15.0, "roll": 0.0, "distance": 3.2},  # NW
    "LEFT": {"yaw": 90.0, "pitch": -15.0, "roll": 0.0, "distance": 3.2},
    "RIGHT": {"yaw": -90.0, "pitch": -15.0, "roll": 0.0, "distance": 3.2},
    # 2D ISO cardinals (map haut/bas): TOP≈Nord=dos, BOT≈Sud=face — not bird's-eye
    "TOP": {"yaw": 180.0, "pitch": -15.0, "roll": 0.0, "distance": 3.2},
    "BOT": {"yaw": 0.0, "pitch": -15.0, "roll": 0.0, "distance": 3.2},
}

VIEW_ORDER = ("SE", "SO", "NE", "NO", "LEFT", "RIGHT", "TOP", "BOT")

VIEW_SETS: dict[str, tuple[str, ...]] = {
    "4_iso": ("SE", "SO", "NE", "NO"),
    "4_cardinal": ("TOP", "BOT", "LEFT", "RIGHT"),
    "6": ("SE", "SO", "NE", "NO", "TOP", "BOT"),
    "8": VIEW_ORDER,
}


def _rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def camera_matrix(
    yaw: float = 0.0,
    pitch: float = -15.0,
    roll: float = 0.0,
    distance: float = 3.2,
    look_at: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return (R_world_to_cam, cam_pos) for orbit camera around look_at."""
    target = np.array(look_at, dtype=np.float64)
    offset = np.array([0.0, 0.0, float(distance)], dtype=np.float64)
    offset = _rot_y(yaw) @ offset
    offset = _rot_x(pitch) @ offset
    cam_pos = target + offset

    forward = target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    # Near top/bottom views: Y-up is parallel to forward — pick a stable alternate up
    if abs(float(np.dot(forward, world_up))) > 0.99:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(forward, world_up))) > 0.99:
            world_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-8)

    if abs(roll) > 1e-6:
        c, s = np.cos(np.deg2rad(roll)), np.sin(np.deg2rad(roll))
        up2 = c * up + s * right
        right2 = -s * up + c * right
        up, right = up2, right2

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
    """Project Nx3 world points → Nx2 pixels + depth (camera Z, positive in front)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    local = (R @ (pts - cam_pos).T).T
    z = local[:, 2]
    z_safe = np.where(np.abs(z) < 1e-5, 1e-5, z)
    f = 0.5 * height / np.tan(np.deg2rad(fov_deg) * 0.5)
    x = local[:, 0] / z_safe * f + width * 0.5
    y = -local[:, 1] / z_safe * f + height * 0.5
    return np.stack([x, y], axis=1), z


def project_joints(
    joints_xyz: dict[str, np.ndarray],
    view: str,
    *,
    width: int = 1024,
    height: int = 1024,
    look_at: tuple[float, float, float] | None = None,
    distance: float | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Project named 3D joints to (x, y, confidence) for one French view label."""
    preset = dict(CAMERA_PRESETS[view.upper()])
    if distance is not None:
        preset["distance"] = float(distance)
    names = list(joints_xyz.keys())
    pts = np.stack([joints_xyz[n] for n in names], axis=0)
    if look_at is None:
        look_at = tuple(float(v) for v in pts.mean(axis=0))
    R, cam_pos = camera_matrix(look_at=look_at, **preset)
    xy, depth = project_points(pts, R, cam_pos, width=width, height=height)
    out: dict[str, tuple[float, float, float]] = {}
    for i, name in enumerate(names):
        conf = 1.0 if depth[i] > 0.05 else 0.0
        out[name] = (float(xy[i, 0]), float(xy[i, 1]), conf)
    return out


def _collect_points(joints_per_frame: list[dict[str, np.ndarray]]) -> np.ndarray:
    chunks = []
    for joints in joints_per_frame:
        if not joints:
            continue
        chunks.append(np.stack(list(joints.values()), axis=0))
    if not chunks:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def _fits_canvas(
    pts: np.ndarray,
    views: list[str],
    look_at: tuple[float, float, float],
    distance: float,
    *,
    size: int,
    margin: float,
) -> bool:
    """True if all projected points of all views lie inside the padded canvas."""
    if pts.shape[0] == 0:
        return True
    pad = float(margin) * size
    lo, hi = pad, size - pad
    for view in views:
        preset = dict(CAMERA_PRESETS[view.upper()])
        preset["distance"] = float(distance)
        R, cam_pos = camera_matrix(look_at=look_at, **preset)
        xy, depth = project_points(pts, R, cam_pos, width=size, height=size)
        valid = depth > 0.05
        if not np.any(valid):
            return False
        xy = xy[valid]
        if xy[:, 0].min() < lo or xy[:, 0].max() > hi:
            return False
        if xy[:, 1].min() < lo or xy[:, 1].max() > hi:
            return False
    return True


def fit_camera_distance(
    joints_per_frame: list[dict[str, np.ndarray]],
    views: list[str] | tuple[str, ...],
    *,
    size: int = 1024,
    margin: float = 0.10,
    look_at: tuple[float, float, float] = (0.0, 0.95, 0.0),
    dist_min: float = 1.5,
    dist_max: float = 12.0,
) -> tuple[tuple[float, float, float], float]:
    """Pick one distance so every joint of every frame/view fits with margin.

    Pan/tilt come from each view's yaw/pitch preset; look_at and distance are
    shared across all frames (no per-frame zoom jump).
    """
    views_u = [v.upper() for v in views]
    for v in views_u:
        if v not in CAMERA_PRESETS:
            raise ValueError(f"Unknown view {v!r}; expected one of {list(CAMERA_PRESETS)}")

    pts = _collect_points(joints_per_frame)
    look = (float(look_at[0]), float(look_at[1]), float(look_at[2]))

    if pts.shape[0] == 0:
        return look, 3.2

    # Binary search: larger distance → smaller on-screen character
    lo, hi = float(dist_min), float(dist_max)
    if not _fits_canvas(pts, views_u, look, hi, size=size, margin=margin):
        # Still too tight at max — return max (best effort)
        return look, hi

    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if _fits_canvas(pts, views_u, look, mid, size=size, margin=margin):
            hi = mid
        else:
            lo = mid
    return look, float(hi)
