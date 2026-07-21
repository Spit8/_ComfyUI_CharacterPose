"""Action presets as joint Euler angles (degrees) for the kinematic skeleton."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# Each preset: bone → (rx, ry, rz) relative to rest T-pose-ish A-pose
# Rest is arms slightly down from T.

def _a(base: dict | None = None, **kwargs: tuple[float, float, float]) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = dict(base or {})
    out.update(kwargs)
    return out


# Slight A-pose rest (arms down ~20°)
_REST = _a(
    r_shoulder=(0, 0, 20),
    l_shoulder=(0, 0, -20),
)

ACTION_PRESETS: dict[str, dict[str, tuple[float, float, float]]] = {
    "idle": _a(
        _REST,
        spine=(2, 0, 0),
        r_elbow=(0, 0, 15),
        l_elbow=(0, 0, -15),
    ),
    "walk_01": _a(
        _REST,
        r_hip=(-28, 0, 0),
        r_knee=(18, 0, 0),
        l_hip=(22, 0, 0),
        l_knee=(8, 0, 0),
        r_shoulder=(0, 0, 25),
        r_elbow=(0, 0, 25),
        l_shoulder=(15, 0, -25),
        l_elbow=(0, 0, -20),
        spine=(0, 8, 0),
    ),
    "walk_02": _a(
        _REST,
        r_hip=(8, 0, 0),
        r_knee=(5, 0, 0),
        l_hip=(-8, 0, 0),
        l_knee=(5, 0, 0),
        r_shoulder=(5, 0, 22),
        l_shoulder=(5, 0, -22),
    ),
    "walk_03": _a(
        _REST,
        r_hip=(22, 0, 0),
        r_knee=(8, 0, 0),
        l_hip=(-28, 0, 0),
        l_knee=(18, 0, 0),
        r_shoulder=(15, 0, 25),
        r_elbow=(0, 0, 20),
        l_shoulder=(0, 0, -25),
        l_elbow=(0, 0, -25),
        spine=(0, -8, 0),
    ),
    "walk_04": _a(
        _REST,
        r_hip=(-8, 0, 0),
        r_knee=(5, 0, 0),
        l_hip=(8, 0, 0),
        l_knee=(5, 0, 0),
        r_shoulder=(5, 0, 22),
        l_shoulder=(5, 0, -22),
    ),
    "run_01": _a(
        _REST,
        spine=(8, 12, 0),
        r_hip=(-45, 0, 0),
        r_knee=(40, 0, 0),
        l_hip=(35, 0, 0),
        l_knee=(15, 0, 0),
        r_shoulder=(-20, 0, 35),
        r_elbow=(0, 0, 50),
        l_shoulder=(25, 0, -35),
        l_elbow=(0, 0, -45),
    ),
    "run_02": _a(
        _REST,
        spine=(8, 0, 0),
        r_hip=(10, 0, 0),
        r_knee=(20, 0, 0),
        l_hip=(-10, 0, 0),
        l_knee=(20, 0, 0),
        r_shoulder=(0, 0, 30),
        l_shoulder=(0, 0, -30),
    ),
    "run_03": _a(
        _REST,
        spine=(8, -12, 0),
        r_hip=(35, 0, 0),
        r_knee=(15, 0, 0),
        l_hip=(-45, 0, 0),
        l_knee=(40, 0, 0),
        r_shoulder=(25, 0, 35),
        r_elbow=(0, 0, 45),
        l_shoulder=(-20, 0, -35),
        l_elbow=(0, 0, -50),
    ),
    "run_04": _a(
        _REST,
        spine=(8, 0, 0),
        r_hip=(-10, 0, 0),
        r_knee=(20, 0, 0),
        l_hip=(10, 0, 0),
        l_knee=(20, 0, 0),
        r_shoulder=(0, 0, 30),
        l_shoulder=(0, 0, -30),
    ),
    "jump": _a(
        _REST,
        spine=(-5, 0, 0),
        r_hip=(-20, 0, 15),
        r_knee=(55, 0, 0),
        l_hip=(-20, 0, -15),
        l_knee=(55, 0, 0),
        r_shoulder=(-50, 0, 40),
        r_elbow=(0, 0, 30),
        l_shoulder=(-50, 0, -40),
        l_elbow=(0, 0, -30),
    ),
    "fight_01": _a(
        _REST,
        spine=(0, -15, 0),
        r_hip=(5, 0, 12),
        l_hip=(5, 0, -12),
        r_shoulder=(-40, -20, 30),
        r_elbow=(0, 0, 70),
        l_shoulder=(10, 20, -40),
        l_elbow=(0, 0, -50),
    ),
    "fight_02": _a(
        _REST,
        spine=(0, -25, 0),
        r_hip=(8, 0, 15),
        l_hip=(5, 0, -10),
        r_shoulder=(-10, -60, 20),
        r_elbow=(0, 0, 20),
        l_shoulder=(20, 30, -50),
        l_elbow=(0, 0, -60),
    ),
    "work_01": _a(
        _REST,
        spine=(15, 10, 0),
        r_shoulder=(-90, -20, 20),
        r_elbow=(0, 0, 20),
        l_shoulder=(-40, 10, -20),
        l_elbow=(0, 0, -30),
    ),
    "work_02": _a(
        _REST,
        spine=(20, 10, 0),
        r_shoulder=(40, -10, 20),
        r_elbow=(0, 0, 40),
        l_shoulder=(10, 10, -20),
        l_elbow=(0, 0, -25),
    ),
    "cast": _a(
        _REST,
        spine=(-5, 0, 0),
        r_shoulder=(-100, -30, 10),
        r_elbow=(0, 0, 30),
        l_shoulder=(-20, 20, -30),
        l_elbow=(0, 0, -40),
        r_hip=(5, 0, 8),
        l_hip=(5, 0, -8),
    ),
    "ride_idle": _a(
        _REST,
        r_hip=(-70, 0, 20),
        r_knee=(90, 0, 0),
        l_hip=(-70, 0, -20),
        l_knee=(90, 0, 0),
        spine=(5, 0, 0),
        r_shoulder=(10, 0, 25),
        l_shoulder=(10, 0, -25),
        r_elbow=(0, 0, 40),
        l_elbow=(0, 0, -40),
    ),
}


def list_action_presets() -> list[str]:
    return sorted(ACTION_PRESETS.keys())


def get_action_angles(name: str) -> dict[str, tuple[float, float, float]]:
    key = name.strip().lower()
    if key not in ACTION_PRESETS:
        raise KeyError(f"Unknown action preset: {name}. Available: {list_action_presets()}")
    return deepcopy(ACTION_PRESETS[key])
