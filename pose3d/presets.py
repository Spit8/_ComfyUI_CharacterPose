"""Action presets as joint Euler angles (degrees) for the kinematic skeleton.

Rest pose (bone offsets): arms hang down along -Y.
Swing axes (character faces +Z):
  - shoulder rx: arm forward (−) / back (+)
  - elbow rx: forearm bend (positive curls wrist toward shoulder)
  - hip rx: thigh forward (−) / back (+)
  - knee rx: shin bend (positive)
"""

from __future__ import annotations

from copy import deepcopy


def _a(base: dict | None = None, **kwargs: tuple[float, float, float]) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = dict(base or {})
    out.update(kwargs)
    return out


# Neutral standing — arms relaxed at sides (bone offsets already hang down)
_REST = _a(
    spine=(0, 0, 0),
    r_shoulder=(0, 0, 0),
    l_shoulder=(0, 0, 0),
    r_elbow=(8, 0, 0),
    l_elbow=(8, 0, 0),
)

ACTION_PRESETS: dict[str, dict[str, tuple[float, float, float]]] = {
    "idle": _a(
        _REST,
        spine=(2, 0, 0),
        r_shoulder=(5, 0, 4),
        l_shoulder=(5, 0, -4),
        r_elbow=(12, 0, 0),
        l_elbow=(12, 0, 0),
    ),
    # Walk: opposite arm/leg. Arms stay low with moderate forward/back swing.
    "walk_01": _a(
        _REST,
        spine=(0, 6, 0),
        r_hip=(-28, 0, 0),
        r_knee=(22, 0, 0),
        l_hip=(20, 0, 0),
        l_knee=(10, 0, 0),
        r_shoulder=(18, 0, 0),   # right arm back
        r_elbow=(25, 0, 0),
        l_shoulder=(-22, 0, 0),  # left arm forward
        l_elbow=(30, 0, 0),
    ),
    "walk_02": _a(
        _REST,
        r_hip=(6, 0, 0),
        r_knee=(8, 0, 0),
        l_hip=(-6, 0, 0),
        l_knee=(8, 0, 0),
        r_shoulder=(4, 0, 0),
        l_shoulder=(-4, 0, 0),
        r_elbow=(15, 0, 0),
        l_elbow=(15, 0, 0),
    ),
    "walk_03": _a(
        _REST,
        spine=(0, -6, 0),
        r_hip=(20, 0, 0),
        r_knee=(10, 0, 0),
        l_hip=(-28, 0, 0),
        l_knee=(22, 0, 0),
        r_shoulder=(-22, 0, 0),  # right arm forward
        r_elbow=(30, 0, 0),
        l_shoulder=(18, 0, 0),   # left arm back
        l_elbow=(25, 0, 0),
    ),
    "walk_04": _a(
        _REST,
        r_hip=(-6, 0, 0),
        r_knee=(8, 0, 0),
        l_hip=(6, 0, 0),
        l_knee=(8, 0, 0),
        r_shoulder=(-4, 0, 0),
        l_shoulder=(4, 0, 0),
        r_elbow=(15, 0, 0),
        l_elbow=(15, 0, 0),
    ),
    "run_01": _a(
        _REST,
        spine=(10, 10, 0),
        r_hip=(-42, 0, 0),
        r_knee=(48, 0, 0),
        l_hip=(32, 0, 0),
        l_knee=(18, 0, 0),
        r_shoulder=(35, 0, 0),
        r_elbow=(55, 0, 0),
        l_shoulder=(-40, 0, 0),
        l_elbow=(60, 0, 0),
    ),
    "run_02": _a(
        _REST,
        spine=(10, 0, 0),
        r_hip=(8, 0, 0),
        r_knee=(25, 0, 0),
        l_hip=(-8, 0, 0),
        l_knee=(25, 0, 0),
        r_shoulder=(8, 0, 0),
        l_shoulder=(-8, 0, 0),
        r_elbow=(40, 0, 0),
        l_elbow=(40, 0, 0),
    ),
    "run_03": _a(
        _REST,
        spine=(10, -10, 0),
        r_hip=(32, 0, 0),
        r_knee=(18, 0, 0),
        l_hip=(-42, 0, 0),
        l_knee=(48, 0, 0),
        r_shoulder=(-40, 0, 0),
        r_elbow=(60, 0, 0),
        l_shoulder=(35, 0, 0),
        l_elbow=(55, 0, 0),
    ),
    "run_04": _a(
        _REST,
        spine=(10, 0, 0),
        r_hip=(-8, 0, 0),
        r_knee=(25, 0, 0),
        l_hip=(8, 0, 0),
        l_knee=(25, 0, 0),
        r_shoulder=(-8, 0, 0),
        l_shoulder=(8, 0, 0),
        r_elbow=(40, 0, 0),
        l_elbow=(40, 0, 0),
    ),
    "jump": _a(
        _REST,
        spine=(-4, 0, 0),
        r_hip=(-15, 0, 10),
        r_knee=(70, 0, 0),
        l_hip=(-15, 0, -10),
        l_knee=(70, 0, 0),
        # Arms raised for jump (intentional)
        r_shoulder=(-70, 0, -25),
        r_elbow=(20, 0, 0),
        l_shoulder=(-70, 0, 25),
        l_elbow=(20, 0, 0),
    ),
    # Guard: both fists up near the face (not hanging at the hips)
    "fight_01": _a(
        _REST,
        spine=(0, -12, 0),
        r_hip=(5, 0, 10),
        l_hip=(5, 0, -10),
        r_shoulder=(-95, -25, -45),
        r_elbow=(105, 15, 0),
        l_shoulder=(-90, 25, 45),
        l_elbow=(105, -15, 0),
    ),
    # Punch: lead fist forward/high, rear hand stays in guard
    "fight_02": _a(
        _REST,
        spine=(0, -18, 0),
        r_hip=(8, 0, 12),
        l_hip=(5, 0, -8),
        r_shoulder=(-95, -60, -20),
        r_elbow=(25, 0, 0),
        l_shoulder=(-85, 30, 40),
        l_elbow=(100, 0, 0),
    ),
    "work_01": _a(
        _REST,
        spine=(12, 8, 0),
        r_shoulder=(-95, -10, -15),  # tool raised
        r_elbow=(25, 0, 0),
        l_shoulder=(-25, 5, 5),
        l_elbow=(35, 0, 0),
    ),
    "work_02": _a(
        _REST,
        spine=(18, 8, 0),
        r_shoulder=(35, -5, -10),    # tool down / strike
        r_elbow=(40, 0, 0),
        l_shoulder=(10, 5, 5),
        l_elbow=(30, 0, 0),
    ),
    "cast": _a(
        _REST,
        spine=(-4, 0, 0),
        r_shoulder=(-110, -25, -10),  # casting arm up
        r_elbow=(35, 0, 0),
        l_shoulder=(-15, 15, 5),
        l_elbow=(40, 0, 0),
        r_hip=(5, 0, 6),
        l_hip=(5, 0, -6),
    ),
    "ride_idle": _a(
        _REST,
        r_hip=(-75, 0, 15),
        r_knee=(95, 0, 0),
        l_hip=(-75, 0, -15),
        l_knee=(95, 0, 0),
        spine=(6, 0, 0),
        r_shoulder=(10, 0, -5),
        l_shoulder=(10, 0, 5),
        r_elbow=(45, 0, 0),
        l_elbow=(45, 0, 0),
    ),
}


def list_action_presets() -> list[str]:
    return sorted(ACTION_PRESETS.keys())


def get_action_angles(name: str) -> dict[str, tuple[float, float, float]]:
    key = name.strip().lower()
    if key not in ACTION_PRESETS:
        raise KeyError(f"Unknown action preset: {name}. Available: {list_action_presets()}")
    return deepcopy(ACTION_PRESETS[key])
