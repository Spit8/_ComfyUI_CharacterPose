"""Generate built-in .pose library for RPG sprite sheets.

Orientations (isometric-style, facing to the right of screen):
  se  South-East — 3/4 front-right (toward camera + right)
  ne  North-East — 3/4 back-right  (away from camera + right)

Actions:
  idle, walk (4), run (4), jump, fight (2), work (2)

Geometric warp only remaps same-view pixels. For SE↔NE or front↔side,
use generative edit (Flux.2 Klein / Qwen-Edit), not TPS.
"""

from __future__ import annotations

from pathlib import Path

# Allow `python poses/_generate_poses.py` from package root or poses/
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from formats.pose_io import make_pose, save_pose  # noqa: E402

W, H = 1024, 1024
CX, CY = 512.0, 420.0
S = 280.0


def clone(kps: list[list[float]]) -> list[list[float]]:
    return [list(p) for p in kps]


def offset_limb(kps: list[list[float]], idx: int, dx: float, dy: float) -> None:
    x, y, c = kps[idx]
    if c > 0:
        kps[idx] = [x + dx, y + dy, c]


def lift(kps: list[list[float]], dy: float) -> None:
    for i in range(18):
        if kps[i][2] > 0:
            kps[i][1] += dy


# ---------------------------------------------------------------------------
# Bases — OpenPose COCO-18
# 0 nose, 1 neck, 2 r_sh, 3 r_el, 4 r_wr, 5 l_sh, 6 l_el, 7 l_wr,
# 8 r_hip, 9 r_knee, 10 r_ankle, 11 l_hip, 12 l_knee, 13 l_ankle,
# 14 r_eye, 15 l_eye, 16 r_ear, 17 l_ear
# ---------------------------------------------------------------------------


def base_idle_front() -> list[list[float]]:
    cx, cy, s = CX, CY, S
    return [
        [cx, cy - s * 0.55, 1.0],
        [cx, cy - s * 0.32, 1.0],
        [cx + s * 0.22, cy - s * 0.32, 1.0],
        [cx + s * 0.28, cy + s * 0.05, 1.0],
        [cx + s * 0.30, cy + s * 0.38, 1.0],
        [cx - s * 0.22, cy - s * 0.32, 1.0],
        [cx - s * 0.28, cy + s * 0.05, 1.0],
        [cx - s * 0.30, cy + s * 0.38, 1.0],
        [cx + s * 0.12, cy + s * 0.20, 1.0],
        [cx + s * 0.14, cy + s * 0.58, 1.0],
        [cx + s * 0.14, cy + s * 0.98, 1.0],
        [cx - s * 0.12, cy + s * 0.20, 1.0],
        [cx - s * 0.14, cy + s * 0.58, 1.0],
        [cx - s * 0.14, cy + s * 0.98, 1.0],
        [cx + s * 0.08, cy - s * 0.58, 1.0],
        [cx - s * 0.08, cy - s * 0.58, 1.0],
        [cx + s * 0.14, cy - s * 0.54, 1.0],
        [cx - s * 0.14, cy - s * 0.54, 1.0],
    ]


def base_idle_se() -> list[list[float]]:
    """South-East: 3/4 front-right (face + torso toward camera-right)."""
    cx, cy, s = CX, CY, S
    return [
        [cx + s * 0.20, cy - s * 0.52, 1.0],  # nose (forward-right)
        [cx + s * 0.06, cy - s * 0.30, 1.0],  # neck
        [cx + s * 0.16, cy - s * 0.28, 1.0],  # r_shoulder (near)
        [cx + s * 0.24, cy + s * 0.02, 1.0],  # r_elbow
        [cx + s * 0.28, cy + s * 0.34, 1.0],  # r_wrist
        [cx - s * 0.10, cy - s * 0.26, 1.0],  # l_shoulder (far, still visible)
        [cx - s * 0.06, cy + s * 0.06, 1.0],  # l_elbow
        [cx + s * 0.00, cy + s * 0.36, 1.0],  # l_wrist
        [cx + s * 0.12, cy + s * 0.22, 1.0],  # r_hip
        [cx + s * 0.16, cy + s * 0.58, 1.0],  # r_knee
        [cx + s * 0.18, cy + s * 0.98, 1.0],  # r_ankle
        [cx - s * 0.04, cy + s * 0.22, 1.0],  # l_hip
        [cx - s * 0.02, cy + s * 0.58, 1.0],  # l_knee
        [cx + s * 0.00, cy + s * 0.98, 1.0],  # l_ankle
        [cx + s * 0.26, cy - s * 0.55, 1.0],  # r_eye
        [cx + s * 0.14, cy - s * 0.55, 1.0],  # l_eye
        [cx + s * 0.30, cy - s * 0.50, 1.0],  # r_ear
        [cx + s * 0.04, cy - s * 0.50, 0.5],  # l_ear (partial)
    ]


def base_idle_ne() -> list[list[float]]:
    """North-East: 3/4 back-right (back toward camera, facing up-right)."""
    cx, cy, s = CX, CY, S
    return [
        [cx + s * 0.10, cy - s * 0.50, 0.7],  # nose (mostly away)
        [cx + s * 0.04, cy - s * 0.30, 1.0],  # neck
        [cx - s * 0.08, cy - s * 0.28, 1.0],  # r_shoulder (far / back-right naming: camera-left is character right-back)
        [cx - s * 0.14, cy + s * 0.00, 1.0],  # r_elbow
        [cx - s * 0.10, cy + s * 0.32, 1.0],  # r_wrist
        [cx + s * 0.18, cy - s * 0.26, 1.0],  # l_shoulder (near, back-left from char = screen-right)
        [cx + s * 0.22, cy + s * 0.04, 1.0],  # l_elbow
        [cx + s * 0.20, cy + s * 0.34, 1.0],  # l_wrist
        [cx - s * 0.02, cy + s * 0.22, 1.0],  # r_hip
        [cx - s * 0.04, cy + s * 0.58, 1.0],  # r_knee
        [cx - s * 0.02, cy + s * 0.98, 1.0],  # r_ankle
        [cx + s * 0.12, cy + s * 0.22, 1.0],  # l_hip
        [cx + s * 0.14, cy + s * 0.58, 1.0],  # l_knee
        [cx + s * 0.16, cy + s * 0.98, 1.0],  # l_ankle
        [cx + s * 0.06, cy - s * 0.54, 0.4],  # r_eye
        [cx + s * 0.16, cy - s * 0.54, 0.6],  # l_eye
        [cx - s * 0.02, cy - s * 0.48, 0.8],  # r_ear (more visible from back)
        [cx + s * 0.22, cy - s * 0.48, 1.0],  # l_ear
    ]


def _base(orient: str) -> list[list[float]]:
    if orient == "se":
        return base_idle_se()
    if orient == "ne":
        return base_idle_ne()
    return base_idle_front()


def walk_cycle(orient: str, frame: int) -> list[list[float]]:
    k = clone(_base(orient))
    phase = [(-1, 1), (1, -1), (-1, 1), (1, -1)][frame % 4]
    amp = 70.0 if orient != "front" else 55.0
    if orient in ("se", "ne"):
        # Profile-ish: swing mostly along X
        fx = 1.0 if orient == "se" else -0.3  # NE less lateral stride on screen
        offset_limb(k, 9, phase[0] * amp * 0.35 * fx, phase[0] * amp * 0.12)
        offset_limb(k, 10, phase[0] * amp * 0.7 * fx, phase[0] * amp * 0.05)
        offset_limb(k, 12, phase[1] * amp * 0.35 * fx, phase[1] * amp * 0.12)
        offset_limb(k, 13, phase[1] * amp * 0.7 * fx, phase[1] * amp * 0.05)
        offset_limb(k, 3, phase[1] * 18, phase[1] * 12)
        offset_limb(k, 4, phase[1] * 35, phase[1] * 28)
        offset_limb(k, 6, phase[0] * 12, phase[0] * 10)
        offset_limb(k, 7, phase[0] * 28, phase[0] * 25)
    else:
        offset_limb(k, 9, phase[0] * 10, phase[0] * amp * 0.15)
        offset_limb(k, 10, phase[0] * 25, phase[0] * amp)
        offset_limb(k, 12, phase[1] * 10, phase[1] * amp * 0.15)
        offset_limb(k, 13, phase[1] * 25, phase[1] * amp)
        offset_limb(k, 3, phase[1] * 8, phase[1] * 20)
        offset_limb(k, 4, phase[1] * 20, phase[1] * 45)
        offset_limb(k, 6, phase[0] * 8, phase[0] * 20)
        offset_limb(k, 7, phase[0] * 20, phase[0] * 45)
    return k


def run_cycle(orient: str, frame: int) -> list[list[float]]:
    k = clone(_base(orient))
    phase = [(-1, 1), (1, -1), (-1, 1), (1, -1)][frame % 4]
    amp = 90.0
    lift(k, -30)
    offset_limb(k, 1, 0, -15)
    fx = 1.0 if orient != "ne" else -0.35
    offset_limb(k, 9, phase[0] * 20 * fx, phase[0] * amp * 0.2)
    offset_limb(k, 10, phase[0] * 45 * fx, phase[0] * amp)
    offset_limb(k, 12, phase[1] * 20 * fx, phase[1] * amp * 0.2)
    offset_limb(k, 13, phase[1] * 45 * fx, phase[1] * amp)
    offset_limb(k, 3, phase[1] * 15, phase[1] * 35)
    offset_limb(k, 4, phase[1] * 40, phase[1] * 70)
    offset_limb(k, 6, phase[0] * 15, phase[0] * 35)
    offset_limb(k, 7, phase[0] * 40, phase[0] * 70)
    return k


def jump_pose(orient: str) -> list[list[float]]:
    k = clone(_base(orient))
    lift(k, -120)
    offset_limb(k, 3, 35, -55)
    offset_limb(k, 4, 60, -35)
    offset_limb(k, 6, -30, -55)
    offset_limb(k, 7, -55, -35)
    offset_limb(k, 9, 12, -25)
    offset_limb(k, 10, 18, -8)
    offset_limb(k, 12, -12, -25)
    offset_limb(k, 13, -18, -8)
    return k


def fight_pose(orient: str, frame: int) -> list[list[float]]:
    """frame 0 = ready stance, frame 1 = strike."""
    k = clone(_base(orient))
    # Wider stance
    offset_limb(k, 8, 18, 5)
    offset_limb(k, 11, -12, 5)
    offset_limb(k, 10, 25, 0)
    offset_limb(k, 13, -20, 0)
    if frame == 0:
        # Guard: lead arm up
        offset_limb(k, 2, 15, -15)
        offset_limb(k, 3, 50, -50)
        offset_limb(k, 4, 70, -30)
        offset_limb(k, 5, -8, 8)
        offset_limb(k, 6, -25, 30)
        offset_limb(k, 7, -15, 55)
    else:
        # Strike forward (SE = +x, NE = slightly -x on screen for rear arm)
        strike = 1.0 if orient != "ne" else -0.6
        offset_limb(k, 2, 25 * strike, -20)
        offset_limb(k, 3, 110 * strike, -45)
        offset_limb(k, 4, 180 * strike, -15)
        offset_limb(k, 5, -15, 15)
        offset_limb(k, 6, -35, 45)
        offset_limb(k, 7, -25, 85)
        offset_limb(k, 8, 30 * strike, 0)
    return k


def work_pose(orient: str, frame: int) -> list[list[float]]:
    """frame 0 = tool up, frame 1 = tool down (chop / hammer)."""
    k = clone(_base(orient))
    # Slight forward lean
    offset_limb(k, 0, 8 if orient != "ne" else -4, 15)
    offset_limb(k, 1, 5 if orient != "ne" else -2, 10)
    if frame == 0:
        offset_limb(k, 3, 20, -90)
        offset_limb(k, 4, 30, -130)
        offset_limb(k, 6, -10, -40)
        offset_limb(k, 7, 5, -70)
    else:
        offset_limb(k, 3, 40, 40)
        offset_limb(k, 4, 55, 90)
        offset_limb(k, 6, 10, 20)
        offset_limb(k, 7, 25, 50)
        lift(k, 15)
    return k


def main() -> None:
    out = Path(__file__).resolve().parent
    poses: dict[str, tuple[list[list[float]], dict]] = {}

    # Legacy front set (kept for older workflows)
    poses["idle.pose"] = (base_idle_front(), {"orientation": "front", "action": "idle"})
    for i in range(4):
        poses[f"walk_{i + 1:02d}.pose"] = (walk_cycle("front", i), {"orientation": "front", "action": "walk", "frame": i})
        poses[f"run_{i + 1:02d}.pose"] = (run_cycle("front", i), {"orientation": "front", "action": "run", "frame": i})
    poses["jump.pose"] = (jump_pose("front"), {"orientation": "front", "action": "jump"})
    poses["attack.pose"] = (fight_pose("front", 1), {"orientation": "front", "action": "fight", "frame": 1})
    poses["hurt.pose"] = (base_idle_front(), {"orientation": "front", "action": "hurt"})  # placeholder
    # simple hurt tweak
    hk = clone(base_idle_front())
    for i in range(18):
        if hk[i][2] > 0:
            hk[i][0] -= 30
            hk[i][1] += 20
    poses["hurt.pose"] = (hk, {"orientation": "front", "action": "hurt"})
    poses["death.pose"] = (base_idle_front(), {"orientation": "front", "action": "death"})

    # Death reclined (keep previous layout via inline)
    cy, cx, s = 720.0, CX, S
    poses["death.pose"] = (
        [
            [cx + s * 0.55, cy - s * 0.05, 1.0],
            [cx + s * 0.30, cy, 1.0],
            [cx + s * 0.25, cy - s * 0.18, 1.0],
            [cx + s * 0.05, cy - s * 0.35, 1.0],
            [cx - s * 0.15, cy - s * 0.40, 1.0],
            [cx + s * 0.25, cy + s * 0.18, 1.0],
            [cx + s * 0.05, cy + s * 0.30, 1.0],
            [cx - s * 0.10, cy + s * 0.35, 1.0],
            [cx - s * 0.05, cy - s * 0.12, 1.0],
            [cx - s * 0.40, cy - s * 0.18, 1.0],
            [cx - s * 0.75, cy - s * 0.20, 1.0],
            [cx - s * 0.05, cy + s * 0.12, 1.0],
            [cx - s * 0.40, cy + s * 0.18, 1.0],
            [cx - s * 0.75, cy + s * 0.20, 1.0],
            [cx + s * 0.58, cy - s * 0.12, 1.0],
            [cx + s * 0.58, cy + s * 0.02, 1.0],
            [cx + s * 0.52, cy - s * 0.18, 1.0],
            [cx + s * 0.52, cy + s * 0.08, 1.0],
        ],
        {"orientation": "front", "action": "death"},
    )

    # New SE / NE RPG set
    for orient, label in (("se", "south_east"), ("ne", "north_east")):
        poses[f"idle_{orient}.pose"] = (_base(orient), {"orientation": label, "action": "idle"})
        for i in range(4):
            poses[f"walk_{orient}_{i + 1:02d}.pose"] = (
                walk_cycle(orient, i),
                {"orientation": label, "action": "walk", "frame": i},
            )
            poses[f"run_{orient}_{i + 1:02d}.pose"] = (
                run_cycle(orient, i),
                {"orientation": label, "action": "run", "frame": i},
            )
        poses[f"jump_{orient}.pose"] = (jump_pose(orient), {"orientation": label, "action": "jump"})
        poses[f"fight_{orient}_01.pose"] = (fight_pose(orient, 0), {"orientation": label, "action": "fight", "frame": 0})
        poses[f"fight_{orient}_02.pose"] = (fight_pose(orient, 1), {"orientation": label, "action": "fight", "frame": 1})
        poses[f"work_{orient}_01.pose"] = (work_pose(orient, 0), {"orientation": label, "action": "work", "frame": 0})
        poses[f"work_{orient}_02.pose"] = (work_pose(orient, 1), {"orientation": label, "action": "work", "frame": 1})

    # Aliases for older workflows
    poses["idle_side.pose"] = (clone(poses["idle_se.pose"][0]), {"orientation": "south_east", "action": "idle", "alias_of": "idle_se"})
    for i in range(4):
        src = f"walk_se_{i + 1:02d}.pose"
        poses[f"walk_side_{i + 1:02d}.pose"] = (
            clone(poses[src][0]),
            {"orientation": "south_east", "action": "walk", "frame": i, "alias_of": src},
        )

    for name, (kps, meta) in poses.items():
        pose = make_pose(kps, width=W, height=H, name=name.replace(".pose", ""), metadata=meta)
        save_pose(pose, out / name)
        print(f"wrote {name}")

    print(f"Total: {len(poses)} poses")


if __name__ == "__main__":
    main()
