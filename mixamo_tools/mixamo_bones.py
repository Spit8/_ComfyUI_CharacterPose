"""Mixamo bone name → OpenPose COCO-18 joint mapping."""

from __future__ import annotations

# Canonical COCO-18 joint names (same order as formats/pose_io.KEYPOINT_NAMES)
COCO18_NAMES = [
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

# Candidate Mixamo / generic names per logical bone (first match wins)
MIXAMO_ALIASES: dict[str, tuple[str, ...]] = {
    "hips": ("mixamorig:Hips", "mixamorig_Hips", "Hips", "hip", "Pelvis"),
    "spine": ("mixamorig:Spine", "mixamorig_Spine", "Spine"),
    "spine1": ("mixamorig:Spine1", "mixamorig_Spine1", "Spine1"),
    "spine2": ("mixamorig:Spine2", "mixamorig_Spine2", "Spine2"),
    "neck": ("mixamorig:Neck", "mixamorig_Neck", "Neck"),
    "head": ("mixamorig:Head", "mixamorig_Head", "Head", "HeadTop_End"),
    "l_shoulder": (
        "mixamorig:LeftArm",
        "mixamorig_LeftArm",
        "LeftArm",
        "mixamorig:LeftShoulder",
        "LeftShoulder",
    ),
    "l_elbow": ("mixamorig:LeftForeArm", "mixamorig_LeftForeArm", "LeftForeArm"),
    "l_wrist": ("mixamorig:LeftHand", "mixamorig_LeftHand", "LeftHand"),
    "r_shoulder": (
        "mixamorig:RightArm",
        "mixamorig_RightArm",
        "RightArm",
        "mixamorig:RightShoulder",
        "RightShoulder",
    ),
    "r_elbow": ("mixamorig:RightForeArm", "mixamorig_RightForeArm", "RightForeArm"),
    "r_wrist": ("mixamorig:RightHand", "mixamorig_RightHand", "RightHand"),
    "l_hip": ("mixamorig:LeftUpLeg", "mixamorig_LeftUpLeg", "LeftUpLeg"),
    "l_knee": ("mixamorig:LeftLeg", "mixamorig_LeftLeg", "LeftLeg"),
    "l_ankle": ("mixamorig:LeftFoot", "mixamorig_LeftFoot", "LeftFoot"),
    "r_hip": ("mixamorig:RightUpLeg", "mixamorig_RightUpLeg", "RightUpLeg"),
    "r_knee": ("mixamorig:RightLeg", "mixamorig_RightLeg", "RightLeg"),
    "r_ankle": ("mixamorig:RightFoot", "mixamorig_RightFoot", "RightFoot"),
    "l_eye": ("mixamorig:LeftEye", "mixamorig_LeftEye", "LeftEye"),
    "r_eye": ("mixamorig:RightEye", "mixamorig_RightEye", "RightEye"),
}


def normalize_bone_name(name: str) -> str:
    """Strip namespace prefixes for fuzzy matching."""
    n = name.strip()
    for prefix in ("mixamorig:", "mixamorig_", "Armature|", "Armature/"):
        if n.startswith(prefix):
            n = n[len(prefix) :]
    return n.lower()


def resolve_bone_name(available: dict[str, object], logical: str) -> str | None:
    """Find an actual node name in `available` for a logical Mixamo bone."""
    aliases = MIXAMO_ALIASES.get(logical, ())
    # Exact first
    for a in aliases:
        if a in available:
            return a
    # Case-insensitive / stripped
    norm_map = {normalize_bone_name(k): k for k in available}
    for a in aliases:
        key = normalize_bone_name(a)
        if key in norm_map:
            return norm_map[key]
    # Suffix match (e.g. "Character:mixamorig:Hips")
    for a in aliases:
        suf = normalize_bone_name(a)
        for nk, original in norm_map.items():
            if nk.endswith(suf) or nk.split(":")[-1] == suf.split(":")[-1]:
                return original
    return None


# Local offsets from head (character space, Y-up, face +Z) for face landmarks
HEAD_LOCAL_OFFSETS = {
    "nose": (0.0, 0.02, 0.10),
    "r_eye": (0.04, 0.04, 0.08),
    "l_eye": (-0.04, 0.04, 0.08),
    "r_ear": (0.08, 0.02, 0.02),
    "l_ear": (-0.08, 0.02, 0.02),
}
