"""Lightweight 3D kinematic pose composer → OpenPose COCO-18 projection."""

from .camera import CAMERA_PRESETS, camera_matrix, list_camera_presets
from .from_prompt import PosePromptError, angles_from_text_prompt
from .presets import ACTION_PRESETS, list_action_presets
from .project import compose_pose, project_joints
from .props import PROP_LIBRARY, list_props, prop_hint_text, resolve_props
from .skeleton import JOINT_NAMES, forward_kinematics

__all__ = [
    "ACTION_PRESETS",
    "CAMERA_PRESETS",
    "JOINT_NAMES",
    "PROP_LIBRARY",
    "PosePromptError",
    "angles_from_text_prompt",
    "camera_matrix",
    "compose_pose",
    "forward_kinematics",
    "list_action_presets",
    "list_camera_presets",
    "list_props",
    "project_joints",
    "prop_hint_text",
    "resolve_props",
]
