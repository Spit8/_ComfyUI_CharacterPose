"""3D pose composer + pose-transfer prep facade nodes."""

from __future__ import annotations

import numpy as np
import torch

from ..formats.pose_io import make_pose, scale_pose
from ..pose3d.camera import CAMERA_PRESETS, list_camera_presets
from ..pose3d.from_prompt import PosePromptError, angles_from_text_prompt
from ..pose3d.presets import list_action_presets
from ..pose3d.project import compose_pose
from ..pose3d.props import list_props
from ..utils import content_bbox, draw_openpose, np_to_tensor, tensor_to_np
from .caption import build_edit_prompt, describe_character
from .warp import fit_pose_to_source

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


PROP_COLOR = (0, 220, 255)  # RGB cyan-ish for props (distinct from OpenPose)
POSE_SOURCES = ["library", "text"]
_LLM_DEFAULT_BASE = "https://api.openai.com/v1"
_LLM_DEFAULT_MODEL = "gpt-4o-mini"


def _llm_input_types() -> dict:
    return {
        "pose_prompt": ("STRING", {"default": "", "multiline": True}),
        "llm_base_url": ("STRING", {"default": _LLM_DEFAULT_BASE}),
        "llm_model": ("STRING", {"default": _LLM_DEFAULT_MODEL}),
        "llm_api_key": ("STRING", {"default": ""}),
    }


def _resolve_compose_angles(
    pose_source: str,
    action: str,
    pose_prompt: str,
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str,
) -> tuple[str, dict | None]:
    """Return (action_label, joint_angles_or_None for compose_pose)."""
    source = (pose_source or "library").strip().lower()
    if source == "text":
        try:
            angles = angles_from_text_prompt(
                pose_prompt,
                base_url=llm_base_url or _LLM_DEFAULT_BASE,
                api_key=llm_api_key or None,
                model=llm_model or _LLM_DEFAULT_MODEL,
                seed_action="idle",
            )
        except PosePromptError:
            raise
        except Exception as e:  # pragma: no cover
            raise PosePromptError(f"Text pose failed: {e}") from e
        note = (pose_prompt or "").strip().replace("\n", " ")
        if len(note) > 48:
            note = note[:45] + "..."
        return (f"text:{note}" if note else "text", angles)
    return (action or "idle", None)


def _draw_prop_polylines(
    canvas: np.ndarray,
    polylines: list[np.ndarray],
    *,
    thickness: int | None = None,
) -> np.ndarray:
    """Draw prop polylines onto an RGB uint8 canvas; return prop-only mask (RGB)."""
    h, w = canvas.shape[:2]
    stick = thickness if thickness is not None else max(2, int(round(min(w, h) / 140)))
    mask = np.zeros_like(canvas)

    def _line(img, a, b, color):
        if cv2 is not None:
            cv2.line(img, a, b, color, stick, lineType=cv2.LINE_AA)
        else:
            from PIL import Image, ImageDraw

            pil = Image.fromarray(img)
            draw = ImageDraw.Draw(pil)
            draw.line([a, b], fill=color, width=stick)
            img[:] = np.asarray(pil)

    for poly in polylines:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            x0, y0 = int(round(pts[i, 0])), int(round(pts[i, 1]))
            x1, y1 = int(round(pts[i + 1, 0])), int(round(pts[i + 1, 1]))
            if min(x0, x1) < -100 or min(y0, y1) < -100:
                continue
            if max(x0, x1) > w + 100 or max(y0, y1) > h + 100:
                continue
            _line(canvas, (x0, y0), (x1, y1), PROP_COLOR)
            _line(mask, (x0, y0), (x1, y1), (255, 255, 255))
    return mask


def _extract_source_pose_from_image(
    image, pose_keypoint=None
) -> tuple[dict | None, str]:
    """Extract COCO-18 pose from DWPose JSON / MediaPipe.

    Returns (pose_or_None, backend). Never invents a fake T-pose — callers
    should skip skeleton overlay when pose is None.
    """
    from ..utils import (
        estimate_keypoints_mediapipe,
        parse_dwpose_json_keypoints,
        try_dwpose_keypoints,
    )

    rgb = tensor_to_np(image)
    h, w = rgb.shape[:2]
    keypoints = None
    backend = "none"

    if pose_keypoint is not None:
        try:
            keypoints = parse_dwpose_json_keypoints(pose_keypoint, w, h)
            if keypoints is not None:
                backend = "dwpose_json"
        except Exception:
            keypoints = None

    if keypoints is None:
        try:
            keypoints = try_dwpose_keypoints(rgb)
            if keypoints is not None:
                backend = "dwpose_detector"
        except Exception:
            keypoints = None

    if keypoints is None:
        try:
            keypoints = estimate_keypoints_mediapipe(rgb)
            if keypoints is not None:
                backend = "mediapipe"
        except Exception:
            keypoints = None

    if keypoints is None:
        return None, backend

    visible = sum(1 for kp in keypoints if kp[2] > 0)
    if visible < 4:
        return None, backend

    return make_pose(keypoints, width=w, height=h, name="source"), backend


def _resize_rgb(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    if rgb.shape[0] == h and rgb.shape[1] == w:
        return rgb
    if cv2 is not None:
        return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    from PIL import Image

    return np.asarray(Image.fromarray(rgb).resize((w, h)), dtype=np.uint8)


def _overlay_dwpose_image(base_rgb: np.ndarray, dwpose_rgb: np.ndarray) -> np.ndarray:
    """Composite DWPreprocessor stick image onto the sprite (pixel-aligned)."""
    h, w = base_rgb.shape[:2]
    skel = _resize_rgb(dwpose_rgb, w, h)
    # DWPose canvas is black + colored sticks — keep bright stick pixels
    stick = skel.max(axis=2) > 24
    if not np.any(stick):
        return base_rgb
    out = base_rgb.astype(np.float32)
    sk = skel.astype(np.float32)
    out[stick] = out[stick] * 0.2 + sk[stick] * 0.8
    return np.clip(out, 0, 255).astype(np.uint8)


def _overlay_skeleton(base_rgb: np.ndarray, pose: dict) -> np.ndarray:
    """Draw OpenPose sticks onto the sprite."""
    h, w = base_rgb.shape[:2]
    skel = draw_openpose(pose, width=w, height=h)
    try:
        from .warp import blend_skeleton_onto_image

        return blend_skeleton_onto_image(
            base_rgb, skel, opacity=0.85, mode="overlay", stick_grow=2
        )
    except Exception:
        stick = skel.max(axis=2) > 12
        out = base_rgb.astype(np.float32)
        out[stick] = out[stick] * 0.25 + skel[stick].astype(np.float32) * 0.75
        return np.clip(out, 0, 255).astype(np.uint8)


def _build_source_pose_preview(
    base_rgb: np.ndarray,
    *,
    src_pose: dict | None,
    dwpose_image=None,
) -> np.ndarray:
    """Sprite + real skeleton only (DWPose image preferred, else keypoints)."""
    if dwpose_image is not None:
        try:
            dw = tensor_to_np(dwpose_image)
            return _overlay_dwpose_image(base_rgb, dw)
        except Exception:
            pass
    if src_pose is not None:
        return _overlay_skeleton(base_rgb, src_pose)
    return base_rgb



class PoseComposer3D:
    """Compose a COCO-18 pose from 3D action presets + orientable camera + props."""

    @classmethod
    def INPUT_TYPES(cls):
        actions = list_action_presets()
        cams = list_camera_presets()
        props = list_props()
        return {
            "required": {
                "pose_source": (POSE_SOURCES, {"default": "library"}),
                "action": (actions, {"default": "idle"}),
                "camera_preset": (cams, {"default": "SE"}),
                "yaw": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 60.0, "step": 1.0}),
                "roll": ("FLOAT", {"default": 0.0, "min": -30.0, "max": 30.0, "step": 1.0}),
                "distance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 8.0, "step": 0.05}),
                "use_manual_camera": ("BOOLEAN", {"default": False}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "prop": (props, {"default": "none"}),
            },
            "optional": {
                "prop2": (props, {"default": "none"}),
                "fov_deg": ("FLOAT", {"default": 35.0, "min": 15.0, "max": 90.0, "step": 1.0}),
                **_llm_input_types(),
            },
        }

    RETURN_TYPES = ("POSE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("pose", "guide", "prop_mask", "prop_hint")
    FUNCTION = "compose"
    CATEGORY = "CharacterPose/Pose"

    def compose(
        self,
        action="idle",
        camera_preset="SE",
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        distance=0.0,
        use_manual_camera=False,
        width=1024,
        height=1024,
        prop="none",
        prop2="none",
        fov_deg=35.0,
        pose_source="library",
        pose_prompt="",
        llm_base_url=_LLM_DEFAULT_BASE,
        llm_model=_LLM_DEFAULT_MODEL,
        llm_api_key="",
    ):
        props = []
        if prop and prop != "none":
            props.append(prop)
        if prop2 and prop2 != "none":
            props.append(prop2)

        cam_kwargs = {}
        if use_manual_camera:
            cam_kwargs = {
                "yaw": float(yaw),
                "pitch": float(pitch),
                "roll": float(roll),
                "distance": float(distance) if distance > 0.1 else None,
            }
        else:
            # Optional fine-tune offsets on top of preset when non-zero
            preset = CAMERA_PRESETS.get(camera_preset.upper(), CAMERA_PRESETS["SE"])
            if abs(yaw) > 1e-3:
                cam_kwargs["yaw"] = preset["yaw"] + float(yaw)
            if abs(pitch) > 1e-3:
                cam_kwargs["pitch"] = preset["pitch"] + float(pitch)
            if abs(roll) > 1e-3:
                cam_kwargs["roll"] = preset["roll"] + float(roll)
            if distance > 0.1:
                cam_kwargs["distance"] = float(distance)

        action_label, joint_angles = _resolve_compose_angles(
            pose_source, action, pose_prompt, llm_base_url, llm_model, llm_api_key
        )
        result = compose_pose(
            action_label,
            camera_preset=camera_preset,
            width=int(width),
            height=int(height),
            props=props,
            fov_deg=float(fov_deg),
            joint_angles=joint_angles,
            **cam_kwargs,
        )
        pose = result["pose"]
        guide = draw_openpose(pose, width=int(width), height=int(height))
        prop_mask = _draw_prop_polylines(guide, result["prop_polylines_2d"])
        return (
            pose,
            np_to_tensor(guide),
            np_to_tensor(prop_mask),
            result["prop_hint"],
        )


class PoseTransferPrep:
    """One-stop prep: caption + 3D compose + optional align → guide + prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        actions = list_action_presets()
        cams = list_camera_presets()
        props = list_props()
        return {
            "required": {
                "image": ("IMAGE",),
                "pose_source": (POSE_SOURCES, {"default": "library"}),
                "action": (actions, {"default": "idle"}),
                "camera_preset": (cams, {"default": "SE"}),
                "prop": (props, {"default": "none"}),
                "align_to_source": ("BOOLEAN", {"default": True}),
                "auto_caption": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Keep pose_keypoint / dwpose_image first so workflow link slots stay stable
                "pose_keypoint": ("POSE_KEYPOINT",),
                "dwpose_image": ("IMAGE",),
                "prop2": (props, {"default": "none"}),
                "yaw": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 60.0, "step": 1.0}),
                "roll": ("FLOAT", {"default": 0.0, "min": -30.0, "max": 30.0, "step": 1.0}),
                "use_manual_camera": ("BOOLEAN", {"default": False}),
                "caption_override": ("STRING", {"default": "", "multiline": True}),
                "extra_prompt": ("STRING", {"default": "", "multiline": True}),
                "source_pose": ("POSE",),
                "width": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "height": ("INT", {"default": 0, "min": 0, "max": 4096}),
                **_llm_input_types(),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "POSE", "IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = (
        "guide",
        "edit_prompt",
        "pose",
        "source_pose",
        "preview_pose",
        "caption",
        "caption_backend",
    )
    FUNCTION = "prep"
    CATEGORY = "CharacterPose/Pose"

    def prep(
        self,
        image,
        action="idle",
        camera_preset="SE",
        prop="none",
        align_to_source=True,
        auto_caption=True,
        prop2="none",
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        use_manual_camera=False,
        caption_override="",
        extra_prompt="",
        source_pose=None,
        pose_keypoint=None,
        dwpose_image=None,
        width=0,
        height=0,
        pose_source="library",
        pose_prompt="",
        llm_base_url=_LLM_DEFAULT_BASE,
        llm_model=_LLM_DEFAULT_MODEL,
        llm_api_key="",
    ):
        rgb = tensor_to_np(image)
        ih, iw = rgb.shape[:2]
        w = int(width) if width and width > 0 else iw
        h = int(height) if height and height > 0 else ih

        props = []
        if prop and prop != "none":
            props.append(prop)
        if prop2 and prop2 != "none":
            props.append(prop2)

        cam_kwargs = {}
        if use_manual_camera:
            cam_kwargs = {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)}
        elif abs(yaw) > 1e-3 or abs(pitch) > 1e-3 or abs(roll) > 1e-3:
            preset = CAMERA_PRESETS.get(camera_preset.upper(), CAMERA_PRESETS["SE"])
            cam_kwargs = {
                "yaw": preset["yaw"] + float(yaw),
                "pitch": preset["pitch"] + float(pitch),
                "roll": preset["roll"] + float(roll),
            }

        action_label, joint_angles = _resolve_compose_angles(
            pose_source, action, pose_prompt, llm_base_url, llm_model, llm_api_key
        )
        result = compose_pose(
            action_label,
            camera_preset=camera_preset,
            width=w,
            height=h,
            props=props,
            joint_angles=joint_angles,
            **cam_kwargs,
        )
        pose = result["pose"]

        # Detect / resolve source skeleton (real detection only — no fake T-pose)
        src = source_pose
        det_backend = "injected" if src is not None else "none"
        if src is None:
            src, det_backend = _extract_source_pose_from_image(
                image, pose_keypoint=pose_keypoint
            )
        if src is not None and (src.get("width") != w or src.get("height") != h):
            src = scale_pose(src, w, h)

        if align_to_source:
            box = content_bbox(rgb)
            if box is not None and (w != iw or h != ih):
                # Scale content bbox into output canvas coords
                sx, sy = w / float(iw), h / float(ih)
                x0, y0, x1, y1 = box
                box = (
                    int(round(x0 * sx)),
                    int(round(y0 * sy)),
                    int(round(x1 * sx)),
                    int(round(y1 * sy)),
                )
            pose = fit_pose_to_source(pose, src, content_box=box)
            pose["width"] = w
            pose["height"] = h

        # preview_pose / guide: output pose only (+ props on guide for generation)
        preview_pose = draw_openpose(pose, width=w, height=h)
        guide = preview_pose.copy()
        _draw_prop_polylines(guide, result["prop_polylines_2d"])

        # source_pose: sprite + real skeleton (DWPose IMAGE preferred)
        base = _resize_rgb(rgb, w, h)
        source_pose_img = _build_source_pose_preview(
            base, src_pose=src, dwpose_image=dwpose_image
        )
        _ = det_backend  # available for future debug output

        # Caption + palette color lock
        override = (caption_override or "").strip()
        if override:
            from .caption import palette_hex_string

            caption = override
            palette_hex = palette_hex_string(rgb)
            backend = "override"
        elif auto_caption:
            caption, palette_hex, backend = describe_character(
                rgb,
                style_bias=(
                    "isometric RTS game sprite, hand-painted / pre-rendered, "
                    "clean readable silhouette, 2D game character"
                ),
                max_tokens=128,
                palette_colors=8,
            )
        else:
            from .caption import palette_hex_string

            palette_hex = palette_hex_string(rgb)
            caption = "full-body 2D game character sprite, cartoon line art"
            backend = "disabled"

        edit_prompt = build_edit_prompt(
            caption,
            palette_hex=palette_hex,
            prop_hint=result["prop_hint"],
            extra=extra_prompt,
        )

        return (
            np_to_tensor(guide),
            edit_prompt,
            pose,
            np_to_tensor(source_pose_img),
            np_to_tensor(preview_pose),
            caption,
            backend,
        )


NODE_CLASS_MAPPINGS = {
    "CP_PoseComposer3D": PoseComposer3D,
    "CP_PoseTransferPrep": PoseTransferPrep,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CP_PoseComposer3D": "Pose Composer 3D",
    "CP_PoseTransferPrep": "Pose Transfer Prep",
}
