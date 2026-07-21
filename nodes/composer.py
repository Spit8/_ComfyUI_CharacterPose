"""3D pose composer + pose-transfer prep facade nodes."""

from __future__ import annotations

import numpy as np
import torch

from ..formats.pose_io import make_pose, scale_pose
from ..pose3d.camera import CAMERA_PRESETS, list_camera_presets
from ..pose3d.presets import list_action_presets
from ..pose3d.project import compose_pose
from ..pose3d.props import list_props
from ..utils import draw_openpose, np_to_tensor, tensor_to_np
from .caption import build_edit_prompt, caption_with_florence
from .warp import align_pose_to_source

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


PROP_COLOR = (0, 220, 255)  # RGB cyan-ish for props (distinct from OpenPose)


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


def _extract_source_pose_from_image(image) -> dict | None:
    """Best-effort DWPose/MediaPipe extract for alignment; None if unavailable."""
    rgb = tensor_to_np(image)
    h, w = rgb.shape[:2]
    try:
        from ..utils import estimate_keypoints_mediapipe, try_dwpose_keypoints

        kps = try_dwpose_keypoints(rgb)
        if kps is None:
            kps = estimate_keypoints_mediapipe(rgb)
        if kps is None:
            return None
        return make_pose(kps, width=w, height=h, name="source")
    except Exception:
        return None


class PoseComposer3D:
    """Compose a COCO-18 pose from 3D action presets + orientable camera + props."""

    @classmethod
    def INPUT_TYPES(cls):
        actions = list_action_presets()
        cams = list_camera_presets()
        props = list_props()
        return {
            "required": {
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

        result = compose_pose(
            action,
            camera_preset=camera_preset,
            width=int(width),
            height=int(height),
            props=props,
            fov_deg=float(fov_deg),
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
                "action": (actions, {"default": "idle"}),
                "camera_preset": (cams, {"default": "SE"}),
                "prop": (props, {"default": "none"}),
                "align_to_source": ("BOOLEAN", {"default": True}),
                "auto_caption": ("BOOLEAN", {"default": True}),
            },
            "optional": {
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
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "POSE", "IMAGE", "STRING")
    RETURN_NAMES = ("guide", "edit_prompt", "pose", "preview", "caption")
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
        width=0,
        height=0,
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

        result = compose_pose(
            action,
            camera_preset=camera_preset,
            width=w,
            height=h,
            props=props,
            **cam_kwargs,
        )
        pose = result["pose"]

        src = source_pose
        if align_to_source:
            if src is None:
                src = _extract_source_pose_from_image(image)
            if src is not None:
                src_scaled = scale_pose(src, w, h) if (src.get("width") != w or src.get("height") != h) else src
                pose = align_pose_to_source(pose, src_scaled)
                pose["width"] = w
                pose["height"] = h

        guide = draw_openpose(pose, width=w, height=h)
        _draw_prop_polylines(guide, result["prop_polylines_2d"])

        # Caption
        override = (caption_override or "").strip()
        if override:
            caption = override
        elif auto_caption:
            caption, _ = caption_with_florence(rgb, style_bias="2D game sprite, cartoon")
        else:
            caption = "full-body 2D game character sprite, cartoon line art"

        edit_prompt = build_edit_prompt(
            caption,
            prop_hint=result["prop_hint"],
            extra=extra_prompt,
        )

        # Preview: sprite with skeleton overlay (simple alpha)
        preview = rgb.copy()
        if preview.shape[0] != h or preview.shape[1] != w:
            if cv2 is not None:
                preview = cv2.resize(preview, (w, h), interpolation=cv2.INTER_AREA)
            else:
                from PIL import Image

                preview = np.asarray(Image.fromarray(preview).resize((w, h)), dtype=np.uint8)
        g = guide
        if g.shape[0] != preview.shape[0] or g.shape[1] != preview.shape[1]:
            if cv2 is not None:
                g = cv2.resize(g, (preview.shape[1], preview.shape[0]), interpolation=cv2.INTER_NEAREST)
        stick = g.sum(axis=2) > 0
        blend = preview.astype(np.float32)
        blend[stick] = blend[stick] * 0.35 + g[stick].astype(np.float32) * 0.65
        preview = np.clip(blend, 0, 255).astype(np.uint8)

        return (
            np_to_tensor(guide),
            edit_prompt,
            pose,
            np_to_tensor(preview),
            caption,
        )


NODE_CLASS_MAPPINGS = {
    "CP_PoseComposer3D": PoseComposer3D,
    "CP_PoseTransferPrep": PoseTransferPrep,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CP_PoseComposer3D": "Pose Composer 3D",
    "CP_PoseTransferPrep": "Pose Transfer Prep",
}
