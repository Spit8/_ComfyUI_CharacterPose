"""Shared helpers for ComfyUI tensor / image conversion and OpenPose drawing."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .formats.pose_io import KEYPOINT_COLORS, LIMB_COLORS, LIMB_PAIRS

try:
    import cv2
except Exception:  # pragma: no cover - optional at import time
    cv2 = None


def tensor_to_np(image: torch.Tensor) -> np.ndarray:
    """ComfyUI IMAGE [B,H,W,C] float -> first frame HWC uint8 RGB."""
    if isinstance(image, torch.Tensor):
        arr = image[0].detach().cpu().numpy()
    else:
        arr = np.asarray(image[0])
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def np_to_tensor(image: np.ndarray) -> torch.Tensor:
    """HWC uint8 or float RGB -> ComfyUI IMAGE [1,H,W,C] float."""
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    else:
        if image.max() > 1.5:
            image = image / 255.0
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    return torch.from_numpy(image.astype(np.float32))[None, ...]


def batch_np_to_tensor(images: list[np.ndarray]) -> torch.Tensor:
    tensors = [np_to_tensor(im) for im in images]
    return torch.cat(tensors, dim=0)


def content_mask(image_rgb: np.ndarray, *, luma_bg: float = 240.0) -> np.ndarray:
    """Boolean HxW mask of likely character pixels (exclude flat bright/dark bg).

    Heuristic for Flux-style sprites on large empty canvases:
    - drop near-white low-saturation pixels
    - drop near-black if they dominate corners (letterbox)
    - keep saturated / mid-tone pixels that form the figure
    """
    rgb = np.asarray(image_rgb)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    # Alpha channel if present (H,W,4)
    if rgb.shape[-1] == 4:
        alpha = rgb[..., 3]
        rgb = rgb[..., :3]
        return alpha > 16

    pix = rgb.astype(np.float32)
    r, g, b = pix[..., 0], pix[..., 1], pix[..., 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = np.where(mx > 1e-3, (mx - mn) / np.maximum(mx, 1.0), 0.0)

    # Bright flat background (typical white / off-white canvas)
    is_bright_bg = (luma >= luma_bg) & (sat < 0.12)
    # Very dark empty (rare) — only if most of image is dark
    dark_frac = float(np.mean(luma < 25.0))
    is_dark_bg = (luma < 18.0) & (sat < 0.08) & (dark_frac > 0.55)

    mask = ~(is_bright_bg | is_dark_bg)

    # If almost everything masked out, fall back to non-extreme luma
    if float(np.mean(mask)) < 0.002:
        mask = (luma > 20.0) & (luma < 245.0)
    return mask.astype(bool)


def content_bbox(
    image_rgb: np.ndarray, *, pad: float = 0.04
) -> tuple[int, int, int, int] | None:
    """Tight bbox (x0,y0,x1,y1) exclusive-end around content_mask, or None."""
    mask = content_mask(image_rgb)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    h, w = mask.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pw = int(round((x1 - x0) * pad))
    ph = int(round((y1 - y0) * pad))
    x0 = max(0, x0 - pw)
    y0 = max(0, y0 - ph)
    x1 = min(w, x1 + pw)
    y1 = min(h, y1 + ph)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def crop_to_content(image_rgb: np.ndarray, *, pad: float = 0.06) -> np.ndarray:
    """Crop RGB to content bbox; return original if bbox unavailable."""
    box = content_bbox(image_rgb, pad=pad)
    if box is None:
        return image_rgb
    x0, y0, x1, y1 = box
    return image_rgb[y0:y1, x0:x1]


def extract_palette(image_rgb: np.ndarray, n_colors: int = 8) -> np.ndarray:
    """K-means palette from character pixels only -> float32 (K,3) in [0,1].

    Background (near-white / empty canvas) is excluded so COLOR LOCK reflects
    garments rather than the Flux empty frame.
    """
    from sklearn.cluster import MiniBatchKMeans

    rgb = np.asarray(image_rgb)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    mask = content_mask(rgb)
    pixels = rgb[mask].reshape(-1, 3).astype(np.float32)
    if pixels.shape[0] < 16:
        pixels = rgb.reshape(-1, 3).astype(np.float32)

    # Drop remaining near-white from palette candidates
    luma = pixels @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    keep = luma < 245.0
    if int(np.sum(keep)) >= 16:
        pixels = pixels[keep]

    if pixels.shape[0] > 20000:
        idx = np.random.default_rng(0).choice(pixels.shape[0], 20000, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    k = max(1, min(n_colors, len(sample)))
    km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=2048)
    km.fit(sample)
    centers = np.clip(km.cluster_centers_ / 255.0, 0.0, 1.0).astype(np.float32)
    # Prefer chromatic / mid tones: demote near-white centers
    lum = centers @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    order = np.argsort(lum)
    return centers[order]


def draw_openpose(
    pose: dict[str, Any],
    width: int | None = None,
    height: int | None = None,
    thickness: int | None = None,
) -> np.ndarray:
    """Render an OpenPose-style skeleton image (RGB uint8)."""
    w = int(width or pose.get("width", 1024))
    h = int(height or pose.get("height", 1024))
    kps = pose["keypoints"]

    # Scale if pose canvas differs
    src_w = max(1, int(pose.get("width", w)))
    src_h = max(1, int(pose.get("height", h)))
    sx = w / src_w
    sy = h / src_h

    pts = []
    for x, y, c in kps:
        if c <= 0:
            pts.append(None)
        else:
            pts.append((int(round(x * sx)), int(round(y * sy))))

    stick = thickness if thickness is not None else max(2, int(round(min(w, h) / 160)))
    circle_r = max(3, stick + 1)

    def _joint_rgb(idx: int) -> tuple[int, int, int]:
        color = KEYPOINT_COLORS[idx % len(KEYPOINT_COLORS)]
        return (color[2], color[1], color[0])

    if cv2 is not None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        for i, (a, b) in enumerate(LIMB_PAIRS):
            if a >= len(pts) or b >= len(pts):
                continue
            pa, pb = pts[a], pts[b]
            if pa is None or pb is None:
                continue
            color = LIMB_COLORS[i % len(LIMB_COLORS)]
            rgb = (color[2], color[1], color[0])
            cv2.line(canvas, pa, pb, rgb, stick, lineType=cv2.LINE_AA)
        for idx, p in enumerate(pts):
            if p is None:
                continue
            cv2.circle(canvas, p, circle_r, _joint_rgb(idx), -1, lineType=cv2.LINE_AA)
        return canvas

    # PIL fallback (no OpenCV)
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i, (a, b) in enumerate(LIMB_PAIRS):
        if a >= len(pts) or b >= len(pts):
            continue
        pa, pb = pts[a], pts[b]
        if pa is None or pb is None:
            continue
        color = LIMB_COLORS[i % len(LIMB_COLORS)]
        rgb = (color[2], color[1], color[0])
        draw.line([pa, pb], fill=rgb, width=stick)
    for idx, p in enumerate(pts):
        if p is None:
            continue
        x, y = p
        draw.ellipse(
            (x - circle_r, y - circle_r, x + circle_r, y + circle_r),
            fill=_joint_rgb(idx),
        )
    return np.asarray(img, dtype=np.uint8)


def try_dwpose_keypoints(image_rgb: np.ndarray) -> list[list[float]] | None:
    """Try to extract COCO-18 keypoints via controlnet_aux DWPose if installed."""
    try:
        from controlnet_aux import DWposeDetector
    except Exception:
        return None

    try:
        from PIL import Image

        detector = DWposeDetector.from_pretrained("yzd-v/DWPose")
        pil = Image.fromarray(image_rgb)
        # Most versions return a PIL image; some expose .pose_keypoints
        # Fall back to mediapipe-style parsing if attribute exists after call.
        result = detector(pil, include_hand=False, include_face=False, include_body=True)
        # controlnet_aux typically only returns the drawn image; keypoint access varies.
        # If we cannot get keypoints, return None and let caller use heuristic.
        if hasattr(detector, "pose") and detector.pose is not None:
            # Unofficial / version-dependent
            pass
        _ = result
        return None
    except Exception:
        return None


def estimate_keypoints_mediapipe(image_rgb: np.ndarray) -> list[list[float]] | None:
    """Optional MediaPipe Pose → approximate COCO-18 mapping."""
    try:
        import mediapipe as mp
    except Exception:
        return None

    h, w, _ = image_rgb.shape
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        res = pose.process(image_rgb)
    if not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark

    def kp(idx: int, conf_min: float = 0.3) -> list[float]:
        p = lm[idx]
        c = float(p.visibility)
        if c < conf_min:
            return [0.0, 0.0, 0.0]
        return [p.x * w, p.y * h, c]

    # MediaPipe indices → COCO-18
    nose = kp(0)
    l_eye = kp(2)
    r_eye = kp(5)
    l_ear = kp(7)
    r_ear = kp(8)
    l_shoulder = kp(11)
    r_shoulder = kp(12)
    l_elbow = kp(13)
    r_elbow = kp(14)
    l_wrist = kp(15)
    r_wrist = kp(16)
    l_hip = kp(23)
    r_hip = kp(24)
    l_knee = kp(25)
    r_knee = kp(26)
    l_ankle = kp(27)
    r_ankle = kp(28)

    def mid(a: list[float], b: list[float]) -> list[float]:
        if a[2] <= 0 or b[2] <= 0:
            return [0.0, 0.0, 0.0]
        return [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, min(a[2], b[2])]

    neck = mid(l_shoulder, r_shoulder)
    return [
        nose,
        neck,
        r_shoulder,
        r_elbow,
        r_wrist,
        l_shoulder,
        l_elbow,
        l_wrist,
        r_hip,
        r_knee,
        r_ankle,
        l_hip,
        l_knee,
        l_ankle,
        r_eye,
        l_eye,
        r_ear,
        l_ear,
    ]


def parse_dwpose_json_keypoints(pose_json: Any, width: int, height: int) -> list[list[float]] | None:
    """Parse POSE_KEYPOINT structures emitted by comfyui_controlnet_aux DWPreprocessor.

    Supported shapes:
    - dict with people[].pose_keypoints_2d (+ optional canvas_width/canvas_height)
    - list of such dicts (batch / multi-frame) — uses the first frame
    - dict with bodies / keypoints
    - flat list of people
    """
    if pose_json is None:
        return None

    frame = pose_json
    # Batch: [ {people, canvas_width, canvas_height}, ... ]
    if isinstance(pose_json, list) and pose_json and isinstance(pose_json[0], dict):
        if "people" in pose_json[0] or "canvas_width" in pose_json[0] or "bodies" in pose_json[0]:
            frame = pose_json[0]

    canvas_w = width
    canvas_h = height
    people = None

    if isinstance(frame, dict):
        canvas_w = int(frame.get("canvas_width") or frame.get("width") or width)
        canvas_h = int(frame.get("canvas_height") or frame.get("height") or height)
        people = frame.get("people") or frame.get("bodies")
        if people is None and "keypoints" in frame:
            people = [frame]
        # DWPose aux sometimes stores body as nested list under "body"
        if people is None and "body" in frame:
            people = [{"pose_keypoints_2d": frame["body"]}]
    elif isinstance(frame, list):
        people = frame

    if not people:
        return None

    person = people[0]
    if isinstance(person, dict):
        cand = (
            person.get("pose_keypoints_2d")
            or person.get("pose_keypoints")
            or person.get("body")
            or person.get("keypoints")
        )
        # Some versions nest candidate = {"candidate": [[x,y,score],...], "subset": ...}
        if cand is None and "candidate" in person:
            cand = person["candidate"]
    else:
        cand = person

    if cand is None:
        return None

    # candidate may be list of [x,y] or [x,y,score]
    arr = np.asarray(cand, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        if arr.shape[1] == 2:
            scores = np.ones((arr.shape[0], 1), dtype=np.float32)
            pts = np.concatenate([arr[:, :2], scores], axis=1)
        else:
            pts = arr[:, :3]
    else:
        flat = arr.reshape(-1)
        if flat.size % 3 == 0:
            pts = flat.reshape(-1, 3)
        elif flat.size % 2 == 0:
            pts = np.concatenate(
                [flat.reshape(-1, 2), np.ones((flat.size // 2, 1), dtype=np.float32)],
                axis=1,
            )
        else:
            return None

    pts = pts.copy()
    # Detect normalized coords (common in DWPreprocessor output)
    xmax = float(np.nanmax(pts[:, 0])) if pts.size else 0.0
    ymax = float(np.nanmax(pts[:, 1])) if pts.size else 0.0
    if xmax <= 1.5 and ymax <= 1.5:
        pts[:, 0] *= float(canvas_w)
        pts[:, 1] *= float(canvas_h)

    # Always map from detector canvas to the requested image size
    if canvas_w > 0 and canvas_h > 0 and (canvas_w != width or canvas_h != height):
        pts[:, 0] *= float(width) / float(canvas_w)
        pts[:, 1] *= float(height) / float(canvas_h)

    out = []
    for i in range(18):
        if i < len(pts):
            x, y, c = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])
            out.append([x, y, c] if c > 0 else [0.0, 0.0, 0.0])
        else:
            out.append([0.0, 0.0, 0.0])
    return out
