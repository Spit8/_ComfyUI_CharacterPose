"""Pose fit and skeleton overlay for Prep preview guides.

Uniform scale/translate (fit_pose_to_source) aligns library poses to the
source character; blend_skeleton_onto_image composites OpenPose sticks
onto the sprite without geometric image warp.
"""

from __future__ import annotations

import copy

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

# OpenPose COCO indices
_NECK = 1
_R_HIP = 8
_L_HIP = 11
_R_SHOULDER = 2
_L_SHOULDER = 5


def _kp(pose: dict, idx: int) -> np.ndarray | None:
    x, y, c = pose["keypoints"][idx]
    if c <= 0:
        return None
    return np.array([x, y], dtype=np.float64)


def _midhip(pose: dict) -> np.ndarray | None:
    rh = _kp(pose, _R_HIP)
    lh = _kp(pose, _L_HIP)
    if rh is not None and lh is not None:
        return (rh + lh) * 0.5
    return rh if rh is not None else lh


def _torso_length(pose: dict) -> float | None:
    neck = _kp(pose, _NECK)
    hip = _midhip(pose)
    if neck is None or hip is None:
        return None
    return float(np.linalg.norm(neck - hip))


def _pose_bbox(pose: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (min_xy, max_xy) over visible keypoints, or None."""
    pts = []
    for x, y, c in pose.get("keypoints") or []:
        if c > 0:
            pts.append([x, y])
    if len(pts) < 2:
        return None
    arr = np.asarray(pts, dtype=np.float64)
    return arr.min(axis=0), arr.max(axis=0)


def _pose_height(pose: dict) -> float | None:
    """Full-body height: prefer head→ankle span, else bbox height."""
    nose = _kp(pose, 0)
    r_ank = _kp(pose, 10)
    l_ank = _kp(pose, 13)
    ankles = [p for p in (r_ank, l_ank) if p is not None]
    if nose is not None and ankles:
        ank = sum(ankles) / len(ankles)
        h = float(abs(ank[1] - nose[1]))
        if h > 1.0:
            return h
    box = _pose_bbox(pose)
    if box is None:
        return None
    mn, mx = box
    h = float(mx[1] - mn[1])
    return h if h > 1.0 else None


def _pose_center(pose: dict) -> np.ndarray | None:
    hip = _midhip(pose)
    if hip is not None:
        return hip
    box = _pose_bbox(pose)
    if box is None:
        return None
    mn, mx = box
    return (mn + mx) * 0.5


def _pose_bbox_size(pose: dict) -> tuple[float, float] | None:
    """Return (width, height) of visible keypoint bbox, or None."""
    box = _pose_bbox(pose)
    if box is None:
        return None
    mn, mx = box
    bw = float(mx[0] - mn[0])
    bh = float(mx[1] - mn[1])
    if bw < 1e-3 and bh < 1e-3:
        return None
    return bw, bh


def _pose_extent(pose: dict) -> float | None:
    """Characteristic size: max(bbox side, diagonal) for crawl / wide poses."""
    sz = _pose_bbox_size(pose)
    if sz is None:
        return None
    bw, bh = sz
    diag = float(np.hypot(bw, bh))
    return max(bw, bh, diag)


def _lowest_point(pose: dict) -> np.ndarray | None:
    """Lowest visible joint (prefer ankles), for ground anchoring."""
    r_ank = _kp(pose, 10)
    l_ank = _kp(pose, 13)
    ankles = [p for p in (r_ank, l_ank) if p is not None]
    if ankles:
        return sum(ankles) / len(ankles)
    box = _pose_bbox(pose)
    if box is None:
        return None
    mn, mx = box
    # Bottom-center of bbox
    return np.array([0.5 * (mn[0] + mx[0]), float(mx[1])], dtype=np.float64)


def _is_horizontal_pose(pose: dict, *, ratio: float = 1.15) -> bool:
    """True when pose bbox is wider than tall (crawl / prone / lying)."""
    sz = _pose_bbox_size(pose)
    if sz is None:
        return False
    bw, bh = sz
    return bh > 1e-3 and bw > bh * float(ratio)


def contain_pose_in_canvas(
    pose: dict,
    *,
    margin: float = 0.06,
    vertical_align: str = "center",
) -> dict:
    """Uniform shrink + recenter so all visible joints stay inside the canvas.

    Never enlarges. No-op if already inside the padded frame or bbox is missing.
    ``vertical_align``: ``\"center\"`` (default) or ``\"bottom\"`` (crawl / prone).
    """
    out = copy.deepcopy(pose)
    w = max(1, int(out.get("width", 1024)))
    h = max(1, int(out.get("height", 1024)))
    m = float(np.clip(margin, 0.0, 0.45))
    pad_x = m * w
    pad_y = m * h
    inner_w = max(1.0, w - 2.0 * pad_x)
    inner_h = max(1.0, h - 2.0 * pad_y)

    box = _pose_bbox(out)
    if box is None:
        return out
    mn, mx = box
    bw = float(mx[0] - mn[0])
    bh = float(mx[1] - mn[1])
    if bw < 1e-3 and bh < 1e-3:
        return out

    cx = 0.5 * (mn[0] + mx[0])
    cy = 0.5 * (mn[1] + mx[1])
    inside = (
        mn[0] >= pad_x
        and mn[1] >= pad_y
        and mx[0] <= w - pad_x
        and mx[1] <= h - pad_y
    )
    if inside and vertical_align != "bottom":
        return out

    s = 1.0
    if not inside:
        if bw > 1e-3:
            s = min(s, inner_w / bw)
        if bh > 1e-3:
            s = min(s, inner_h / bh)
        s = float(min(s, 1.0))

    # After shrink around bbox center, place horizontally centered;
    # vertical: canvas center, or bottom of padded frame for crawl.
    target_cx = 0.5 * w
    if vertical_align == "bottom":
        scaled_bh = bh * s
        target_cy = (h - pad_y) - 0.5 * scaled_bh
    else:
        target_cy = 0.5 * h
    new_kps = []
    for x, y, c in out["keypoints"]:
        if c <= 0:
            new_kps.append([0.0, 0.0, 0.0])
            continue
        nx = (float(x) - cx) * s + target_cx
        ny = (float(y) - cy) * s + target_cy
        new_kps.append([nx, ny, float(c)])
    out["keypoints"] = new_kps
    out["width"] = w
    out["height"] = h
    return out


def fit_pose_to_source(
    target: dict,
    source: dict | None = None,
    *,
    content_box: tuple[int, int, int, int] | None = None,
    height_fill: float = 0.90,
    scale_min: float = 0.25,
    scale_max: float = 3.5,
) -> dict:
    """Uniform scale + translate so target matches source character size.

    Prefers full-body height (bbox / head→ankle) over torso-only. If ``source``
    is missing, fits into ``content_box`` (sprite content bbox) at ~height_fill.
    Wide / crawl poses (bbox wider than tall) use extent-based scale and ground
    (lowest-point) anchoring instead of midhip → standing center.
    After fit, shrinks/recenters if joints would leave the canvas.
    """
    out = copy.deepcopy(target)
    horizontal = _is_horizontal_pose(target)

    if horizontal:
        tgt_ext = _pose_extent(target)
        tgt_anchor = _lowest_point(target)
        if tgt_ext is None or tgt_anchor is None or tgt_ext < 1e-3:
            return out

        src_ext = None
        src_anchor = None
        if source is not None:
            src_ext = _pose_extent(source) or _pose_height(source)
            src_anchor = _lowest_point(source)
        if (src_ext is None or src_anchor is None) and content_box is not None:
            x0, y0, x1, y1 = content_box
            # Match character content size (max side), ground at content bottom
            src_ext = max(1.0, max(float(x1 - x0), float(y1 - y0)) * float(height_fill))
            src_anchor = np.array(
                [(x0 + x1) * 0.5, float(y1) * 0.98 + float(y0) * 0.02],
                dtype=np.float64,
            )
        if src_ext is None or src_anchor is None:
            if source is not None:
                return contain_pose_in_canvas(align_pose_to_source(target, source))
            return out

        scale = float(np.clip(src_ext / tgt_ext, scale_min, scale_max))
        # Also blend shoulder width when available (keeps character thickness)
        if source is not None:
            src_rs, src_ls = _kp(source, _R_SHOULDER), _kp(source, _L_SHOULDER)
            tgt_rs, tgt_ls = _kp(target, _R_SHOULDER), _kp(target, _L_SHOULDER)
            if all(p is not None for p in (src_rs, src_ls, tgt_rs, tgt_ls)):
                src_sw = float(np.linalg.norm(src_rs - src_ls))
                tgt_sw = float(np.linalg.norm(tgt_rs - tgt_ls))
                if tgt_sw > 1e-3 and src_sw > 1e-3:
                    sw_scale = float(src_sw / tgt_sw)
                    scale = float(np.clip(0.7 * scale + 0.3 * sw_scale, scale_min, scale_max))

        new_kps = []
        for x, y, c in target["keypoints"]:
            if c <= 0:
                new_kps.append([0.0, 0.0, 0.0])
                continue
            p = (np.array([x, y], dtype=np.float64) - tgt_anchor) * scale + src_anchor
            new_kps.append([float(p[0]), float(p[1]), float(c)])
        out["keypoints"] = new_kps
        if source is not None:
            out["width"] = source.get("width", out.get("width"))
            out["height"] = source.get("height", out.get("height"))
        # Keep crawl near the ground; don't re-center vertically to mid-canvas
        return contain_pose_in_canvas(out, vertical_align="bottom")

    tgt_h = _pose_height(target)
    tgt_c = _pose_center(target)
    if tgt_h is None or tgt_c is None or tgt_h < 1e-3:
        return out

    src_h = None
    src_c = None
    if source is not None:
        src_h = _pose_height(source)
        src_c = _pose_center(source)

    if (src_h is None or src_c is None) and content_box is not None:
        x0, y0, x1, y1 = content_box
        src_h = max(1.0, float(y1 - y0) * float(height_fill))
        src_c = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)

    if src_h is None or src_c is None:
        # Last resort: old torso align if we have a source pose
        if source is not None:
            return contain_pose_in_canvas(align_pose_to_source(target, source))
        return out

    # Blend full-body scale with torso/shoulder when available
    scale = float(src_h / tgt_h)
    if source is not None:
        src_len = _torso_length(source)
        tgt_len = _torso_length(target)
        scales = [scale]
        if src_len is not None and tgt_len is not None and tgt_len > 1e-3:
            scales.append(float(src_len / tgt_len))
        src_rs, src_ls = _kp(source, _R_SHOULDER), _kp(source, _L_SHOULDER)
        tgt_rs, tgt_ls = _kp(target, _R_SHOULDER), _kp(target, _L_SHOULDER)
        if all(p is not None for p in (src_rs, src_ls, tgt_rs, tgt_ls)):
            src_sw = float(np.linalg.norm(src_rs - src_ls))
            tgt_sw = float(np.linalg.norm(tgt_rs - tgt_ls))
            if tgt_sw > 1e-3 and src_sw > 1e-3:
                scales.append(float(src_sw / tgt_sw))
        # Weight full-body higher so overall size matches the sprite
        if len(scales) == 1:
            scale = scales[0]
        else:
            scale = 0.75 * scales[0] + 0.25 * float(np.mean(scales[1:]))

    scale = float(np.clip(scale, scale_min, scale_max))

    new_kps = []
    for x, y, c in target["keypoints"]:
        if c <= 0:
            new_kps.append([0.0, 0.0, 0.0])
            continue
        p = (np.array([x, y], dtype=np.float64) - tgt_c) * scale + src_c
        new_kps.append([float(p[0]), float(p[1]), float(c)])
    out["keypoints"] = new_kps
    if source is not None:
        out["width"] = source.get("width", out.get("width"))
        out["height"] = source.get("height", out.get("height"))
    return contain_pose_in_canvas(out)


def align_pose_to_source(target: dict, source: dict) -> dict:
    """Rigid-align target pose onto source (uniform scale + translate)."""
    out = copy.deepcopy(target)
    src_hip = _midhip(source)
    tgt_hip = _midhip(target)
    src_len = _torso_length(source)
    tgt_len = _torso_length(target)

    if src_hip is None or tgt_hip is None:
        return out

    scale = 1.0
    if src_len is not None and tgt_len is not None and tgt_len > 1e-3:
        scale = float(np.clip(src_len / tgt_len, 0.35, 2.8))

    src_rs, src_ls = _kp(source, _R_SHOULDER), _kp(source, _L_SHOULDER)
    tgt_rs, tgt_ls = _kp(target, _R_SHOULDER), _kp(target, _L_SHOULDER)
    if all(p is not None for p in (src_rs, src_ls, tgt_rs, tgt_ls)):
        src_sw = float(np.linalg.norm(src_rs - src_ls))
        tgt_sw = float(np.linalg.norm(tgt_rs - tgt_ls))
        if tgt_sw > 1e-3 and src_sw > 1e-3:
            sw_scale = float(np.clip(src_sw / tgt_sw, 0.35, 2.8))
            scale = 0.5 * (scale + sw_scale) if src_len else sw_scale

    new_kps = []
    for x, y, c in target["keypoints"]:
        if c <= 0:
            new_kps.append([0.0, 0.0, 0.0])
            continue
        p = (np.array([x, y], dtype=np.float64) - tgt_hip) * scale + src_hip
        new_kps.append([float(p[0]), float(p[1]), float(c)])
    out["keypoints"] = new_kps
    out["width"] = source.get("width", out.get("width"))
    out["height"] = source.get("height", out.get("height"))
    return out


def blend_skeleton_onto_image(
    image_rgb: np.ndarray,
    skeleton_rgb: np.ndarray,
    opacity: float = 0.65,
    mode: str = "overlay",
    stick_grow: int = 1,
) -> np.ndarray:
    """Composite OpenPose sticks onto the sprite — pose cue without geometric warp.

    mode:
      - overlay: lerp only where skeleton sticks are lit (recommended)
      - soft: blurred stick mask for gentler edges
      - mix: whole-frame lerp (washes the sprite — debug only)
    """
    opacity = float(np.clip(opacity, 0.0, 1.0))
    if opacity <= 1e-6:
        return image_rgb.copy()

    img = image_rgb.astype(np.float32)
    sk = skeleton_rgb.astype(np.float32)
    if sk.shape[:2] != img.shape[:2]:
        if cv2 is None:
            return image_rgb.copy()
        sk = cv2.resize(sk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    mode = str(mode).lower().strip()
    if mode == "mix":
        out = img * (1.0 - opacity) + sk * opacity
        return out.clip(0, 255).astype(np.uint8)

    # Lit stick / joint pixels (OpenPose canvas is black elsewhere)
    mask = (sk.max(axis=2) > 12.0).astype(np.float32)
    if stick_grow > 0 and cv2 is not None:
        k = 2 * int(stick_grow) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel)
        if mode == "soft":
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, stick_grow))
            mask = np.clip(mask, 0.0, 1.0)

    a = mask[:, :, None] * opacity
    out = img * (1.0 - a) + sk * a
    return out.clip(0, 255).astype(np.uint8)


