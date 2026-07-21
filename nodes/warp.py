"""Geometric warp from source pose to target pose — identity-preserving path.

TPS (global thin-plate) is unstable on side-view skeletons (near-collinear
keypoints) and melts sprites into multi-head smears. Default is Delaunay
piecewise-affine, which keeps the warp local to each body triangle.
"""

from __future__ import annotations

import copy

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from ..formats.pose_io import LIMB_PAIRS
from ..utils import draw_openpose, np_to_tensor, tensor_to_np

# OpenPose COCO indices
_NECK = 1
_R_HIP = 8
_L_HIP = 11
_R_SHOULDER = 2
_L_SHOULDER = 5


def _valid_points(pose: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, Nx2 points) for keypoints with confidence > 0."""
    pts = []
    idxs = []
    for i, (x, y, c) in enumerate(pose["keypoints"]):
        if c > 0:
            pts.append([x, y])
            idxs.append(i)
    if not pts:
        return np.array([], dtype=np.int64), np.zeros((0, 2), dtype=np.float64)
    return np.array(idxs, dtype=np.int64), np.asarray(pts, dtype=np.float64)


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


def blend_poses(source: dict, target: dict, strength: float) -> dict:
    """Interpolate keypoints source→target. strength 0 = source, 1 = target."""
    strength = float(np.clip(strength, 0.0, 1.0))
    out = copy.deepcopy(target)
    kps = []
    for i in range(18):
        sx, sy, sc = source["keypoints"][i]
        tx, ty, tc = target["keypoints"][i]
        if sc <= 0 and tc <= 0:
            kps.append([0.0, 0.0, 0.0])
        elif sc <= 0:
            kps.append([tx, ty, tc])
        elif tc <= 0:
            kps.append([sx, sy, sc])
        else:
            kps.append(
                [
                    sx + (tx - sx) * strength,
                    sy + (ty - sy) * strength,
                    min(sc, tc),
                ]
            )
    out["keypoints"] = kps
    out["width"] = source.get("width", out.get("width"))
    out["height"] = source.get("height", out.get("height"))
    return out


def clamp_pose_displacements(source: dict, target: dict, max_ratio: float = 0.45) -> dict:
    """Limit per-keypoint travel relative to torso length (stops wild teleports)."""
    torso = _torso_length(source) or _torso_length(target)
    if torso is None or torso < 1.0:
        return target
    max_d = float(torso) * float(max_ratio)
    out = copy.deepcopy(target)
    kps = []
    for i in range(18):
        sx, sy, sc = source["keypoints"][i]
        tx, ty, tc = target["keypoints"][i]
        if sc <= 0 or tc <= 0:
            kps.append([tx, ty, tc] if tc > 0 else [sx, sy, sc] if sc > 0 else [0.0, 0.0, 0.0])
            continue
        d = np.array([tx - sx, ty - sy], dtype=np.float64)
        n = float(np.linalg.norm(d))
        if n > max_d and n > 1e-6:
            d = d * (max_d / n)
            tx, ty = sx + d[0], sy + d[1]
        kps.append([float(tx), float(ty), float(min(sc, tc))])
    out["keypoints"] = kps
    return out


def _control_points(
    src_pose: dict,
    dst_pose: dict,
    include_limb_mids: bool,
    w: int,
    h: int,
    add_corners: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    src_idxs, _ = _valid_points(src_pose)
    dst_idxs, _ = _valid_points(dst_pose)
    common = [i for i in src_idxs.tolist() if i in set(dst_idxs.tolist())]
    if len(common) < 3:
        return None

    src_sel = np.array([src_pose["keypoints"][i][:2] for i in common], dtype=np.float64)
    dst_sel = np.array([dst_pose["keypoints"][i][:2] for i in common], dtype=np.float64)

    if include_limb_mids:
        extra_src, extra_dst = [], []
        for a, b in LIMB_PAIRS:
            sa, sb = src_pose["keypoints"][a], src_pose["keypoints"][b]
            da, db = dst_pose["keypoints"][a], dst_pose["keypoints"][b]
            if sa[2] > 0 and sb[2] > 0 and da[2] > 0 and db[2] > 0:
                extra_src.append([(sa[0] + sb[0]) * 0.5, (sa[1] + sb[1]) * 0.5])
                extra_dst.append([(da[0] + db[0]) * 0.5, (da[1] + db[1]) * 0.5])
        if extra_src:
            src_sel = np.vstack([src_sel, np.asarray(extra_src)])
            dst_sel = np.vstack([dst_sel, np.asarray(extra_dst)])

    if add_corners:
        corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float64)
        src_sel = np.vstack([src_sel, corners])
        dst_sel = np.vstack([dst_sel, corners])

    # Drop near-duplicate points (Delaunay / TPS choke on them)
    keep = []
    used = []
    for i, p in enumerate(dst_sel):
        if any(np.linalg.norm(p - q) < 1.5 for q in used):
            continue
        used.append(p)
        keep.append(i)
    src_sel = src_sel[keep]
    dst_sel = dst_sel[keep]
    if len(src_sel) < 3:
        return None
    return src_sel, dst_sel


def warp_image_piecewise_affine(
    image_rgb: np.ndarray,
    src_pose: dict,
    dst_pose: dict,
    include_limb_mids: bool = True,
) -> np.ndarray:
    """Warp via Delaunay triangles in destination space (inverse map).

    Local affine per triangle — stable on side-view / near-collinear skeletons
    where global TPS melts the sprite.
    """
    if cv2 is None:
        return image_rgb.copy()

    h, w, _ = image_rgb.shape
    pts = _control_points(src_pose, dst_pose, include_limb_mids, w, h, add_corners=True)
    if pts is None:
        return image_rgb.copy()
    src_sel, dst_sel = pts

    try:
        tri = Delaunay(dst_sel)
    except Exception:
        return _affine_fallback(image_rgb, src_sel, dst_sel, w, h)

    map_x = np.full((h, w), -1.0, dtype=np.float32)
    map_y = np.full((h, w), -1.0, dtype=np.float32)
    # Identity outside the hull so background stays put
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x[:] = grid_x
    map_y[:] = grid_y

    for simplex in tri.simplices:
        dst_tri = dst_sel[simplex].astype(np.float32)
        src_tri = src_sel[simplex].astype(np.float32)
        # Skip degenerate triangles
        area = cv2.contourArea(dst_tri.reshape(-1, 1, 2))
        if area < 1.0:
            continue
        M = cv2.getAffineTransform(dst_tri, src_tri)
        x0 = int(np.floor(dst_tri[:, 0].min()))
        y0 = int(np.floor(dst_tri[:, 1].min()))
        x1 = int(np.ceil(dst_tri[:, 0].max())) + 1
        y1 = int(np.ceil(dst_tri[:, 1].max())) + 1
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue

        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        local = dst_tri.copy()
        local[:, 0] -= x0
        local[:, 1] -= y0
        cv2.fillConvexPoly(mask, np.round(local).astype(np.int32), 1)

        yy, xx = np.where(mask > 0)
        if len(xx) == 0:
            continue
        ones = np.ones(len(xx), dtype=np.float32)
        dst_pts = np.stack([xx.astype(np.float32) + x0, yy.astype(np.float32) + y0, ones], axis=0)
        src_pts = M @ dst_pts
        map_x[y0 + yy, x0 + xx] = src_pts[0]
        map_y[y0 + yy, x0 + xx] = src_pts[1]

    return cv2.remap(
        image_rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def warp_image_tps(
    image_rgb: np.ndarray,
    src_pose: dict,
    dst_pose: dict,
    include_limb_mids: bool = True,
    smoothing: float = 4.0,
) -> np.ndarray:
    """Legacy global TPS — prefer piecewise_affine for sprites."""
    h, w, _ = image_rgb.shape
    pts = _control_points(src_pose, dst_pose, include_limb_mids, w, h, add_corners=True)
    if pts is None:
        return image_rgb.copy()
    src_sel, dst_sel = pts

    try:
        rbf_x = RBFInterpolator(dst_sel, src_sel[:, 0], kernel="thin_plate_spline", smoothing=smoothing)
        rbf_y = RBFInterpolator(dst_sel, src_sel[:, 1], kernel="thin_plate_spline", smoothing=smoothing)
    except Exception:
        return _affine_fallback(image_rgb, src_sel, dst_sel, w, h)

    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float64)
    query = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    map_x = rbf_x(query).reshape(h, w).astype(np.float32)
    map_y = rbf_y(query).reshape(h, w).astype(np.float32)

    if cv2 is not None:
        return cv2.remap(
            image_rgb,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    map_x_i = np.clip(np.rint(map_x), 0, w - 1).astype(np.int32)
    map_y_i = np.clip(np.rint(map_y), 0, h - 1).astype(np.int32)
    return image_rgb[map_y_i, map_x_i]


def _affine_fallback(image_rgb, src_sel, dst_sel, w, h):
    if cv2 is not None:
        M, _ = cv2.estimateAffinePartial2D(src_sel, dst_sel)
        if M is None:
            return image_rgb.copy()
        return cv2.warpAffine(image_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return image_rgb.copy()


def _mean_displacement(src_pose: dict, dst_pose: dict) -> float:
    ds = []
    for i in range(18):
        sx, sy, sc = src_pose["keypoints"][i]
        tx, ty, tc = dst_pose["keypoints"][i]
        if sc > 0 and tc > 0:
            ds.append(float(np.hypot(tx - sx, ty - sy)))
    return float(np.mean(ds)) if ds else 0.0


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


class WarpToPose:
    """Align target pose + optional guidance blend (no melt by default).

    Recommended path: method=none, skeleton_blend>0 → guided image for Repair,
    clean skeleton for ControlNet.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_pose": ("POSE",),
                "target_pose": ("POSE",),
                # none = keep pixels; blend skeleton onto image for Repair guidance.
                "method": (["none", "piecewise_affine", "tps"], {"default": "none"}),
                "smoothing": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 50.0, "step": 0.5}),
                "warp_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_displace_ratio": ("FLOAT", {"default": 0.20, "min": 0.05, "max": 2.0, "step": 0.05}),
                "align_to_source": (["true", "false"], {"default": "true"}),
                "include_limb_mids": (["true", "false"], {"default": "false"}),
                "skeleton_blend": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 1.0, "step": 0.05}),
                "blend_mode": (["overlay", "soft", "mix"], {"default": "overlay"}),
                "stick_grow": ("INT", {"default": 2, "min": 0, "max": 12, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "POSE")
    RETURN_NAMES = ("guided", "skeleton", "image_clean", "aligned_target_pose")
    FUNCTION = "warp"
    CATEGORY = "CharacterPose/Warp"

    def warp(
        self,
        image,
        source_pose,
        target_pose,
        method="none",
        smoothing=4.0,
        warp_strength=0.0,
        max_displace_ratio=0.20,
        align_to_source="true",
        include_limb_mids="false",
        skeleton_blend=0.70,
        blend_mode="overlay",
        stick_grow=2,
    ):
        rgb = tensor_to_np(image)
        h, w, _ = rgb.shape

        from ..formats.pose_io import scale_pose

        src = scale_pose(source_pose, w, h)
        dst = scale_pose(target_pose, w, h)

        if str(align_to_source).lower() in ("true", "1", "yes"):
            dst = align_pose_to_source(dst, src)

        strength = float(warp_strength)
        method = str(method).lower().strip()
        base = rgb
        if strength > 0.001 and method not in ("none", "off", "skip", "0"):
            dst_w = blend_poses(src, dst, strength)
            dst_w = clamp_pose_displacements(src, dst_w, max_ratio=float(max_displace_ratio))
            use_mids = str(include_limb_mids).lower() in ("true", "1", "yes")
            if method == "tps":
                base = warp_image_tps(
                    rgb, src, dst_w, include_limb_mids=use_mids, smoothing=float(smoothing)
                )
            else:
                base = warp_image_piecewise_affine(rgb, src, dst_w, include_limb_mids=use_mids)
            dst = dst_w

        skeleton = draw_openpose(dst, width=w, height=h)
        guided = blend_skeleton_onto_image(
            base,
            skeleton,
            opacity=float(skeleton_blend),
            mode=str(blend_mode),
            stick_grow=int(stick_grow),
        )
        return (
            np_to_tensor(guided),
            np_to_tensor(skeleton),
            np_to_tensor(base),
            dst,
        )


class BlendSkeleton:
    """Blend a target OpenPose skeleton onto a sprite (standalone)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "skeleton": ("IMAGE",),
                "opacity": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 1.0, "step": 0.05}),
                "mode": (["overlay", "soft", "mix"], {"default": "overlay"}),
                "stick_grow": ("INT", {"default": 2, "min": 0, "max": 12, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("guided",)
    FUNCTION = "run"
    CATEGORY = "CharacterPose/Warp"

    def run(self, image, skeleton, opacity=0.70, mode="overlay", stick_grow=2):
        out = blend_skeleton_onto_image(
            tensor_to_np(image),
            tensor_to_np(skeleton),
            opacity=float(opacity),
            mode=str(mode),
            stick_grow=int(stick_grow),
        )
        return (np_to_tensor(out),)


class WarpThenPreview:
    """Convenience: guided blend preview."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_pose": ("POSE",),
                "target_pose": ("POSE",),
                "skeleton_blend": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "run"
    CATEGORY = "CharacterPose/Warp"

    def run(self, image, source_pose, target_pose, skeleton_blend=0.70):
        guided, _, _, _ = WarpToPose().warp(
            image, source_pose, target_pose, skeleton_blend=skeleton_blend
        )
        return (guided,)


NODE_CLASS_MAPPINGS = {
    "CP_WarpToPose": WarpToPose,
    "CP_BlendSkeleton": BlendSkeleton,
    "CP_WarpThenPreview": WarpThenPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CP_WarpToPose": "Warp To Pose + Skeleton Blend",
    "CP_BlendSkeleton": "Blend Skeleton Onto Image",
    "CP_WarpThenPreview": "Warp Preview Overlay",
}
