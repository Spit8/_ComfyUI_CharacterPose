"""Diagnostic: WarpToPose melt investigation on Example_Comfy.png."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PKG = Path(r"C:\Users\wiwil\OneDrive\Bureau\_ComfyUI_CharacterPose")
LIVE = Path(r"C:\Users\wiwil\Documents\ComfyUI\custom_nodes\_ComfyUI_CharacterPose")
sys.path.insert(0, str(LIVE.parent))  # so _ComfyUI_CharacterPose is importable
sys.path.insert(0, str(PKG.parent))

# Prefer live package name
import importlib
mod_name = "_ComfyUI_CharacterPose"
try:
    pkg = importlib.import_module(mod_name)
except Exception:
    # fallback: load by path
    sys.path.insert(0, str(PKG))
    import utils as _utils  # noqa
    from formats.pose_io import load_pose, KEYPOINT_NAMES, make_pose, scale_pose
    from nodes.warp import (
        align_pose_to_source,
        blend_poses,
        clamp_pose_displacements,
        warp_image_piecewise_affine,
        warp_image_tps,
    )
    from utils import draw_openpose, try_dwpose_keypoints
else:
    from _ComfyUI_CharacterPose.formats.pose_io import (
        load_pose,
        KEYPOINT_NAMES,
        make_pose,
        scale_pose,
    )
    from _ComfyUI_CharacterPose.nodes.warp import (
        align_pose_to_source,
        blend_poses,
        clamp_pose_displacements,
        warp_image_piecewise_affine,
        warp_image_tps,
    )
    from _ComfyUI_CharacterPose.utils import draw_openpose, try_dwpose_keypoints

OUT = PKG / "tests"
OUT.mkdir(parents=True, exist_ok=True)

IMG_PATH = Path(r"C:\Users\wiwil\Documents\ComfyUI\input\Example_Comfy.png")
POSE_PATH = PKG / "poses" / "idle_side.pose"

# Look for saved source pose near project
CANDIDATE_POSES = [
    PKG / "tests" / "source.pose",
    PKG / "poses" / "source.pose",
    Path(r"C:\Users\wiwil\Documents\ComfyUI\input") / "Example_Comfy.pose",
    Path(r"C:\Users\wiwil\Documents\ComfyUI\user\default\workflows") / "Example_Comfy.pose",
]


def overlay_skeleton(rgb: np.ndarray, pose: dict, alpha: float = 0.65) -> np.ndarray:
    sk = draw_openpose(pose, width=rgb.shape[1], height=rgb.shape[0])
    mask = sk.sum(axis=2) > 0
    out = rgb.copy().astype(np.float32)
    out[mask] = out[mask] * (1 - alpha) + sk[mask].astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def heuristic_stick_from_bbox(rgb: np.ndarray) -> dict:
    """Centered COCO-18 stick figure matching non-white pixel bbox."""
    h, w, _ = rgb.shape
    # non-near-white
    white = (rgb[:, :, 0] > 245) & (rgb[:, :, 1] > 245) & (rgb[:, :, 2] > 245)
    ys, xs = np.where(~white)
    if len(xs) < 50:
        # fallback full canvas figure
        x0, x1, y0, y1 = int(w * 0.35), int(w * 0.65), int(h * 0.15), int(h * 0.9)
    else:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    cx = (x0 + x1) * 0.5

    def p(fx, fy, c=1.0):
        return [x0 + fx * bw, y0 + fy * bh, c]

    # Approximate frontal/side character proportions in bbox
    # 0 nose, 1 neck, 2 R shoulder, 3 R elbow, 4 R wrist,
    # 5 L shoulder, 6 L elbow, 7 L wrist, 8 R hip, 9 R knee, 10 R ankle,
    # 11 L hip, 12 L knee, 13 L ankle, 14 R eye, 15 L eye, 16 R ear, 17 L ear
    kps = [
        p(0.55, 0.08),  # nose
        p(0.52, 0.18),  # neck
        p(0.62, 0.20),  # R shoulder (image right)
        p(0.72, 0.35),  # R elbow
        p(0.78, 0.48),  # R wrist
        p(0.42, 0.20),  # L shoulder
        p(0.38, 0.38),  # L elbow
        p(0.40, 0.50),  # L wrist
        p(0.58, 0.48),  # R hip
        p(0.60, 0.68),  # R knee
        p(0.62, 0.92),  # R ankle
        p(0.46, 0.48),  # L hip
        p(0.44, 0.68),  # L knee
        p(0.42, 0.92),  # L ankle
        p(0.58, 0.06),  # R eye
        p(0.52, 0.06),  # L eye
        p(0.64, 0.08),  # R ear
        p(0.46, 0.08),  # L ear
    ]
    print(f"HEURISTIC bbox=({x0},{y0})-({x1},{y1}) size={bw}x{bh} canvas={w}x{h}")
    return make_pose(kps, width=w, height=h, name="heuristic_bbox")


def get_source_pose(rgb: np.ndarray) -> tuple[dict, str]:
    for p in CANDIDATE_POSES:
        if p.is_file():
            print(f"SOURCE: loaded saved pose {p}")
            return load_pose(p), f"file:{p}"
    # DWPose
    try:
        kps = try_dwpose_keypoints(rgb)
        if kps is not None:
            print("SOURCE: DWPose via try_dwpose_keypoints")
            return make_pose(kps, width=rgb.shape[1], height=rgb.shape[0], name="dwpose"), "dwpose"
    except Exception as e:
        print(f"SOURCE: DWPose failed: {e!r}")
    # try import controlnet aux directly
    try:
        from custom_controlnet_aux.dwpose import DwposeDetector  # type: ignore
        print("SOURCE: found DwposeDetector but skipping heavy init; use heuristic")
    except Exception as e:
        print(f"SOURCE: comfyui_controlnet_aux not usable: {e!r}")
    pose = heuristic_stick_from_bbox(rgb)
    return pose, "heuristic"


def print_displacements(src: dict, dst: dict, tag: str, diag: float):
    print(f"\n=== Displacements [{tag}] ===")
    ds = []
    for i, name in enumerate(KEYPOINT_NAMES):
        sx, sy, sc = src["keypoints"][i]
        dx, dy, dc = dst["keypoints"][i]
        if sc <= 0 or dc <= 0:
            print(f"  {i:2d} {name:12s}: INVALID src=({sx:.1f},{sy:.1f},{sc:.2f}) dst=({dx:.1f},{dy:.1f},{dc:.2f})")
            continue
        d = float(np.hypot(dx - sx, dy - sy))
        ds.append(d)
        print(
            f"  {i:2d} {name:12s}: src=({sx:7.1f},{sy:7.1f}) dst=({dx:7.1f},{dy:7.1f}) "
            f"disp={d:6.1f}px ({100*d/diag:.1f}% diag)"
        )
    if not ds:
        print("  no valid pairs")
        return
    mean_d, max_d = float(np.mean(ds)), float(np.max(ds))
    thr = 0.30 * diag
    over = [d for d in ds if d > thr]
    print(
        f"  MEAN={mean_d:.1f} MAX={max_d:.1f} diag={diag:.1f} "
        f"30%diag={thr:.1f} any>30%diag={len(over)>0} count={len(over)}"
    )


def shift_pose(pose: dict, dx: float, dy: float) -> dict:
    out = json.loads(json.dumps(pose))
    kps = []
    for x, y, c in out["keypoints"]:
        if c <= 0:
            kps.append([0.0, 0.0, 0.0])
        else:
            kps.append([x + dx, y + dy, c])
    out["keypoints"] = kps
    return out


def main():
    rgb = np.array(Image.open(IMG_PATH).convert("RGB"))
    h, w, _ = rgb.shape
    diag = float(np.hypot(w, h))
    print(f"IMAGE {IMG_PATH} {w}x{h} diag={diag:.1f}")

    src, src_how = get_source_pose(rgb)
    src = scale_pose(src, w, h)
    print(f"source method={src_how}")

    target = load_pose(POSE_PATH)
    target = scale_pose(target, w, h)
    print(f"target={POSE_PATH.name}")

    # pipeline matching WarpToPose defaults
    dst = align_pose_to_source(target, src)
    dst = blend_poses(src, dst, 0.45)
    dst = clamp_pose_displacements(src, dst, max_ratio=0.40)

    print_displacements(src, dst, "aligned+blend0.45+clamp0.40", diag)

    Image.fromarray(overlay_skeleton(rgb, src)).save(OUT / "diag_source_skel.png")
    Image.fromarray(overlay_skeleton(rgb, dst)).save(OUT / "diag_aligned_target.png")

    warped_pa = warp_image_piecewise_affine(rgb, src, dst, include_limb_mids=False)
    warped_tps = warp_image_tps(rgb, src, dst, include_limb_mids=False, smoothing=4.0)
    Image.fromarray(warped_pa).save(OUT / "diag_warped_piecewise.png")
    Image.fromarray(warped_tps).save(OUT / "diag_warped_tps.png")

    # intentional wrong source (+200px)
    bad_src = shift_pose(src, 200, 0)
    dst_bad = align_pose_to_source(target, bad_src)
    dst_bad = blend_poses(bad_src, dst_bad, 0.45)
    dst_bad = clamp_pose_displacements(bad_src, dst_bad, max_ratio=0.40)
    print_displacements(bad_src, dst_bad, "WRONG_SRC +200x then align/blend/clamp", diag)

    warped_bad_pa = warp_image_piecewise_affine(rgb, bad_src, dst_bad, include_limb_mids=False)
    warped_bad_tps = warp_image_tps(rgb, bad_src, dst_bad, include_limb_mids=False, smoothing=4.0)
    Image.fromarray(overlay_skeleton(rgb, bad_src)).save(OUT / "diag_wrong_src_skel.png")
    Image.fromarray(warped_bad_pa).save(OUT / "diag_wrong_src_piecewise.png")
    Image.fromarray(warped_bad_tps).save(OUT / "diag_wrong_src_tps.png")

    # metrics on warp change
    def diff_stats(a, b, name):
        d = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()
        changed = (np.abs(a.astype(np.int16) - b.astype(np.int16)).sum(axis=2) > 15).mean()
        print(f"WARP_DIFF {name}: mean_abs={d:.2f} frac_changed>15={changed*100:.1f}%")

    diff_stats(rgb, warped_pa, "piecewise")
    diff_stats(rgb, warped_tps, "tps")
    diff_stats(rgb, warped_bad_pa, "wrong_src_piecewise")
    diff_stats(rgb, warped_bad_tps, "wrong_src_tps")
    print(f"Saved overlays under {OUT}")


if __name__ == "__main__":
    main()
