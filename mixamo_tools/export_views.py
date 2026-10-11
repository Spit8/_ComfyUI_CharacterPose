"""CLI: Mixamo FBX → isometric OpenPose skeleton PNG + .pose (SE/SO/NE/NO)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
# Allow: python -m mixamo_tools.export_views  (repo root)
#    and: python -m export_views / python export_views.py  (cwd = mixamo_tools/)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from formats.pose_io import make_pose, save_pose  # noqa: E402
from mixamo_tools.cameras import (  # noqa: E402
    CAMERA_PRESETS,
    VIEW_ORDER,
    fit_camera_distance,
    project_joints,
)
from mixamo_tools.draw import draw_openpose  # noqa: E402
from mixamo_tools.fbx_skeleton import (  # noqa: E402
    coco18_keypoints_2d,
    load_animation,
    sample_coco18_joints,
)

DEFAULT_FBX = REPO / "mixamo" / "Slow Run.fbx"
DEFAULT_OUT = ROOT / "out"


def _slug(name: str) -> str:
    s = Path(name).stem.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "anim"


def sample_frame_indices(n_frames: int, positions: int) -> list[int]:
    """Evenly spaced frame indices over the clip (endpoints included)."""
    n_frames = max(int(n_frames), 1)
    n = max(int(positions), 1)
    if n == 1:
        return [0]
    raw = [int(round(i * (n_frames - 1) / (n - 1))) for i in range(n)]
    out: list[int] = []
    for idx in raw:
        idx = int(np.clip(idx, 0, n_frames - 1))
        if not out or out[-1] != idx:
            out.append(idx)
    return out


def resolve_views(view: str) -> list[str]:
    v = view.strip().upper()
    if v in ("ALL", "*"):
        return list(VIEW_ORDER)
    if v not in CAMERA_PRESETS:
        raise ValueError(f"Unknown view {view!r}; expected SE|SO|NE|NO|ALL")
    return [v]


def export_views(
    fbx_path: Path,
    out_dir: Path,
    *,
    positions: int = 4,
    frame: int | None = None,
    view: str = "ALL",
    size: int = 1024,
    stem: str | None = None,
    margin: float = 0.10,
    write_png: bool = True,
    write_pose: bool = True,
) -> list[Path]:
    if not write_png and not write_pose:
        raise ValueError("Nothing to write: both --no-png and --no-pose were set")

    anim = load_animation(fbx_path)
    n_frames = anim["n_frames"]
    views = resolve_views(view)

    if frame is not None:
        frame_indices = [int(np.clip(frame, 0, n_frames - 1))]
    else:
        frame_indices = sample_frame_indices(n_frames, positions)

    joints_per_frame = [sample_coco18_joints(anim, frame=fi) for fi in frame_indices]
    look_at, distance = fit_camera_distance(
        joints_per_frame,
        views,
        size=size,
        margin=margin,
        look_at=(0.0, 0.95, 0.0),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = stem or _slug(fbx_path.name)
    n_pos = len(frame_indices)
    written: list[Path] = []

    for pi, (fi, joints) in enumerate(zip(frame_indices, joints_per_frame)):
        for v in views:
            projected = project_joints(
                joints,
                v,
                width=size,
                height=size,
                look_at=look_at,
                distance=distance,
            )
            kps = coco18_keypoints_2d(projected)
            base = f"{prefix}_{v}_{pi:02d}"

            if write_pose:
                pose = make_pose(
                    kps,
                    width=size,
                    height=size,
                    name=base,
                    metadata={
                        "source": str(fbx_path),
                        "frame": int(fi),
                        "view": v,
                        "look_at": [float(x) for x in look_at],
                        "distance": float(distance),
                        "sample_index": int(pi),
                        "positions": int(n_pos),
                    },
                )
                pose_path = out_dir / f"{base}.pose"
                save_pose(pose, pose_path)
                written.append(pose_path)
                print(f"wrote {pose_path}")

            if write_png:
                label = f"{v}  p{pi}/{n_pos}  f{fi}"
                img = draw_openpose(kps, width=size, height=size, label=label)
                png_path = out_dir / f"{base}.png"
                Image.fromarray(img).save(png_path)
                written.append(png_path)
                print(f"wrote {png_path}")

    print(
        f"n_frames={n_frames} sampled={frame_indices} views={','.join(views)} "
        f"look_at={look_at} distance={distance:.3f} "
        f"png={write_png} pose={write_pose}"
    )
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Export isometric OpenPose views from a Mixamo FBX "
        "(PNG debug + .pose for ComfyUI PoseLibrary; fixed camera across frames)."
    )
    p.add_argument("--fbx", type=Path, default=DEFAULT_FBX, help="Path to Mixamo .fbx")
    p.add_argument(
        "--positions",
        type=int,
        default=4,
        help="Number of evenly spaced animation samples (default: 4). Ignored if --frame is set.",
    )
    p.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Single frame index override (ignores --positions)",
    )
    p.add_argument(
        "--view",
        type=str,
        default="ALL",
        help="SE|SO|NE|NO|ALL (default: ALL)",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output directory")
    p.add_argument("--size", type=int, default=1024, help="Canvas size (square)")
    p.add_argument("--stem", type=str, default=None, help="Output filename prefix")
    p.add_argument(
        "--margin",
        type=float,
        default=0.10,
        help="Canvas margin fraction when fitting camera distance (default: 0.10)",
    )
    p.add_argument(
        "--no-png",
        action="store_true",
        help="Skip writing debug PNG stick figures",
    )
    p.add_argument(
        "--no-pose",
        action="store_true",
        help="Skip writing .pose JSON files",
    )
    args = p.parse_args(argv)

    if not args.fbx.is_file():
        print(f"FBX not found: {args.fbx}", file=sys.stderr)
        return 1
    if args.positions < 1:
        print("--positions must be >= 1", file=sys.stderr)
        return 1

    try:
        export_views(
            args.fbx,
            args.out_dir,
            positions=args.positions,
            frame=args.frame,
            view=args.view,
            size=args.size,
            stem=args.stem,
            margin=args.margin,
            write_png=not args.no_png,
            write_pose=not args.no_pose,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
