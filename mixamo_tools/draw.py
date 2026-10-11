"""Draw OpenPose-style stick figures (RGB uint8, black background)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mixamo_tools.mixamo_bones import LIMB_PAIRS

# Approximate OpenPose limb colors (RGB)
LIMB_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
]

JOINT_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 85, 85),
]


def draw_openpose(
    keypoints: list[list[float]],
    *,
    width: int = 1024,
    height: int = 1024,
    thickness: int | None = None,
    label: str | None = None,
) -> np.ndarray:
    """Render COCO-18 keypoints as OpenPose sticks. keypoints: [x,y,conf] × 18."""
    stick = thickness if thickness is not None else max(2, int(round(min(width, height) / 160)))
    circle_r = max(3, stick + 1)
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    pts: list[tuple[int, int] | None] = []
    for x, y, c in keypoints:
        if c <= 0:
            pts.append(None)
        else:
            pts.append((int(round(x)), int(round(y))))

    for i, (a, b) in enumerate(LIMB_PAIRS):
        if a >= len(pts) or b >= len(pts):
            continue
        pa, pb = pts[a], pts[b]
        if pa is None or pb is None:
            continue
        color = LIMB_COLORS[i % len(LIMB_COLORS)]
        draw.line([pa, pb], fill=color, width=stick)

    for idx, p in enumerate(pts):
        if p is None:
            continue
        x, y = p
        color = JOINT_COLORS[idx % len(JOINT_COLORS)]
        draw.ellipse(
            (x - circle_r, y - circle_r, x + circle_r, y + circle_r),
            fill=color,
        )

    if label:
        try:
            font = ImageFont.truetype("arial.ttf", size=max(18, width // 40))
        except OSError:
            font = ImageFont.load_default()
        draw.text((24, 24), label, fill=(220, 220, 220), font=font)

    return np.asarray(img, dtype=np.uint8)
