"""Smoke tests that do not require ComfyUI runtime."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PARENT = ROOT.parent
PKG = ROOT.name  # _ComfyUI_CharacterPose

if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

pose_io = importlib.import_module(f"{PKG}.formats.pose_io")
utils = importlib.import_module(f"{PKG}.utils")
warp_mod = importlib.import_module(f"{PKG}.nodes.warp")
caption_mod = importlib.import_module(f"{PKG}.nodes.caption")

load_pose = pose_io.load_pose
make_pose = pose_io.make_pose
scale_pose = pose_io.scale_pose
draw_openpose = utils.draw_openpose
extract_palette = utils.extract_palette
warp_image_tps = warp_mod.warp_image_tps
blend_skeleton_onto_image = warp_mod.blend_skeleton_onto_image
build_edit_prompt = caption_mod.build_edit_prompt


def test_poses_library():
    poses_dir = ROOT / "poses"
    files = sorted(poses_dir.glob("*.pose"))
    if not files:
        print("SKIP poses library (poses/ empty — run Mixamo export)")
        return
    for p in files:
        pose = load_pose(p)
        assert len(pose["keypoints"]) == 18
        img = draw_openpose(pose, width=256, height=256)
        assert img.shape == (256, 256, 3)
        assert img.sum() > 0
    print(f"OK poses library ({len(files)} files)")


def test_scale_pose():
    pose = make_pose([[10, 20, 1]] + [[0, 0, 0]] * 17, width=100, height=200, name="t")
    scaled = scale_pose(pose, 200, 400)
    assert scaled["keypoints"][0][0] == 20
    assert scaled["keypoints"][0][1] == 40
    print("OK scale_pose")


def test_warp_helpers():
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[40:90, 50:80] = 200
    src = make_pose(
        [
            [64, 30, 1],
            [64, 50, 1],
            [80, 50, 1],
            [90, 70, 1],
            [95, 90, 1],
            [48, 50, 1],
            [38, 70, 1],
            [33, 90, 1],
            [72, 80, 1],
            [74, 100, 1],
            [74, 120, 1],
            [56, 80, 1],
            [54, 100, 1],
            [54, 120, 1],
            [70, 28, 1],
            [58, 28, 1],
            [75, 32, 1],
            [53, 32, 1],
        ],
        width=128,
        height=128,
    )
    dst = make_pose(
        [[p[0] + 5, p[1] - 3, p[2]] for p in src["keypoints"]], width=128, height=128
    )
    out = warp_image_tps(img, src, dst)
    assert out.shape == img.shape

    sk = np.zeros((64, 64, 3), dtype=np.uint8)
    sk[20:40, 30:34] = (0, 255, 0)
    blended = blend_skeleton_onto_image(
        np.full((64, 64, 3), 40, dtype=np.uint8),
        sk,
        opacity=0.8,
        mode="overlay",
        stick_grow=1,
    )
    assert blended.shape == (64, 64, 3)
    assert blended[30, 32].sum() > 40 * 3
    print("OK warp helpers (tps + blend)")


def test_source_pose_no_fake_tpose():
    """Without keypoints, preview must not invent a standing T-pose skeleton."""
    parse = utils.parse_dwpose_json_keypoints

    flat = []
    for i in range(18):
        flat.extend([20.0 + i * 2.0, 30.0 + (i % 5) * 8.0, 1.0])
    payload = [
        {
            "people": [{"pose_keypoints_2d": flat}],
            "canvas_width": 100,
            "canvas_height": 200,
        }
    ]
    kps = parse(payload, 50, 100)
    assert kps is not None
    assert len(kps) == 18
    assert kps[0][2] > 0
    assert abs(kps[0][0] - flat[0] * 0.5) < 1e-3

    empty = parse([{"people": [], "canvas_width": 64, "canvas_height": 64}], 64, 64)
    assert empty is None
    print("OK dwpose parse + no empty-people fake pose")


def test_edit_prompt_and_caption_fallback():
    build = caption_mod.build_edit_prompt
    describe = caption_mod.describe_character
    palette_hex_string = caption_mod.palette_hex_string
    primary_garment_color = caption_mod.primary_garment_color

    rgb = np.full((64, 48, 3), 255, dtype=np.uint8)
    rgb[10:50, 15:35] = (180, 40, 40)
    rgb[40:48, 18:32] = (20, 20, 20)
    rgb[28:32, 16:34] = (15, 15, 15)
    rgb[20:26, 20:30] = (40, 80, 180)
    pal = palette_hex_string(rgb, n_colors=4)
    assert "#" in pal

    pname, phex = primary_garment_color(rgb, n_colors=4)
    assert "red" in pname.lower(), f"expected red primary, got {pname!r} {phex}"
    assert phex.startswith("#")
    pr, pg, pb = int(phex[1:3], 16), int(phex[3:5], 16), int(phex[5:7], 16)
    assert pr > pg + 30 and pr > pb + 30, f"primary not red-ish: {phex}"

    prompt = build(
        "red bearded knight in tunic with gold trim",
        palette_hex=pal,
        extra="holding a sword in the right hand",
    )
    assert "APPEARANCE LOCK" in prompt
    assert "COLOR LOCK" in prompt
    assert "primary garment" in prompt.lower()
    assert "red bearded knight" in prompt
    assert "sword" in prompt.lower()
    assert "NEVER draw bones" in prompt or "never draw bones" in prompt.lower()
    assert "POSE LOCK" in prompt
    assert "QUALITY LOCK" in prompt

    crawl = build(
        "red bearded knight in tunic with gold trim",
        palette_hex=pal,
        view_hint=caption_mod.view_hint_text("RIGHT"),
        view_code="RIGHT",
        animation="crawling",
    )
    cl = crawl.lower()
    assert "POSE LOCK" in crawl
    assert "crawling" in cl

    back = build(
        "red bearded knight in tunic with gold trim",
        palette_hex=pal,
        view_hint=caption_mod.view_hint_text("TOP"),
        view_code="TOP",
    )
    assert "VIEW LOCK" in back
    assert "from behind" in back.lower()
    assert "do not invent a backpack" in back.lower()

    for code in ("NE", "NO", "N", "TOP"):
        hint = caption_mod.view_hint_text(code).lower()
        assert "backpack if any" not in hint
        assert "do not invent" in hint and "backpack" in hint

    cap, pal2, backend = describe(rgb, style_bias="game sprite")
    assert isinstance(cap, str) and len(cap) > 5
    assert "#" in pal2
    assert "fallback" in backend or "florence" in backend.lower()
    print(f"OK caption/prompt ({backend})")


def test_fit_pose_and_filtered_palette():
    fit = warp_mod.fit_pose_to_source
    content_bbox = utils.content_bbox
    palette_hex_string = caption_mod.palette_hex_string

    rgb = np.full((256, 256, 3), 255, dtype=np.uint8)
    rgb[80:200, 100:160] = (200, 40, 40)
    rgb[100:140, 110:150] = (50, 90, 180)
    box = content_bbox(rgb)
    assert box is not None
    x0, y0, x1, y1 = box
    assert (x1 - x0) < 200 and (y1 - y0) < 200

    pal = extract_palette(rgb, n_colors=4)
    for c in pal:
        lum = float(c @ np.array([0.299, 0.587, 0.114], dtype=np.float32))
        assert lum < 0.92, f"palette still has near-white: {c}"
    c0 = pal[0]
    assert float(c0[0]) > float(c0[1]) and float(c0[0]) > float(c0[2])

    hex_s = palette_hex_string(rgb, n_colors=4)
    assert "#" in hex_s
    assert "#FFFFFF" not in hex_s.upper()

    src = make_pose(
        [
            [130, 90, 1],
            [130, 110, 1],
            [150, 110, 1],
            [160, 140, 1],
            [165, 165, 1],
            [110, 110, 1],
            [100, 140, 1],
            [95, 165, 1],
            [145, 160, 1],
            [148, 190, 1],
            [150, 210, 1],
            [115, 160, 1],
            [112, 190, 1],
            [110, 210, 1],
            [135, 85, 1],
            [125, 85, 1],
            [140, 88, 1],
            [120, 88, 1],
        ],
        width=256,
        height=256,
        name="src",
    )
    # Tall target (larger bbox than src)
    tgt = make_pose(
        [
            [128, 20, 1],
            [128, 50, 1],
            [155, 50, 1],
            [170, 100, 1],
            [175, 150, 1],
            [101, 50, 1],
            [86, 100, 1],
            [81, 150, 1],
            [145, 150, 1],
            [148, 200, 1],
            [150, 240, 1],
            [111, 150, 1],
            [108, 200, 1],
            [106, 240, 1],
            [135, 15, 1],
            [121, 15, 1],
            [140, 18, 1],
            [116, 18, 1],
        ],
        width=256,
        height=256,
        name="tgt",
    )
    fitted = fit(tgt, src)
    src_ys = [kp[1] for kp in src["keypoints"] if kp[2] > 0]
    fit_ys = [kp[1] for kp in fitted["keypoints"] if kp[2] > 0]
    src_h = max(src_ys) - min(src_ys)
    fit_h = max(fit_ys) - min(fit_ys)
    ratio = fit_h / max(1e-3, src_h)
    assert 0.85 <= ratio <= 1.15, f"fit height ratio {ratio}"

    fitted2 = fit(tgt, None, content_box=box)
    fit2_ys = [kp[1] for kp in fitted2["keypoints"] if kp[2] > 0]
    fit2_h = max(fit2_ys) - min(fit2_ys)
    box_h = (y1 - y0) * 0.90
    assert abs(fit2_h - box_h) / box_h < 0.2

    print("OK fit_pose + filtered palette")


def test_horizontal_fit_crawl():
    """Wide crawl pose: scale stays bounded; ankles near content-box ground."""
    fit = warp_mod.fit_pose_to_source

    src = make_pose(
        [
            [128, 40, 1],
            [128, 70, 1],
            [150, 70, 1],
            [160, 110, 1],
            [165, 150, 1],
            [106, 70, 1],
            [96, 110, 1],
            [91, 150, 1],
            [140, 150, 1],
            [142, 190, 1],
            [144, 220, 1],
            [116, 150, 1],
            [114, 190, 1],
            [112, 220, 1],
            [135, 35, 1],
            [121, 35, 1],
            [140, 38, 1],
            [116, 38, 1],
        ],
        width=256,
        height=256,
        name="src_stand",
    )
    crawl = make_pose(
        [
            [200, 140, 1],
            [160, 145, 1],
            [170, 130, 1],
            [210, 125, 1],
            [230, 140, 1],
            [150, 160, 1],
            [120, 170, 1],
            [90, 180, 1],
            [155, 175, 1],
            [130, 185, 1],
            [100, 190, 1],
            [145, 170, 1],
            [115, 180, 1],
            [85, 190, 1],
            [205, 135, 1],
            [195, 135, 1],
            [210, 138, 1],
            [190, 138, 1],
        ],
        width=256,
        height=256,
        name="crawl",
    )
    assert warp_mod._is_horizontal_pose(crawl)

    box = (100, 80, 160, 220)
    fitted = fit(crawl, src, content_box=box)
    src_ext = warp_mod._pose_extent(src)
    fit_ext = warp_mod._pose_extent(fitted)
    assert src_ext and fit_ext
    ratio = fit_ext / src_ext
    assert 0.2 <= ratio <= 3.6, f"extent ratio out of bounds: {ratio}"

    low = warp_mod._lowest_point(fitted)
    assert low is not None
    canvas_h = float(fitted.get("height", 256))
    assert float(low[1]) > canvas_h * 0.55
    assert float(low[1]) < canvas_h * 0.98

    print("OK horizontal fit crawl")


def test_node_mappings_import():
    root_mod = importlib.import_module(PKG)
    keys = set(root_mod.NODE_CLASS_MAPPINGS)
    expected = {
        "CP_PoseTransferPrep",
        "CP_KleinPoseBatch",
        "CP_MixamoPoseExport",
        "CP_PoseLibraryLoad",
        "CP_ApplyPose",
    }
    assert keys == expected, f"unexpected mappings: {keys ^ expected}"
    print(f"OK node mappings ({len(keys)} nodes)")


def test_live_rig_and_preview_batch_wiring():
    """Live sticks on preview_pose; Klein has images + images_with_guides."""
    composer_src = (ROOT / "nodes" / "composer.py").read_text(encoding="utf-8")
    klein_src = (ROOT / "nodes" / "klein_batch.py").read_text(encoding="utf-8")
    assert "first_preview" not in composer_src
    assert "preview_pose = full guide batch" in composer_src
    assert "prop2" not in composer_src
    assert "list_props" not in composer_src
    assert "PoseComposer3D" not in composer_src
    assert "_push_guide_preview" in klein_src
    assert "images_with_guides" in klein_src

    klein = importlib.import_module(f"{PKG}.nodes.klein_batch")
    find = klein._find_guide_preview_nodes
    prompt = {
        "5": {"class_type": "CP_PoseTransferPrep", "inputs": {}},
        "6": {"class_type": "PreviewImage", "inputs": {"images": ["5", 3]}},
        "8": {"class_type": "PreviewImage", "inputs": {"images": ["7", 0]}},
        "25": {"class_type": "PreviewImage", "inputs": {"images": ["5", 4]}},
    }
    assert find(prompt) == ["25"]

    assert klein.KleinPoseBatch.RETURN_NAMES == ("images", "images_with_guides", "count")
    req = klein.KleinPoseBatch.INPUT_TYPES()["required"]
    assert "pose_weight" in req

    prep = importlib.import_module(f"{PKG}.nodes.composer").PoseTransferPrep
    widgets = prep.INPUT_TYPES()
    assert "prop" not in widgets["required"]
    assert "prop" not in widgets.get("optional", {})
    assert "prop2" not in widgets.get("optional", {})

    wf = (ROOT / "workflows" / "pose_transfer_flux_klein.json").read_text(encoding="utf-8")
    assert "images_with_guides" in wf
    assert '"prop":' not in wf and '"prop2":' not in wf
    print("OK live sticks + Prep sans props wiring")


def test_parse_edit_prompts():
    klein = importlib.import_module(f"{PKG}.nodes.klein_batch")
    parse = klein.parse_edit_prompts
    assert parse('["a", "b", "c"]', 3) == ["a", "b", "c"]
    assert parse('["only"]', 3) == ["only", "only", "only"]
    assert parse('["a", "b", "c", "d"]', 2) == ["a", "b"]
    try:
        parse("", 1)
        raise AssertionError("expected empty edit_prompts to fail")
    except ValueError:
        pass
    print("OK parse_edit_prompts")


def test_no_pose3d_package():
    assert not (ROOT / "pose3d").exists(), "pose3d/ should be removed"
    try:
        importlib.import_module(f"{PKG}.pose3d")
        raise AssertionError("pose3d should not import")
    except ModuleNotFoundError:
        pass
    print("OK pose3d removed")


if __name__ == "__main__":
    test_poses_library()
    test_scale_pose()
    test_warp_helpers()
    test_source_pose_no_fake_tpose()
    test_edit_prompt_and_caption_fallback()
    test_fit_pose_and_filtered_palette()
    test_horizontal_fit_crawl()
    test_node_mappings_import()
    test_live_rig_and_preview_batch_wiring()
    test_parse_edit_prompts()
    test_no_pose3d_package()
    print("All smoke tests passed.")
