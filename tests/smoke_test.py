"""Smoke tests that do not require ComfyUI runtime."""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PARENT = ROOT.parent
PKG = ROOT.name  # _ComfyUI_CharacterPose

if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

pose_io = importlib.import_module(f"{PKG}.formats.pose_io")
char_io = importlib.import_module(f"{PKG}.formats.char_io")
utils = importlib.import_module(f"{PKG}.utils")
warp_mod = importlib.import_module(f"{PKG}.nodes.warp")
pose3d = importlib.import_module(f"{PKG}.pose3d")
caption_mod = importlib.import_module(f"{PKG}.nodes.caption")

load_pose = pose_io.load_pose
make_pose = pose_io.make_pose
scale_pose = pose_io.scale_pose
make_character = char_io.make_character
save_character = char_io.save_character
load_character = char_io.load_character
draw_openpose = utils.draw_openpose
extract_palette = utils.extract_palette
warp_image_tps = warp_mod.warp_image_tps
blend_skeleton_onto_image = warp_mod.blend_skeleton_onto_image
compose_pose = pose3d.compose_pose
build_edit_prompt = caption_mod.build_edit_prompt


def test_poses_library():
    poses_dir = ROOT / "poses"
    files = sorted(poses_dir.glob("*.pose"))
    assert len(files) >= 13, f"expected >=13 poses, got {len(files)}"
    for p in files:
        pose = load_pose(p)
        assert len(pose["keypoints"]) == 18
        img = draw_openpose(pose, width=256, height=256)
        assert img.shape == (256, 256, 3)
        assert img.sum() > 0
    print(f"OK poses library ({len(files)} files)")


def test_char_roundtrip():
    rng = np.random.default_rng(0)
    ref = rng.random((64, 48, 3), dtype=np.float32)
    palette = extract_palette((ref * 255).astype(np.uint8), n_colors=4)
    emb = rng.random((16,), dtype=np.float32)
    ch = make_character(
        "test",
        ref,
        palette=palette,
        embedding=emb,
        metadata={"k": 1, "caption": "red hero sprite"},
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.char"
        save_character(ch, path)
        loaded = load_character(path)
        assert loaded["name"] == "test"
        assert loaded["reference_image"].shape == ref.shape
        assert np.allclose(loaded["reference_image"], ref, atol=1e-5)
        assert np.allclose(loaded["embedding"], emb, atol=1e-5)
        assert loaded["metadata"]["k"] == 1
        assert loaded["metadata"]["caption"] == "red hero sprite"
    print("OK character roundtrip")


def test_scale_pose():
    pose = make_pose([[10, 20, 1]] + [[0, 0, 0]] * 17, width=100, height=200, name="t")
    scaled = scale_pose(pose, 200, 400)
    assert scaled["keypoints"][0][0] == 20
    assert scaled["keypoints"][0][1] == 40
    print("OK scale_pose")


def test_warp_import():
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
    dst = make_pose([[p[0] + 5, p[1] - 3, p[2]] for p in src["keypoints"]], width=128, height=128)
    out = warp_image_tps(img, src, dst)
    assert out.shape == img.shape
    print("OK warp_image_tps")


def test_blend_skeleton():
    img = np.full((64, 64, 3), 40, dtype=np.uint8)
    sk = np.zeros((64, 64, 3), dtype=np.uint8)
    sk[20:40, 30:34] = (0, 255, 0)
    out = blend_skeleton_onto_image(img, sk, opacity=0.8, mode="overlay", stick_grow=1)
    assert out.shape == img.shape
    assert out[30, 32].sum() > img[30, 32].sum()
    print("OK blend_skeleton_onto_image")


def test_pose3d_compose():
    result = compose_pose("idle", camera_preset="SE", width=512, height=512)
    pose = result["pose"]
    assert len(pose["keypoints"]) == 18
    visible = sum(1 for x, y, c in pose["keypoints"] if c > 0)
    assert visible >= 12, f"too few visible keypoints: {visible}"
    img = draw_openpose(pose, width=512, height=512)
    assert img.sum() > 0

    se = compose_pose("idle", camera_preset="SE", width=512, height=512)
    ne = compose_pose("idle", camera_preset="NE", width=512, height=512)
    nose_se = se["pose"]["keypoints"][0]
    nose_ne = ne["pose"]["keypoints"][0]
    if nose_se[2] > 0 and nose_ne[2] > 0:
        dist = abs(nose_se[0] - nose_ne[0]) + abs(nose_se[1] - nose_ne[1])
        assert dist > 1.0, "SE vs NE should differ"

    armed = compose_pose("fight_01", camera_preset="SE", props=["sword", "shield"], width=512, height=512)
    assert "sword" in armed["prop_hint"].lower() or "holding" in armed["prop_hint"].lower()
    assert len(armed["prop_polylines_2d"]) >= 1

    horse = compose_pose("ride_idle", camera_preset="E", props="horse", width=512, height=512)
    assert "horse" in horse["prop_hint"].lower() or "riding" in horse["prop_hint"].lower()
    assert len(horse["prop_polylines_2d"]) >= 1
    print("OK pose3d compose + props + camera")


def test_source_pose_no_fake_tpose():
    """Without keypoints, preview must not invent a standing T-pose skeleton."""
    utils_mod = importlib.import_module(f"{PKG}.utils")
    parse = utils_mod.parse_dwpose_json_keypoints

    # Typical DWPreprocessor OpenPose JSON (flat 18*3) in pixel coords
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
    # scaled from 100x200 canvas → 50x100 image
    assert abs(kps[0][0] - flat[0] * 0.5) < 1e-3

    empty = parse([{"people": [], "canvas_width": 64, "canvas_height": 64}], 64, 64)
    assert empty is None
    print("OK dwpose parse + no empty-people fake pose")


def test_pose_from_prompt_parse_clamp():
    from_prompt = importlib.import_module(f"{PKG}.pose3d.from_prompt")
    parse = from_prompt.parse_llm_angles_text
    clamp = from_prompt.clamp_joint_angles
    angles_from_parsed = from_prompt.angles_from_parsed_joints
    PosePromptError = from_prompt.PosePromptError

    fenced = """```json
    {"joints": {"r_shoulder": [-95, -25, -45], "r_elbow": [200, 0, 0], "bogus": [1,2,3]}, "notes": "guard"}
    ```"""
    parsed = parse(fenced)
    assert "r_shoulder" in parsed
    assert "bogus" not in parsed
    assert parsed["r_elbow"][0] == 140.0  # clamped

    over = clamp({"l_knee": (-50.0, 0.0, 0.0), "unknown_bone": (1.0, 2.0, 3.0)})
    assert "unknown_bone" not in over
    assert over["l_knee"][0] == 0.0

    merged = angles_from_parsed({"r_shoulder": [-80.0, 0.0, -30.0]}, seed_action="idle")
    assert merged["r_shoulder"][0] == -80.0
    assert "l_shoulder" in merged  # from idle seed

    result = compose_pose(
        "text:test",
        camera_preset="SE",
        width=256,
        height=256,
        joint_angles=merged,
    )
    assert len(result["pose"]["keypoints"]) == 18
    assert result["action"] == "text:test"

    try:
        parse("not json at all")
        raise AssertionError("expected PosePromptError")
    except PosePromptError:
        pass

    print("OK pose-from-prompt parse/clamp + joint_angles compose")


def test_edit_prompt_and_caption_fallback():
    from importlib import import_module

    caption_mod2 = import_module(f"{PKG}.nodes.caption")
    build = caption_mod2.build_edit_prompt
    describe = caption_mod2.describe_character
    palette_hex_string = caption_mod2.palette_hex_string
    primary_garment_color = caption_mod2.primary_garment_color

    # Saturated red robe + black belt/boots (must not win as primary)
    rgb = np.full((64, 48, 3), 255, dtype=np.uint8)
    rgb[10:50, 15:35] = (180, 40, 40)
    rgb[40:48, 18:32] = (20, 20, 20)  # boots
    rgb[28:32, 16:34] = (15, 15, 15)  # belt
    rgb[20:26, 20:30] = (40, 80, 180)  # small accent
    pal = palette_hex_string(rgb, n_colors=4)
    assert "#" in pal

    pname, phex = primary_garment_color(rgb, n_colors=4)
    assert "red" in pname.lower(), f"expected red primary, got {pname!r} {phex}"
    assert phex.startswith("#")
    # Primary hex should be reddish (R dominant)
    pr, pg, pb = int(phex[1:3], 16), int(phex[3:5], 16), int(phex[5:7], 16)
    assert pr > pg + 30 and pr > pb + 30, f"primary not red-ish: {phex}"

    prompt = build(
        "red bearded knight in tunic with gold trim",
        palette_hex=pal,
        prop_hint="holding a sword in the right hand",
    )
    assert "APPEARANCE LOCK" in prompt
    assert "COLOR LOCK" in prompt
    assert "primary garment" in prompt.lower()
    assert phex.upper() in prompt.upper() or "red" in prompt.lower()
    low = prompt.lower()
    assert "do not" in low or "do NOT" in prompt
    assert "black" in low or "darken" in low or "blacken" in low
    assert "red bearded knight" in prompt
    assert "#" in prompt
    assert "sword" in prompt.lower()
    assert "NEVER draw bones" in prompt or "never draw bones" in prompt.lower()
    assert "POSE LOCK" in prompt
    assert "single silhouette" in prompt.lower() or "one single silhouette" in prompt.lower()
    assert "QUALITY LOCK" in prompt
    assert "extra limbs" in low
    assert "duplicate" in low or "extra arms" in low
    assert "malformed" in low or "mutated hands" in low
    assert "logo" in low
    assert "ANTI_ARTIFACTS" in caption_mod2.__dict__ or hasattr(caption_mod2, "ANTI_ARTIFACTS")

    crawl = build(
        "red bearded knight in tunic with gold trim",
        palette_hex=pal,
        view_hint=caption_mod2.view_hint_text("RIGHT"),
        view_code="RIGHT",
        animation="crawling",
    )
    cl = crawl.lower()
    assert "POSE LOCK" in crawl
    assert "crawling" in cl
    assert "body action" in cl
    assert "duplicate" in cl or "double body" in cl or "ghost" in cl
    assert "standing" in cl or "t-pose" in cl

    back = build(
        "red bearded knight in tunic with gold trim",
        palette_hex=pal,
        view_hint=caption_mod2.view_hint_text("TOP"),
        view_code="TOP",
    )
    assert "VIEW LOCK" in back
    assert "from behind" in back.lower()
    assert "cape or backpack if any" not in back.lower()
    assert "do not invent a backpack" in back.lower()
    # Hint itself must not invite gear as content
    for code in ("NE", "NO", "N", "TOP"):
        hint = caption_mod2.view_hint_text(code).lower()
        assert "backpack if any" not in hint
        assert "do not invent" in hint and "backpack" in hint

    cap, pal2, backend = describe(rgb, style_bias="game sprite")
    assert isinstance(cap, str) and len(cap) > 5
    assert "#" in pal2
    assert "primary garment" in cap.lower() or "red" in cap.lower()
    assert "fallback" in backend or "florence" in backend.lower()
    print(f"OK caption/prompt ({backend})")


def test_fit_pose_and_filtered_palette():
    warp_mod2 = importlib.import_module(f"{PKG}.nodes.warp")
    utils_mod = importlib.import_module(f"{PKG}.utils")
    fit = warp_mod2.fit_pose_to_source
    content_bbox = utils_mod.content_bbox
    extract_palette = utils_mod.extract_palette
    caption_mod2 = importlib.import_module(f"{PKG}.nodes.caption")
    palette_hex_string = caption_mod2.palette_hex_string

    # Fake Flux canvas: mostly white, small red character blob
    rgb = np.full((256, 256, 3), 255, dtype=np.uint8)
    rgb[80:200, 100:160] = (200, 40, 40)
    rgb[100:140, 110:150] = (50, 90, 180)
    box = content_bbox(rgb)
    assert box is not None
    x0, y0, x1, y1 = box
    assert (x1 - x0) < 200 and (y1 - y0) < 200

    pal = extract_palette(rgb, n_colors=4)
    # No center should be near-white
    for c in pal:
        lum = float(c @ np.array([0.299, 0.587, 0.114], dtype=np.float32))
        assert lum < 0.92, f"palette still has near-white: {c}"
    # First center should be the saturated red (not blue accent / black)
    c0 = pal[0]
    r0, g0, b0 = float(c0[0]), float(c0[1]), float(c0[2])
    assert r0 > g0 and r0 > b0, f"palette[0] should be red-dominant, got {c0}"

    hex_s = palette_hex_string(rgb, n_colors=4)
    assert "#" in hex_s
    assert "#FFFFFF" not in hex_s.upper()
    primary_garment_color = caption_mod2.primary_garment_color
    pname, phex = primary_garment_color(rgb, n_colors=4)
    assert "red" in pname.lower(), f"expected red primary, got {pname!r}"
    assert phex.startswith("#")

    # Source pose matching content height ~120px
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
    # Large target pose (taller bbox)
    tgt = compose_pose("idle", camera_preset="SE", width=256, height=256)["pose"]
    fitted = fit(tgt, src)
    src_ys = [kp[1] for kp in src["keypoints"] if kp[2] > 0]
    fit_ys = [kp[1] for kp in fitted["keypoints"] if kp[2] > 0]
    src_h = max(src_ys) - min(src_ys)
    fit_h = max(fit_ys) - min(fit_ys)
    ratio = fit_h / max(1e-3, src_h)
    assert 0.85 <= ratio <= 1.15, f"fit height ratio {ratio}"

    # Content-box fallback when no source pose
    fitted2 = fit(tgt, None, content_box=box)
    fit2_ys = [kp[1] for kp in fitted2["keypoints"] if kp[2] > 0]
    fit2_h = max(fit2_ys) - min(fit2_ys)
    box_h = (y1 - y0) * 0.90
    assert abs(fit2_h - box_h) / box_h < 0.2

    print("OK fit_pose + filtered palette")


def test_horizontal_fit_crawl():
    """Wide crawl pose: scale stays bounded; ankles near content-box ground."""
    warp_mod2 = importlib.import_module(f"{PKG}.nodes.warp")
    fit = warp_mod2.fit_pose_to_source
    assert warp_mod2._is_horizontal_pose  # helper used by fit

    # Standing source (tall)
    src = make_pose(
        [
            [128, 40, 1],   # nose
            [128, 70, 1],   # neck
            [150, 70, 1],
            [160, 110, 1],
            [165, 150, 1],
            [106, 70, 1],
            [96, 110, 1],
            [91, 150, 1],
            [140, 150, 1],
            [142, 190, 1],
            [144, 220, 1],  # R ankle
            [116, 150, 1],
            [114, 190, 1],
            [112, 220, 1],  # L ankle
            [135, 35, 1],
            [121, 35, 1],
            [140, 38, 1],
            [116, 38, 1],
        ],
        width=256,
        height=256,
        name="src_stand",
    )
    # Crawl-like: wide bbox (horizontal body)
    crawl = make_pose(
        [
            [200, 140, 1],  # nose forward
            [160, 145, 1],  # neck
            [170, 130, 1],
            [210, 125, 1],
            [230, 140, 1],
            [150, 160, 1],
            [120, 170, 1],
            [90, 180, 1],
            [155, 175, 1],
            [130, 185, 1],
            [100, 190, 1],  # R ankle (ground)
            [145, 170, 1],
            [115, 180, 1],
            [85, 190, 1],   # L ankle
            [205, 135, 1],
            [195, 135, 1],
            [210, 138, 1],
            [190, 138, 1],
        ],
        width=256,
        height=256,
        name="crawl",
    )
    assert warp_mod2._is_horizontal_pose(crawl)

    box = (100, 80, 160, 220)  # standing character content
    fitted = fit(crawl, src, content_box=box)
    # Scale must stay within fit clamps (no explosion)
    src_ext = warp_mod2._pose_extent(src)
    fit_ext = warp_mod2._pose_extent(fitted)
    assert src_ext and fit_ext
    ratio = fit_ext / src_ext
    assert 0.2 <= ratio <= 3.6, f"extent ratio out of bounds: {ratio}"

    # After bottom-aligned contain: ankles in lower half of canvas
    low = warp_mod2._lowest_point(fitted)
    assert low is not None
    canvas_h = float(fitted.get("height", 256))
    assert float(low[1]) > canvas_h * 0.55, f"ankle not in lower half: {low[1]} / {canvas_h}"
    # Still near padded bottom (margin ~6%)
    assert float(low[1]) < canvas_h * 0.98, f"ankle past canvas: {low[1]}"

    print("OK horizontal fit crawl")


def test_node_mappings_import():
    root_mod = importlib.import_module(PKG)
    keys = set(root_mod.NODE_CLASS_MAPPINGS)
    for required in (
        "CP_CharacterCaption",
        "CP_BuildEditPrompt",
        "CP_PoseComposer3D",
        "CP_PoseTransferPrep",
        "CP_MixamoPoseExport",
        "CP_KleinPoseBatch",
    ):
        assert required in keys, f"missing node mapping: {required}"
    print(f"OK node mappings ({len(keys)} nodes)")


def test_live_rig_and_preview_batch_wiring():
    """Live sticks on preview_pose; Klein has images + images_with_guides."""
    composer_src = (ROOT / "nodes" / "composer.py").read_text(encoding="utf-8")
    klein_src = (ROOT / "nodes" / "klein_batch.py").read_text(encoding="utf-8")
    assert "first_preview" not in composer_src
    assert "preview_pose = full guide batch" in composer_src
    assert "_push_guide_preview" in klein_src
    assert "_find_guide_preview_nodes" in klein_src
    assert "_PREVIEW_POSE_SLOT = 4" in klein_src
    assert "_push_source_pose_preview" not in klein_src
    assert "images_with_guides" in klein_src
    assert "_overlay_guide_on_image" in klein_src

    klein = importlib.import_module(f"{PKG}.nodes.klein_batch")
    find = klein._find_guide_preview_nodes
    prompt = {
        "5": {"class_type": "CP_PoseTransferPrep", "inputs": {}},
        "6": {"class_type": "PreviewImage", "inputs": {"images": ["5", 3]}},
        "8": {"class_type": "PreviewImage", "inputs": {"images": ["7", 0]}},
        "25": {"class_type": "PreviewImage", "inputs": {"images": ["5", 4]}},
    }
    assert find(prompt) == ["25"]
    assert find({}) == []
    assert find(None) == []

    assert "images_with_guides" in klein.KleinPoseBatch.RETURN_NAMES
    assert klein.KleinPoseBatch.RETURN_NAMES == ("images", "images_with_guides", "count")
    req = klein.KleinPoseBatch.INPUT_TYPES()["required"]
    assert "pose_weight" in req
    assert req["pose_weight"][1].get("default") == 2
    assert "for _ in range(pw)" in klein_src or "range(pw)" in klein_src

    utils_mod = importlib.import_module(f"{PKG}.utils")
    guides = [
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.full((16, 16, 3), 40, dtype=np.uint8),
        np.full((16, 16, 3), 80, dtype=np.uint8),
    ]
    t = utils_mod.batch_np_to_tensor(guides)
    assert int(t.shape[0]) == 3

    # Overlay helper: sticks on image
    rgb = np.full((32, 32, 3), 180, dtype=np.uint8)
    skel = np.zeros((32, 32, 3), dtype=np.uint8)
    skel[8:24, 15:18] = (0, 255, 0)
    img_t = utils_mod.np_to_tensor(rgb)
    sk_t = utils_mod.np_to_tensor(skel)
    over = klein._overlay_guide_on_image(img_t, sk_t)
    assert over.shape[0] == 1

    wf = (ROOT / "workflows" / "pose_transfer_flux_klein.json").read_text(encoding="utf-8")
    assert "squelettes seuls" in wf
    assert "base + squelette detecte" in wf
    assert "images_with_guides" in wf
    assert "QC pose" in wf
    assert "pose_weight" in wf
    print("OK live sticks + images_with_guides wiring")


def test_parse_edit_prompts():
    klein = importlib.import_module(f"{PKG}.nodes.klein_batch")
    parse = klein.parse_edit_prompts
    prompts = parse('["a", "b", "c"]', 3)
    assert prompts == ["a", "b", "c"]
    padded = parse('["only"]', 3)
    assert padded == ["only", "only", "only"]
    truncated = parse('["a", "b", "c", "d"]', 2)
    assert truncated == ["a", "b"]
    try:
        parse("", 1)
        raise AssertionError("expected empty edit_prompts to fail")
    except ValueError:
        pass
    try:
        parse("not-json", 1)
        raise AssertionError("expected invalid JSON to fail")
    except ValueError:
        pass
    print("OK parse_edit_prompts")


if __name__ == "__main__":
    test_poses_library()
    test_char_roundtrip()
    test_scale_pose()
    test_warp_import()
    test_blend_skeleton()
    test_pose3d_compose()
    test_source_pose_no_fake_tpose()
    test_pose_from_prompt_parse_clamp()
    test_edit_prompt_and_caption_fallback()
    test_fit_pose_and_filtered_palette()
    test_horizontal_fit_crawl()
    test_node_mappings_import()
    test_live_rig_and_preview_batch_wiring()
    test_parse_edit_prompts()
    print("All smoke tests passed.")
