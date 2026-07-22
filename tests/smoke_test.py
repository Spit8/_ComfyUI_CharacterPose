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

    rgb = np.zeros((64, 48, 3), dtype=np.uint8)
    rgb[10:50, 15:35] = (180, 40, 40)
    rgb[20:30, 18:32] = (40, 80, 180)
    pal = palette_hex_string(rgb, n_colors=4)
    assert "#" in pal

    prompt = build(
        "red bearded knight in tunic with gold trim",
        palette_hex=pal,
        prop_hint="holding a sword in the right hand",
    )
    assert "APPEARANCE LOCK" in prompt
    assert "COLOR LOCK" in prompt
    assert "garment" in prompt.lower() or "character colors" in prompt.lower()
    assert "red bearded knight" in prompt
    assert "#" in prompt
    assert "sword" in prompt.lower()
    assert "NEVER draw bones" in prompt or "never draw bones" in prompt.lower()

    cap, pal2, backend = describe(rgb, style_bias="game sprite")
    assert isinstance(cap, str) and len(cap) > 5
    assert "#" in pal2
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

    hex_s = palette_hex_string(rgb, n_colors=4)
    assert "#" in hex_s
    assert "#FFFFFF" not in hex_s.upper()

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


def test_node_mappings_import():
    root_mod = importlib.import_module(PKG)
    keys = set(root_mod.NODE_CLASS_MAPPINGS)
    for required in (
        "CP_CharacterCaption",
        "CP_BuildEditPrompt",
        "CP_PoseComposer3D",
        "CP_PoseTransferPrep",
    ):
        assert required in keys, f"missing node mapping: {required}"
    print(f"OK node mappings ({len(keys)} nodes)")


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
    test_node_mappings_import()
    print("All smoke tests passed.")
