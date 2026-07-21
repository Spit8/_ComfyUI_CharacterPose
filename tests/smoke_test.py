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
caption_with_florence = caption_mod.caption_with_florence


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


def test_edit_prompt_and_caption_fallback():
    prompt = build_edit_prompt(
        "red bearded knight in tunic",
        prop_hint="holding a sword in the right hand",
    )
    assert "red bearded knight" in prompt
    assert "sword" in prompt.lower()
    assert "NEVER draw bones" in prompt or "never draw bones" in prompt.lower()

    rgb = np.zeros((64, 48, 3), dtype=np.uint8)
    rgb[10:50, 15:35] = (180, 40, 40)
    cap, backend = caption_with_florence(rgb, style_bias="game sprite")
    assert isinstance(cap, str) and len(cap) > 5
    assert "fallback" in backend or "florence" in backend.lower()
    print(f"OK caption/prompt ({backend})")


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
    test_edit_prompt_and_caption_fallback()
    test_node_mappings_import()
    print("All smoke tests passed.")
