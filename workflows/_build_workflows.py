"""Generate valid ComfyUI workflow JSON with consistent link wiring."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def link(lid, src, src_slot, dst, dst_slot, typ):
    return [lid, src, src_slot, dst, dst_slot, typ]


def write_flux_klein():
    """Flux.2 Klein + Prep + sequential CP_KleinPoseBatch."""
    # 1 UNET  2 CLIP  3 VAE  4 LoadImage  5 Prep
    # 6 Preview source  7 CP_KleinPoseBatch  8 Preview results  9 Save
    # 10 Note  25 Preview pose  26 DWPreprocessor  27 Preview with guides

    links = [
        link(1, 1, 0, 7, 0, "MODEL"),
        link(2, 2, 0, 7, 1, "CLIP"),
        link(3, 3, 0, 7, 2, "VAE"),
        link(6, 4, 0, 5, 0, "IMAGE"),
        link(33, 4, 0, 26, 0, "IMAGE"),
        link(34, 26, 1, 5, 7, "POSE_KEYPOINT"),
        link(35, 26, 0, 5, 8, "IMAGE"),
        link(7, 5, 9, 7, 3, "IMAGE"),  # image_batch
        link(8, 5, 0, 7, 4, "IMAGE"),  # guide
        link(9, 5, 7, 7, 5, "STRING"),  # edit_prompts
        link(10, 5, 3, 6, 0, "IMAGE"),  # source_pose preview
        link(32, 5, 4, 25, 0, "IMAGE"),  # preview_pose (sticks)
        link(11, 7, 0, 9, 0, "IMAGE"),  # results -> Save
        link(12, 7, 0, 8, 0, "IMAGE"),  # results -> Preview
        link(36, 7, 1, 27, 0, "IMAGE"),  # images_with_guides -> Preview
    ]

    def outs(nid):
        m = {}
        for L in links:
            if L[1] == nid:
                m.setdefault(L[2], []).append(L[0])
        return m

    def inp(nid, slot):
        for L in links:
            if L[3] == nid and L[4] == slot:
                return L[0]
        return None

    o = {i: outs(i) for i in range(1, 27)}

    nodes = [
        {
            "id": 1,
            "type": "UNETLoader",
            "pos": [40, 40],
            "size": [360, 82],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {"name": "MODEL", "type": "MODEL", "links": o[1].get(0), "slot_index": 0}
            ],
            "properties": {
                "Node name for S&R": "UNETLoader",
                "models": [
                    {
                        "name": "flux-2-klein-4b-fp8.safetensors",
                        "url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors",
                        "directory": "diffusion_models",
                    }
                ],
            },
            "widgets_values": ["flux-2-klein-4b-fp8.safetensors", "default"],
        },
        {
            "id": 2,
            "type": "CLIPLoader",
            "pos": [40, 160],
            "size": [360, 106],
            "flags": {},
            "order": 1,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {"name": "CLIP", "type": "CLIP", "links": o[2].get(0), "slot_index": 0}
            ],
            "properties": {
                "Node name for S&R": "CLIPLoader",
                "models": [
                    {
                        "name": "qwen_3_4b.safetensors",
                        "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors",
                        "directory": "text_encoders",
                    }
                ],
            },
            "widgets_values": ["qwen_3_4b.safetensors", "flux2", "default"],
        },
        {
            "id": 3,
            "type": "VAELoader",
            "pos": [40, 300],
            "size": [360, 58],
            "flags": {},
            "order": 2,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {"name": "VAE", "type": "VAE", "links": o[3].get(0), "slot_index": 0}
            ],
            "properties": {
                "Node name for S&R": "VAELoader",
                "models": [
                    {
                        "name": "flux2-vae.safetensors",
                        "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
                        "directory": "vae",
                    }
                ],
            },
            "widgets_values": ["flux2-vae.safetensors"],
        },
        {
            "id": 4,
            "type": "LoadImage",
            "pos": [40, 400],
            "size": [320, 314],
            "flags": {},
            "order": 3,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": o[4].get(0), "slot_index": 0},
                {"name": "MASK", "type": "MASK", "links": None, "slot_index": 1},
            ],
            "title": "Sprite de reference",
            "properties": {"Node name for S&R": "LoadImage"},
            "widgets_values": ["Example_Comfy.png", "image"],
        },
        {
            "id": 26,
            "type": "DWPreprocessor",
            "pos": [400, 280],
            "size": [315, 200],
            "flags": {},
            "order": 4,
            "mode": 0,
            "inputs": [{"name": "image", "type": "IMAGE", "link": inp(26, 0)}],
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": o[26].get(0), "slot_index": 0},
                {
                    "name": "POSE_KEYPOINT",
                    "type": "POSE_KEYPOINT",
                    "links": o[26].get(1),
                    "slot_index": 1,
                },
            ],
            "title": "DWPose (source skeleton)",
            "properties": {"Node name for S&R": "DWPreprocessor"},
            "widgets_values": [
                "enable",
                "enable",
                "enable",
                1024,
                "yolox_l.onnx",
                "dw-ll_ucoco_384.onnx",
                "disable",
            ],
        },
        {
            "id": 5,
            "type": "CP_PoseTransferPrep",
            "pos": [420, 500],
            "size": [360, 360],
            "flags": {},
            "order": 5,
            "mode": 0,
            "inputs": [
                {"name": "image", "type": "IMAGE", "link": inp(5, 0)},
                {"name": "animation", "type": "COMBO", "widget": {"name": "animation"}, "link": None},
                {"name": "orientation", "type": "COMBO", "widget": {"name": "orientation"}, "link": None},
                {"name": "frame", "type": "COMBO", "widget": {"name": "frame"}, "link": None},
                {"name": "prop", "type": "COMBO", "widget": {"name": "prop"}, "link": None},
                {
                    "name": "align_to_source",
                    "type": "BOOLEAN",
                    "widget": {"name": "align_to_source"},
                    "link": None,
                },
                {
                    "name": "auto_caption",
                    "type": "BOOLEAN",
                    "widget": {"name": "auto_caption"},
                    "link": None,
                },
                {"name": "pose_keypoint", "type": "POSE_KEYPOINT", "link": inp(5, 7)},
                {"name": "dwpose_image", "type": "IMAGE", "link": inp(5, 8)},
                {"name": "prop2", "type": "COMBO", "widget": {"name": "prop2"}, "link": None},
                {"name": "caption", "type": "STRING", "widget": {"name": "caption"}, "link": None},
                {"name": "extra", "type": "STRING", "widget": {"name": "extra"}, "link": None},
                {"name": "source_pose", "type": "POSE", "link": None},
                {"name": "pose_list", "type": "STRING", "widget": {"name": "pose_list"}, "link": None},
                {"name": "pose_file", "type": "STRING", "widget": {"name": "pose_file"}, "link": None},
                {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": None},
                {"name": "height", "type": "INT", "widget": {"name": "height"}, "link": None},
            ],
            "outputs": [
                {"name": "guide", "type": "IMAGE", "links": o[5].get(0), "slot_index": 0},
                {"name": "edit_prompt", "type": "STRING", "links": None, "slot_index": 1},
                {"name": "pose", "type": "POSE", "links": None, "slot_index": 2},
                {"name": "source_pose", "type": "IMAGE", "links": o[5].get(3), "slot_index": 3},
                {"name": "preview_pose", "type": "IMAGE", "links": o[5].get(4), "slot_index": 4},
                {"name": "caption", "type": "STRING", "links": None, "slot_index": 5},
                {"name": "caption_backend", "type": "STRING", "links": None, "slot_index": 6},
                {"name": "edit_prompts", "type": "STRING", "links": o[5].get(7), "slot_index": 7},
                {"name": "pose_names", "type": "STRING", "links": None, "slot_index": 8},
                {"name": "image_batch", "type": "IMAGE", "links": o[5].get(9), "slot_index": 9},
                {"name": "count", "type": "INT", "links": None, "slot_index": 10},
            ],
            "title": "Prep: filters -> guide batch",
            "properties": {"Node name for S&R": "CP_PoseTransferPrep"},
            "widgets_values": [
                "ALL",
                "ALL",
                "ALL",
                "none",
                True,
                True,
                "none",
                "",
                "",
                "",
                "",
                0,
                0,
            ],
            "widgets_values_named": {
                "animation": "ALL",
                "orientation": "ALL",
                "frame": "ALL",
                "prop": "none",
                "align_to_source": True,
                "auto_caption": True,
                "prop2": "none",
                "caption": "",
                "extra": "",
                "pose_list": "",
                "pose_file": "",
                "width": 0,
                "height": 0,
            },
        },
        {
            "id": 6,
            "type": "PreviewImage",
            "pos": [420, 900],
            "size": [280, 260],
            "flags": {},
            "order": 6,
            "mode": 0,
            "inputs": [{"name": "images", "type": "IMAGE", "link": inp(6, 0)}],
            "outputs": [],
            "title": "source_pose (base + squelette detecte)",
            "properties": {"Node name for S&R": "PreviewImage"},
            "widgets_values": [],
        },
        {
            "id": 25,
            "type": "PreviewImage",
            "pos": [720, 900],
            "size": [280, 260],
            "flags": {},
            "order": 6,
            "mode": 0,
            "inputs": [{"name": "images", "type": "IMAGE", "link": inp(25, 0)}],
            "outputs": [],
            "title": "guides (squelettes seuls, live pose en cours)",
            "properties": {"Node name for S&R": "PreviewImage"},
            "widgets_values": [],
        },
        {
            "id": 7,
            "type": "CP_KleinPoseBatch",
            "pos": [840, 80],
            "size": [360, 340],
            "flags": {},
            "order": 7,
            "mode": 0,
            "inputs": [
                {"name": "model", "type": "MODEL", "link": inp(7, 0)},
                {"name": "clip", "type": "CLIP", "link": inp(7, 1)},
                {"name": "vae", "type": "VAE", "link": inp(7, 2)},
                {"name": "image_batch", "type": "IMAGE", "link": inp(7, 3)},
                {"name": "guide", "type": "IMAGE", "link": inp(7, 4)},
                # forceInput: socket only — do NOT declare a widget (avoids widgets_values shift)
                {"name": "edit_prompts", "type": "STRING", "link": inp(7, 5)},
                {"name": "seed", "type": "INT", "widget": {"name": "seed"}, "link": None},
                {"name": "steps", "type": "INT", "widget": {"name": "steps"}, "link": None},
                {"name": "cfg", "type": "FLOAT", "widget": {"name": "cfg"}, "link": None},
                {
                    "name": "sampler_name",
                    "type": "COMBO",
                    "widget": {"name": "sampler_name"},
                    "link": None,
                },
                {
                    "name": "megapixels",
                    "type": "FLOAT",
                    "widget": {"name": "megapixels"},
                    "link": None,
                },
                {
                    "name": "pose_weight",
                    "type": "INT",
                    "widget": {"name": "pose_weight"},
                    "link": None,
                },
                {
                    "name": "filename_prefix",
                    "type": "STRING",
                    "widget": {"name": "filename_prefix"},
                    "link": None,
                },
            ],
            "outputs": [
                {"name": "images", "type": "IMAGE", "links": o[7].get(0), "slot_index": 0},
                {
                    "name": "images_with_guides",
                    "type": "IMAGE",
                    "links": o[7].get(1),
                    "slot_index": 1,
                },
                {"name": "count", "type": "INT", "links": None, "slot_index": 2},
            ],
            "title": "Klein batch (resultats progressifs)",
            "properties": {"Node name for S&R": "CP_KleinPoseBatch"},
            # seed, steps, cfg, sampler, megapixels, pose_weight, prefix
            "widgets_values": [42, 4, 1.0, "euler", 1.0, 2, "pose_transfer_flux_klein"],
            "widgets_values_named": {
                "seed": 42,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "megapixels": 1.0,
                "pose_weight": 2,
                "filename_prefix": "pose_transfer_flux_klein",
            },
        },
        {
            "id": 8,
            "type": "PreviewImage",
            "pos": [1260, 80],
            "size": [360, 360],
            "flags": {},
            "order": 8,
            "mode": 0,
            "inputs": [{"name": "images", "type": "IMAGE", "link": inp(8, 0)}],
            "outputs": [],
            "title": "Results (images seules)",
            "properties": {"Node name for S&R": "PreviewImage"},
            "widgets_values": [],
        },
        {
            "id": 27,
            "type": "PreviewImage",
            "pos": [1640, 80],
            "size": [360, 360],
            "flags": {},
            "order": 8,
            "mode": 0,
            "inputs": [{"name": "images", "type": "IMAGE", "link": inp(27, 0)}],
            "outputs": [],
            "title": "Results + guides (QC pose)",
            "properties": {"Node name for S&R": "PreviewImage"},
            "widgets_values": [],
        },
        {
            "id": 9,
            "type": "SaveImage",
            "pos": [1260, 480],
            "size": [360, 300],
            "flags": {},
            "order": 9,
            "mode": 0,
            "inputs": [{"name": "images", "type": "IMAGE", "link": inp(9, 0)}],
            "outputs": [],
            "properties": {"Node name for S&R": "SaveImage"},
            "widgets_values": ["pose_transfer_flux_klein_all"],
        },
        {
            "id": 10,
            "type": "Note",
            "pos": [40, 760],
            "size": [340, 340],
            "flags": {},
            "order": 10,
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "properties": {"text": ""},
            "widgets_values": [
                "CharacterPose — Flux.2 Klein (Prep)\n\n"
                "1 Queue = N poses SEQUENTIELLES.\n"
                "Prep source_pose = base + squelette detecte (fixe).\n"
                "Prep preview_pose = sticks seuls ;\n"
                "  live pose en cours pendant Klein.\n"
                "Klein images = resultats nets (progressifs).\n"
                "Klein images_with_guides = QC pose.\n"
                "pose_weight: 1 soft … 2 defaut … 4 pose forte.\n"
                "Redemarrer Comfy apres update nodes.\n"
            ],
        },
    ]

    data = {
        "last_node_id": 27,
        "last_link_id": 36,
        "nodes": nodes,
        "links": links,
        "groups": [
            {
                "id": 1,
                "title": "Modeles Flux.2 Klein",
                "bounding": [20, 0, 400, 380],
                "color": "#3f789e",
                "font_size": 24,
                "flags": {},
            },
            {
                "id": 2,
                "title": "CharacterPose Prep",
                "bounding": [400, 360, 420, 700],
                "color": "#88A",
                "font_size": 24,
                "flags": {},
            },
            {
                "id": 3,
                "title": "Klein sequential batch",
                "bounding": [820, 0, 860, 820],
                "color": "#3f789e",
                "font_size": 24,
                "flags": {},
            },
        ],
        "config": {},
        "extra": {
            "info": {
                "name": "CharacterPose — Flux.2 Klein Pose Transfer (Prep)",
                "version": "0.6.0",
                "description": "Prep filters → CP_KleinPoseBatch (per-prompt sequential + progressive preview).",
            }
        },
        "models": [
            {
                "name": "flux-2-klein-4b-fp8.safetensors",
                "url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors",
                "directory": "diffusion_models",
            },
            {
                "name": "qwen_3_4b.safetensors",
                "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors",
                "directory": "text_encoders",
            },
            {
                "name": "flux2-vae.safetensors",
                "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
                "directory": "vae",
            },
        ],
        "version": 0.4,
    }
    path = OUT / "pose_transfer_flux_klein.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {path}")


def write_qwen():
    """Qwen-Image-Edit + ControlNet — single pose, guide scaled to match latent."""
    # 1 UNET  2 CLIP  3 VAE  4 ControlNet  5 LoadImage  6 Prep  7 Preview source
    # 8 CLIP+  9 CLIP-  10 ControlNetApply  11 Scale sprite  12 VAEEncode
    # 13 KSampler  14 VAEDecode  15 Save  16 Note  17 Preview pose
    # 18 DWPreprocessor  19 Scale guide (same MP as sprite)

    def _outs(nid):
        m = {}
        for L in links:
            if L[1] == nid:
                m.setdefault(L[2], []).append(L[0])
        return m

    def inp(nid, slot):
        for L in links:
            if L[3] == nid and L[4] == slot:
                return L[0]
        return None

    links = [
        link(1, 1, 0, 13, 0, "MODEL"),
        link(2, 2, 0, 8, 0, "CLIP"),
        link(3, 2, 0, 9, 0, "CLIP"),
        link(4, 4, 0, 10, 2, "CONTROL_NET"),
        link(5, 5, 0, 6, 0, "IMAGE"),
        link(6, 5, 0, 11, 0, "IMAGE"),
        link(22, 5, 0, 18, 0, "IMAGE"),
        link(23, 18, 1, 6, 7, "POSE_KEYPOINT"),
        link(24, 18, 0, 6, 8, "IMAGE"),
        link(26, 6, 0, 19, 0, "IMAGE"),  # Prep guide -> scale guide
        link(8, 19, 0, 10, 3, "IMAGE"),  # scaled guide -> ControlNet
        link(9, 6, 1, 8, 1, "STRING"),
        link(10, 6, 3, 7, 0, "IMAGE"),
        link(21, 6, 4, 17, 0, "IMAGE"),
        link(11, 8, 0, 10, 0, "CONDITIONING"),
        link(12, 9, 0, 10, 1, "CONDITIONING"),
        link(25, 3, 0, 10, 4, "VAE"),  # VAE required by Qwen ControlNet
        link(13, 10, 0, 13, 1, "CONDITIONING"),
        link(14, 10, 1, 13, 2, "CONDITIONING"),
        link(15, 11, 0, 12, 0, "IMAGE"),
        link(16, 3, 0, 12, 1, "VAE"),
        link(17, 3, 0, 14, 1, "VAE"),
        link(18, 12, 0, 13, 3, "LATENT"),
        link(27, 13, 0, 14, 0, "LATENT"),
        link(20, 14, 0, 15, 0, "IMAGE"),
    ]

    o = {i: _outs(i) for i in range(1, 20)}

    nodes = [
        {
            "id": 1,
            "type": "UNETLoader",
            "pos": [40, 40],
            "size": [400, 82],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {"name": "MODEL", "type": "MODEL", "links": o[1].get(0), "slot_index": 0}
            ],
            "properties": {
                "Node name for S&R": "UNETLoader",
                "models": [
                    {
                        "name": "qwen_image_edit_2511_fp8mixed.safetensors",
                        "url": "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
                        "directory": "diffusion_models",
                    }
                ],
            },
            "widgets_values": ["qwen_image_edit_2511_fp8mixed.safetensors", "default"],
        },
        {
            "id": 2,
            "type": "CLIPLoader",
            "pos": [40, 160],
            "size": [400, 106],
            "flags": {},
            "order": 1,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {"name": "CLIP", "type": "CLIP", "links": o[2].get(0), "slot_index": 0}
            ],
            "properties": {
                "Node name for S&R": "CLIPLoader",
                "models": [
                    {
                        "name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                        "url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
                        "directory": "text_encoders",
                    }
                ],
            },
            "widgets_values": ["qwen_2.5_vl_7b_fp8_scaled.safetensors", "qwen_image", "default"],
        },
        {
            "id": 3,
            "type": "VAELoader",
            "pos": [40, 300],
            "size": [400, 58],
            "flags": {},
            "order": 2,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {"name": "VAE", "type": "VAE", "links": o[3].get(0), "slot_index": 0}
            ],
            "properties": {
                "Node name for S&R": "VAELoader",
                "models": [
                    {
                        "name": "qwen_image_vae.safetensors",
                        "url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
                        "directory": "vae",
                    }
                ],
            },
            "widgets_values": ["qwen_image_vae.safetensors"],
        },
        {
            "id": 4,
            "type": "ControlNetLoader",
            "pos": [40, 400],
            "size": [400, 58],
            "flags": {},
            "order": 3,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {
                    "name": "CONTROL_NET",
                    "type": "CONTROL_NET",
                    "links": o[4].get(0),
                    "slot_index": 0,
                }
            ],
            "properties": {
                "Node name for S&R": "ControlNetLoader",
                "models": [
                    {
                        "name": "Qwen-Image-InstantX-ControlNet-Union.safetensors",
                        "url": "https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union/resolve/main/diffusion_pytorch_model.safetensors",
                        "directory": "controlnet",
                    }
                ],
            },
            "title": "ControlNet Union",
            "widgets_values": ["Qwen-Image-InstantX-ControlNet-Union.safetensors"],
        },
        {
            "id": 5,
            "type": "LoadImage",
            "pos": [40, 520],
            "size": [320, 314],
            "flags": {},
            "order": 4,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": o[5].get(0), "slot_index": 0},
                {"name": "MASK", "type": "MASK", "links": None, "slot_index": 1},
            ],
            "title": "Sprite de reference",
            "properties": {"Node name for S&R": "LoadImage"},
            "widgets_values": ["Example_Comfy.png", "image"],
        },
        {
            "id": 6,
            "type": "CP_PoseTransferPrep",
            "pos": [420, 520],
            "size": [360, 420],
            "flags": {},
            "order": 5,
            "mode": 0,
            "inputs": [
                {"name": "image", "type": "IMAGE", "link": inp(6, 0)},
                {"name": "pose_keypoint", "type": "POSE_KEYPOINT", "link": inp(6, 7)},
                {"name": "dwpose_image", "type": "IMAGE", "link": inp(6, 8)},
                {"name": "prop2", "type": "COMBO", "widget": {"name": "prop2"}, "link": None},
                {"name": "caption", "type": "STRING", "widget": {"name": "caption"}, "link": None},
                {"name": "extra", "type": "STRING", "widget": {"name": "extra"}, "link": None},
                {"name": "source_pose", "type": "POSE", "link": None},
                {"name": "pose_list", "type": "STRING", "widget": {"name": "pose_list"}, "link": None},
                {"name": "pose_file", "type": "STRING", "widget": {"name": "pose_file"}, "link": None},
                {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": None},
                {"name": "height", "type": "INT", "widget": {"name": "height"}, "link": None},
            ],
            "outputs": [
                {"name": "guide", "type": "IMAGE", "links": o[6].get(0), "slot_index": 0},
                {"name": "edit_prompt", "type": "STRING", "links": o[6].get(1), "slot_index": 1},
                {"name": "pose", "type": "POSE", "links": None, "slot_index": 2},
                {"name": "source_pose", "type": "IMAGE", "links": o[6].get(3), "slot_index": 3},
                {"name": "preview_pose", "type": "IMAGE", "links": o[6].get(4), "slot_index": 4},
                {"name": "caption", "type": "STRING", "links": None, "slot_index": 5},
                {"name": "caption_backend", "type": "STRING", "links": None, "slot_index": 6},
                {"name": "edit_prompts", "type": "STRING", "links": None, "slot_index": 7},
                {"name": "pose_names", "type": "STRING", "links": None, "slot_index": 8},
                {"name": "image_batch", "type": "IMAGE", "links": None, "slot_index": 9},
                {"name": "count", "type": "INT", "links": None, "slot_index": 10},
            ],
            "title": "Prep (single pose → guide)",
            "properties": {"Node name for S&R": "CP_PoseTransferPrep"},
            "widgets_values": [
                "ALL",
                "ALL",
                "01",
                "none",
                True,
                True,
                "none",
                "",
                "",
                "",
                "",
                0,
                0,
            ],
            "widgets_values_named": {
                "animation": "ALL",
                "orientation": "ALL",
                "frame": "01",
                "prop": "none",
                "align_to_source": True,
                "auto_caption": True,
                "prop2": "none",
                "caption": "",
                "extra": "",
                "pose_list": "",
                "pose_file": "",
                "width": 0,
                "height": 0,
            },
        },
        {
            "id": 18,
            "type": "DWPreprocessor",
            "pos": [400, 280],
            "size": [315, 200],
            "flags": {},
            "order": 4,
            "mode": 0,
            "inputs": [{"name": "image", "type": "IMAGE", "link": inp(18, 0)}],
            "outputs": [
                {
                    "name": "IMAGE",
                    "type": "IMAGE",
                    "links": o[18].get(0),
                    "slot_index": 0,
                },
                {
                    "name": "POSE_KEYPOINT",
                    "type": "POSE_KEYPOINT",
                    "links": o[18].get(1),
                    "slot_index": 1,
                },
            ],
            "title": "DWPose (source skeleton)",
            "properties": {"Node name for S&R": "DWPreprocessor"},
            "widgets_values": [
                "enable",
                "enable",
                "enable",
                1024,
                "yolox_l.onnx",
                "dw-ll_ucoco_384.onnx",
                "disable",
            ],
        },
        {
            "id": 7,
            "type": "PreviewImage",
            "pos": [420, 900],
            "size": [280, 260],
            "flags": {},
            "order": 6,
            "mode": 0,
            "inputs": [{"name": "images", "type": "IMAGE", "link": inp(7, 0)}],
            "outputs": [],
            "title": "source_pose (sprite + squelette)",
            "properties": {"Node name for S&R": "PreviewImage"},
            "widgets_values": [],
        },
        {
            "id": 17,
            "type": "PreviewImage",
            "pos": [720, 900],
            "size": [280, 260],
            "flags": {},
            "order": 6,
            "mode": 0,
            "inputs": [{"name": "images", "type": "IMAGE", "link": inp(17, 0)}],
            "outputs": [],
            "title": "preview_pose (pose sortie)",
            "properties": {"Node name for S&R": "PreviewImage"},
            "widgets_values": [],
        },
        {
            "id": 8,
            "type": "CLIPTextEncode",
            "pos": [840, 40],
            "size": [480, 160],
            "flags": {},
            "order": 7,
            "mode": 0,
            "inputs": [
                {"name": "clip", "type": "CLIP", "link": inp(8, 0)},
                {"name": "text", "type": "STRING", "link": inp(8, 1), "widget": {"name": "text"}},
            ],
            "outputs": [
                {
                    "name": "CONDITIONING",
                    "type": "CONDITIONING",
                    "links": o[8].get(0),
                    "slot_index": 0,
                }
            ],
            "title": "Positive (prompt auto Prep)",
            "properties": {"Node name for S&R": "CLIPTextEncode"},
            "widgets_values": [""],
        },
        {
            "id": 9,
            "type": "CLIPTextEncode",
            "pos": [840, 240],
            "size": [480, 120],
            "flags": {},
            "order": 8,
            "mode": 0,
            "inputs": [{"name": "clip", "type": "CLIP", "link": inp(9, 0)}],
            "outputs": [
                {
                    "name": "CONDITIONING",
                    "type": "CONDITIONING",
                    "links": o[9].get(0),
                    "slot_index": 0,
                }
            ],
            "title": "Negative",
            "properties": {"Node name for S&R": "CLIPTextEncode"},
            "widgets_values": [
                "bones, skeleton, x-ray, openpose sticks, extra limbs, blurry, low quality"
            ],
        },
        {
            "id": 10,
            "type": "ControlNetApplyAdvanced",
            "pos": [1380, 80],
            "size": [320, 200],
            "flags": {},
            "order": 9,
            "mode": 0,
            "inputs": [
                {"name": "positive", "type": "CONDITIONING", "link": inp(10, 0)},
                {"name": "negative", "type": "CONDITIONING", "link": inp(10, 1)},
                {"name": "control_net", "type": "CONTROL_NET", "link": inp(10, 2)},
                {"name": "image", "type": "IMAGE", "link": inp(10, 3)},
                {"name": "vae", "type": "VAE", "link": inp(10, 4)},
            ],
            "outputs": [
                {
                    "name": "positive",
                    "type": "CONDITIONING",
                    "links": o[10].get(0),
                    "slot_index": 0,
                },
                {
                    "name": "negative",
                    "type": "CONDITIONING",
                    "links": o[10].get(1),
                    "slot_index": 1,
                },
            ],
            "title": "Apply ControlNet (OpenPose guide)",
            "properties": {"Node name for S&R": "ControlNetApplyAdvanced"},
            "widgets_values": [1.4, 0.0, 1.0],
        },
        {
            "id": 11,
            "type": "ImageScaleToTotalPixels",
            "pos": [840, 520],
            "size": [300, 106],
            "flags": {},
            "order": 10,
            "mode": 0,
            "inputs": [{"name": "image", "type": "IMAGE", "link": inp(11, 0)}],
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": o[11].get(0), "slot_index": 0}
            ],
            "title": "Scale sprite",
            "properties": {"Node name for S&R": "ImageScaleToTotalPixels"},
            "widgets_values": ["lanczos", 1.0, 64],
        },
        {
            "id": 19,
            "type": "ImageScaleToTotalPixels",
            "pos": [840, 660],
            "size": [300, 106],
            "flags": {},
            "order": 10,
            "mode": 0,
            "inputs": [{"name": "image", "type": "IMAGE", "link": inp(19, 0)}],
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": o[19].get(0), "slot_index": 0}
            ],
            "title": "Scale guide (match latent)",
            "properties": {"Node name for S&R": "ImageScaleToTotalPixels"},
            "widgets_values": ["lanczos", 1.0, 64],
        },
        {
            "id": 12,
            "type": "VAEEncode",
            "pos": [1180, 520],
            "size": [210, 46],
            "flags": {},
            "order": 11,
            "mode": 0,
            "inputs": [
                {"name": "pixels", "type": "IMAGE", "link": inp(12, 0)},
                {"name": "vae", "type": "VAE", "link": inp(12, 1)},
            ],
            "outputs": [
                {"name": "LATENT", "type": "LATENT", "links": o[12].get(0), "slot_index": 0}
            ],
            "title": "Encode sprite latent",
            "properties": {"Node name for S&R": "VAEEncode"},
            "widgets_values": [],
        },
        {
            "id": 13,
            "type": "KSampler",
            "pos": [1740, 80],
            "size": [320, 460],
            "flags": {},
            "order": 12,
            "mode": 0,
            "inputs": [
                {"name": "model", "type": "MODEL", "link": inp(13, 0)},
                {"name": "positive", "type": "CONDITIONING", "link": inp(13, 1)},
                {"name": "negative", "type": "CONDITIONING", "link": inp(13, 2)},
                {"name": "latent_image", "type": "LATENT", "link": inp(13, 3)},
            ],
            "outputs": [
                {"name": "LATENT", "type": "LATENT", "links": o[13].get(0), "slot_index": 0}
            ],
            "properties": {"Node name for S&R": "KSampler"},
            "widgets_values": [42, "randomize", 20, 3.5, "euler", "simple", 1.0],
        },
        {
            "id": 14,
            "type": "VAEDecode",
            "pos": [2100, 80],
            "size": [220, 46],
            "flags": {},
            "order": 13,
            "mode": 0,
            "inputs": [
                {"name": "samples", "type": "LATENT", "link": inp(14, 0)},
                {"name": "vae", "type": "VAE", "link": inp(14, 1)},
            ],
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": o[14].get(0), "slot_index": 0}
            ],
            "properties": {"Node name for S&R": "VAEDecode"},
            "widgets_values": [],
        },
        {
            "id": 15,
            "type": "SaveImage",
            "pos": [2100, 180],
            "size": [360, 400],
            "flags": {},
            "order": 14,
            "mode": 0,
            "inputs": [{"name": "images", "type": "IMAGE", "link": inp(15, 0)}],
            "outputs": [],
            "properties": {"Node name for S&R": "SaveImage"},
            "widgets_values": ["pose_transfer_qwen_cn"],
        },
        {
            "id": 16,
            "type": "Note",
            "pos": [40, 880],
            "size": [360, 280],
            "flags": {},
            "order": 15,
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "properties": {"text": ""},
            "widgets_values": [
                "CharacterPose — Qwen + ControlNet (single pose)\n\n"
                "frame=01 (pas ALL) — multi-pose = Klein batch\n"
                "guide scale 1 MP = meme res que sprite latent\n"
                "VAE branche sur ControlNetApplyAdvanced\n"
                "UNET = qwen_image_edit_2511 (edit)\n"
            ],
        },
    ]

    data = {
        "last_node_id": 19,
        "last_link_id": 27,
        "nodes": nodes,
        "links": links,
        "groups": [
            {
                "id": 1,
                "title": "Modeles Qwen + ControlNet",
                "bounding": [20, 0, 420, 480],
                "color": "#3f789e",
                "font_size": 24,
                "flags": {},
            },
            {
                "id": 2,
                "title": "CharacterPose Prep",
                "bounding": [400, 480, 420, 700],
                "color": "#88A",
                "font_size": 24,
                "flags": {},
            },
            {
                "id": 3,
                "title": "Sample (lock pose)",
                "bounding": [1360, 40, 1140, 560],
                "color": "#3f789e",
                "font_size": 24,
                "flags": {},
            },
        ],
        "config": {},
        "extra": {
            "info": {
                "name": "CharacterPose — Qwen-Image-Edit + ControlNet",
                "version": "0.5.2",
                "description": "Single-pose Prep → scaled guide + VAE → ControlNet → KSampler.",
            }
        },
        "models": [
            {
                "name": "qwen_image_edit_2511_fp8mixed.safetensors",
                "url": "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
                "directory": "diffusion_models",
            },
            {
                "name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "directory": "text_encoders",
            },
            {
                "name": "qwen_image_vae.safetensors",
                "url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
                "directory": "vae",
            },
            {
                "name": "Qwen-Image-InstantX-ControlNet-Union.safetensors",
                "url": "https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union/resolve/main/diffusion_pytorch_model.safetensors",
                "directory": "controlnet",
            },
        ],
        "version": 0.4,
    }
    path = OUT / "pose_transfer_qwen_controlnet.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {path}")


def validate(path: Path) -> None:
    """Sanity-check link ids exist; destination slots need not equal the
    sparse inputs[] index (ComfyUI slots follow full INPUT_TYPES order).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in data["nodes"]}
    links = {L[0]: L for L in data["links"]}
    errors = []
    for nid, n in nodes.items():
        for inp in n.get("inputs") or []:
            lid = inp.get("link")
            if lid is None:
                continue
            if lid not in links:
                errors.append(f"node {nid}/{inp['name']}: missing link {lid}")
                continue
            L = links[lid]
            if L[3] != nid:
                errors.append(
                    f"link {lid}: expected dst node {nid} got {L[3]} for {inp['name']}"
                )
        for oi, out in enumerate(n.get("outputs") or []):
            for lid in out.get("links") or []:
                if lid not in links:
                    errors.append(f"node {nid}/{out['name']}: missing out link {lid}")
                    continue
                L = links[lid]
                if L[1] != nid or L[2] != oi:
                    errors.append(
                        f"link {lid}: expected src ({nid},{oi}) got ({L[1]},{L[2]})"
                    )
    for lid, L in links.items():
        if L[1] not in nodes or L[3] not in nodes:
            errors.append(f"link {lid}: dangling node")
    print(path.name, "OK" if not errors else "FAIL")
    for e in errors:
        print(" ", e)


if __name__ == "__main__":
    write_flux_klein()
    write_qwen()
    validate(OUT / "pose_transfer_flux_klein.json")
    validate(OUT / "pose_transfer_qwen_controlnet.json")
