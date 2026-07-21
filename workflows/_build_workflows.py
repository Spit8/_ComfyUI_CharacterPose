"""Generate valid ComfyUI workflow JSON with consistent link wiring."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def link(lid, src, src_slot, dst, dst_slot, typ):
    return [lid, src, src_slot, dst, dst_slot, typ]


def write_flux_klein():
    """Flux.2 Klein + CP_PoseTransferPrep — fully wired."""
    # Node IDs
    # 1 UNET  2 CLIP  3 VAE  4 LoadImage  5 Prep  6 Preview
    # 7 CLIPTextEncode(+prompt)  8 ZeroOut
    # 9 ScaleSprite  10 GetSize  11 VAEEnc1  12 RefLat1
    # 13 ScaleGuide  14 VAEEnc2  15 RefLat2
    # 16 EmptyLatent  17 Scheduler  18 CFG  19 SamplerSel  20 Noise
    # 21 Sampler  22 Decode  23 Save  24 Note

    links = [
        # models
        link(1, 1, 0, 18, 0, "MODEL"),
        link(2, 2, 0, 7, 0, "CLIP"),
        link(3, 3, 0, 11, 1, "VAE"),
        link(4, 3, 0, 14, 1, "VAE"),
        link(5, 3, 0, 22, 1, "VAE"),
        # image / prep
        link(6, 4, 0, 5, 0, "IMAGE"),
        link(7, 4, 0, 9, 0, "IMAGE"),
        link(8, 5, 0, 13, 0, "IMAGE"),  # guide -> scale2
        link(9, 5, 1, 7, 1, "STRING"),  # edit_prompt -> CLIPTextEncode.text
        link(10, 5, 3, 6, 0, "IMAGE"),  # preview
        # conditioning chain
        link(11, 7, 0, 12, 0, "CONDITIONING"),
        link(12, 7, 0, 8, 0, "CONDITIONING"),
        link(13, 8, 0, 18, 2, "CONDITIONING"),
        # sprite encode
        link(14, 9, 0, 10, 0, "IMAGE"),
        link(15, 9, 0, 11, 0, "IMAGE"),
        link(16, 10, 0, 17, 0, "INT"),
        link(17, 10, 0, 16, 0, "INT"),
        link(18, 10, 1, 17, 1, "INT"),
        link(19, 10, 1, 16, 1, "INT"),
        link(20, 11, 0, 12, 1, "LATENT"),
        link(21, 12, 0, 15, 0, "CONDITIONING"),
        # guide encode
        link(22, 13, 0, 14, 0, "IMAGE"),
        link(23, 14, 0, 15, 1, "LATENT"),
        link(24, 15, 0, 18, 1, "CONDITIONING"),
        # sample
        link(25, 16, 0, 21, 4, "LATENT"),
        link(26, 17, 0, 21, 3, "SIGMAS"),
        link(27, 18, 0, 21, 1, "GUIDER"),
        link(28, 19, 0, 21, 2, "SAMPLER"),
        link(29, 20, 0, 21, 0, "NOISE"),
        link(30, 21, 0, 22, 0, "LATENT"),
        link(31, 22, 0, 23, 0, "IMAGE"),
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

    o = {i: outs(i) for i in range(1, 25)}

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
            "id": 5,
            "type": "CP_PoseTransferPrep",
            "pos": [420, 400],
            "size": [360, 340],
            "flags": {},
            "order": 4,
            "mode": 0,
            "inputs": [
                {"name": "image", "type": "IMAGE", "link": inp(5, 0)},
                {"name": "source_pose", "type": "POSE", "link": None},
            ],
            "outputs": [
                {"name": "guide", "type": "IMAGE", "links": o[5].get(0), "slot_index": 0},
                {"name": "edit_prompt", "type": "STRING", "links": o[5].get(1), "slot_index": 1},
                {"name": "pose", "type": "POSE", "links": None, "slot_index": 2},
                {"name": "preview", "type": "IMAGE", "links": o[5].get(3), "slot_index": 3},
                {"name": "caption", "type": "STRING", "links": None, "slot_index": 4},
            ],
            "title": "Prep: caption + pose 3D + props",
            "properties": {"Node name for S&R": "CP_PoseTransferPrep"},
            "widgets_values": [
                "walk_01",
                "SE",
                "none",
                True,
                True,
                "none",
                0.0,
                0.0,
                0.0,
                False,
                "",
                "",
                0,
                0,
            ],
        },
        {
            "id": 6,
            "type": "PreviewImage",
            "pos": [420, 780],
            "size": [280, 260],
            "flags": {},
            "order": 5,
            "mode": 0,
            "inputs": [{"name": "images", "type": "IMAGE", "link": inp(6, 0)}],
            "outputs": [],
            "title": "Preview pose overlay",
            "properties": {"Node name for S&R": "PreviewImage"},
            "widgets_values": [],
        },
        {
            "id": 7,
            "type": "CLIPTextEncode",
            "pos": [840, 40],
            "size": [520, 160],
            "flags": {},
            "order": 6,
            "mode": 0,
            "inputs": [
                {"name": "clip", "type": "CLIP", "link": inp(7, 0)},
                {"name": "text", "type": "STRING", "link": inp(7, 1), "widget": {"name": "text"}},
            ],
            "outputs": [
                {
                    "name": "CONDITIONING",
                    "type": "CONDITIONING",
                    "links": o[7].get(0),
                    "slot_index": 0,
                }
            ],
            "title": "Prompt auto (depuis Prep)",
            "properties": {"Node name for S&R": "CLIPTextEncode"},
            "widgets_values": [""],
        },
        {
            "id": 8,
            "type": "ConditioningZeroOut",
            "pos": [1400, 40],
            "size": [240, 46],
            "flags": {},
            "order": 7,
            "mode": 0,
            "inputs": [{"name": "conditioning", "type": "CONDITIONING", "link": inp(8, 0)}],
            "outputs": [
                {
                    "name": "CONDITIONING",
                    "type": "CONDITIONING",
                    "links": o[8].get(0),
                    "slot_index": 0,
                }
            ],
            "properties": {"Node name for S&R": "ConditioningZeroOut"},
            "widgets_values": [],
        },
        {
            "id": 9,
            "type": "ImageScaleToTotalPixels",
            "pos": [840, 280],
            "size": [300, 106],
            "flags": {},
            "order": 8,
            "mode": 0,
            "inputs": [{"name": "image", "type": "IMAGE", "link": inp(9, 0)}],
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": o[9].get(0), "slot_index": 0}
            ],
            "title": "Scale ref 1 (sprite)",
            "properties": {"Node name for S&R": "ImageScaleToTotalPixels"},
            "widgets_values": ["lanczos", 1.0, 64],
        },
        {
            "id": 10,
            "type": "GetImageSize",
            "pos": [1180, 280],
            "size": [210, 66],
            "flags": {},
            "order": 9,
            "mode": 0,
            "inputs": [{"name": "image", "type": "IMAGE", "link": inp(10, 0)}],
            "outputs": [
                {"name": "width", "type": "INT", "links": o[10].get(0), "slot_index": 0},
                {"name": "height", "type": "INT", "links": o[10].get(1), "slot_index": 1},
                {"name": "batch_size", "type": "INT", "links": None, "slot_index": 2},
            ],
            "properties": {"Node name for S&R": "GetImageSize"},
            "widgets_values": [],
        },
        {
            "id": 11,
            "type": "VAEEncode",
            "pos": [1180, 380],
            "size": [210, 46],
            "flags": {},
            "order": 10,
            "mode": 0,
            "inputs": [
                {"name": "pixels", "type": "IMAGE", "link": inp(11, 0)},
                {"name": "vae", "type": "VAE", "link": inp(11, 1)},
            ],
            "outputs": [
                {"name": "LATENT", "type": "LATENT", "links": o[11].get(0), "slot_index": 0}
            ],
            "title": "Encode ref 1",
            "properties": {"Node name for S&R": "VAEEncode"},
            "widgets_values": [],
        },
        {
            "id": 12,
            "type": "ReferenceLatent",
            "pos": [1400, 140],
            "size": [260, 46],
            "flags": {},
            "order": 11,
            "mode": 0,
            "inputs": [
                {"name": "conditioning", "type": "CONDITIONING", "link": inp(12, 0)},
                {"name": "latent", "type": "LATENT", "link": inp(12, 1)},
            ],
            "outputs": [
                {
                    "name": "CONDITIONING",
                    "type": "CONDITIONING",
                    "links": o[12].get(0),
                    "slot_index": 0,
                }
            ],
            "title": "Ref latent 1 (sprite)",
            "properties": {"Node name for S&R": "ReferenceLatent"},
            "widgets_values": [],
        },
        {
            "id": 13,
            "type": "ImageScaleToTotalPixels",
            "pos": [840, 520],
            "size": [300, 106],
            "flags": {},
            "order": 12,
            "mode": 0,
            "inputs": [{"name": "image", "type": "IMAGE", "link": inp(13, 0)}],
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": o[13].get(0), "slot_index": 0}
            ],
            "title": "Scale ref 2 (guide pose)",
            "properties": {"Node name for S&R": "ImageScaleToTotalPixels"},
            "widgets_values": ["lanczos", 1.0, 64],
        },
        {
            "id": 14,
            "type": "VAEEncode",
            "pos": [1180, 520],
            "size": [210, 46],
            "flags": {},
            "order": 13,
            "mode": 0,
            "inputs": [
                {"name": "pixels", "type": "IMAGE", "link": inp(14, 0)},
                {"name": "vae", "type": "VAE", "link": inp(14, 1)},
            ],
            "outputs": [
                {"name": "LATENT", "type": "LATENT", "links": o[14].get(0), "slot_index": 0}
            ],
            "title": "Encode ref 2",
            "properties": {"Node name for S&R": "VAEEncode"},
            "widgets_values": [],
        },
        {
            "id": 15,
            "type": "ReferenceLatent",
            "pos": [1700, 140],
            "size": [260, 46],
            "flags": {},
            "order": 14,
            "mode": 0,
            "inputs": [
                {"name": "conditioning", "type": "CONDITIONING", "link": inp(15, 0)},
                {"name": "latent", "type": "LATENT", "link": inp(15, 1)},
            ],
            "outputs": [
                {
                    "name": "CONDITIONING",
                    "type": "CONDITIONING",
                    "links": o[15].get(0),
                    "slot_index": 0,
                }
            ],
            "title": "Ref latent 2 (guide)",
            "properties": {"Node name for S&R": "ReferenceLatent"},
            "widgets_values": [],
        },
        {
            "id": 16,
            "type": "EmptyFlux2LatentImage",
            "pos": [1700, 280],
            "size": [270, 106],
            "flags": {},
            "order": 15,
            "mode": 0,
            "inputs": [
                {"name": "width", "type": "INT", "link": inp(16, 0), "widget": {"name": "width"}},
                {"name": "height", "type": "INT", "link": inp(16, 1), "widget": {"name": "height"}},
            ],
            "outputs": [
                {"name": "LATENT", "type": "LATENT", "links": o[16].get(0), "slot_index": 0}
            ],
            "properties": {"Node name for S&R": "EmptyFlux2LatentImage"},
            "widgets_values": [1024, 1024, 1],
        },
        {
            "id": 17,
            "type": "Flux2Scheduler",
            "pos": [1700, 420],
            "size": [270, 106],
            "flags": {},
            "order": 16,
            "mode": 0,
            "inputs": [
                {"name": "width", "type": "INT", "link": inp(17, 0), "widget": {"name": "width"}},
                {"name": "height", "type": "INT", "link": inp(17, 1), "widget": {"name": "height"}},
            ],
            "outputs": [
                {"name": "SIGMAS", "type": "SIGMAS", "links": o[17].get(0), "slot_index": 0}
            ],
            "properties": {"Node name for S&R": "Flux2Scheduler"},
            "widgets_values": [4, 1024, 1024],
        },
        {
            "id": 18,
            "type": "CFGGuider",
            "pos": [2000, 40],
            "size": [270, 98],
            "flags": {},
            "order": 17,
            "mode": 0,
            "inputs": [
                {"name": "model", "type": "MODEL", "link": inp(18, 0)},
                {"name": "positive", "type": "CONDITIONING", "link": inp(18, 1)},
                {"name": "negative", "type": "CONDITIONING", "link": inp(18, 2)},
            ],
            "outputs": [
                {"name": "GUIDER", "type": "GUIDER", "links": o[18].get(0), "slot_index": 0}
            ],
            "properties": {"Node name for S&R": "CFGGuider"},
            "widgets_values": [1.0],
        },
        {
            "id": 19,
            "type": "KSamplerSelect",
            "pos": [2000, 180],
            "size": [270, 58],
            "flags": {},
            "order": 18,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {"name": "SAMPLER", "type": "SAMPLER", "links": o[19].get(0), "slot_index": 0}
            ],
            "properties": {"Node name for S&R": "KSamplerSelect"},
            "widgets_values": ["euler"],
        },
        {
            "id": 20,
            "type": "RandomNoise",
            "pos": [2000, 280],
            "size": [270, 82],
            "flags": {},
            "order": 19,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {"name": "NOISE", "type": "NOISE", "links": o[20].get(0), "slot_index": 0}
            ],
            "properties": {"Node name for S&R": "RandomNoise"},
            "widgets_values": [42, "randomize"],
        },
        {
            "id": 21,
            "type": "SamplerCustomAdvanced",
            "pos": [2320, 120],
            "size": [280, 106],
            "flags": {},
            "order": 20,
            "mode": 0,
            "inputs": [
                {"name": "noise", "type": "NOISE", "link": inp(21, 0)},
                {"name": "guider", "type": "GUIDER", "link": inp(21, 1)},
                {"name": "sampler", "type": "SAMPLER", "link": inp(21, 2)},
                {"name": "sigmas", "type": "SIGMAS", "link": inp(21, 3)},
                {"name": "latent_image", "type": "LATENT", "link": inp(21, 4)},
            ],
            "outputs": [
                {"name": "output", "type": "LATENT", "links": o[21].get(0), "slot_index": 0},
                {"name": "denoised_output", "type": "LATENT", "links": None, "slot_index": 1},
            ],
            "properties": {"Node name for S&R": "SamplerCustomAdvanced"},
            "widgets_values": [],
        },
        {
            "id": 22,
            "type": "VAEDecode",
            "pos": [2640, 120],
            "size": [220, 46],
            "flags": {},
            "order": 21,
            "mode": 0,
            "inputs": [
                {"name": "samples", "type": "LATENT", "link": inp(22, 0)},
                {"name": "vae", "type": "VAE", "link": inp(22, 1)},
            ],
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": o[22].get(0), "slot_index": 0}
            ],
            "properties": {"Node name for S&R": "VAEDecode"},
            "widgets_values": [],
        },
        {
            "id": 23,
            "type": "SaveImage",
            "pos": [2640, 220],
            "size": [360, 400],
            "flags": {},
            "order": 22,
            "mode": 0,
            "inputs": [{"name": "images", "type": "IMAGE", "link": inp(23, 0)}],
            "outputs": [],
            "properties": {"Node name for S&R": "SaveImage"},
            "widgets_values": ["pose_transfer_flux_klein"],
        },
        {
            "id": 24,
            "type": "Note",
            "pos": [40, 760],
            "size": [340, 280],
            "flags": {},
            "order": 23,
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "properties": {"text": ""},
            "widgets_values": [
                "CharacterPose — Flux.2 Klein (Prep)\n\n"
                "LoadImage → CP_PoseTransferPrep → guide + edit_prompt\n"
                "ref1 = sprite | ref2 = OpenPose guide\n"
                "Prompt auto branche sur CLIPTextEncode.\n\n"
                "Hard lock: pose_transfer_qwen_controlnet.json"
            ],
        },
    ]

    data = {
        "last_node_id": 24,
        "last_link_id": 31,
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
                "title": "Sampler Flux.2",
                "bounding": [1660, 0, 1380, 520],
                "color": "#3f789e",
                "font_size": 24,
                "flags": {},
            },
        ],
        "config": {},
        "extra": {
            "info": {
                "name": "CharacterPose — Flux.2 Klein Pose Transfer (Prep)",
                "version": "0.5.1",
                "description": "Fully wired: Prep → dual ReferenceLatent Flux Klein.",
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
    """Qwen-Image-Edit + ControlNet — fully wired, no duplicate links."""
    # 1 UNET  2 CLIP  3 VAE  4 ControlNet  5 LoadImage  6 Prep  7 Preview
    # 8 CLIPTextEncode positive (prompt from prep)
    # 9 CLIPTextEncode negative
    # 10 ControlNetApplyAdvanced
    # 11 ImageScale  12 VAEEncode  13 KSampler  14 VAEDecode  15 Save  16 Note

    links = [
        link(1, 1, 0, 13, 0, "MODEL"),
        link(2, 2, 0, 8, 0, "CLIP"),
        link(3, 2, 0, 9, 0, "CLIP"),
        link(4, 4, 0, 10, 2, "CONTROL_NET"),
        link(5, 5, 0, 6, 0, "IMAGE"),
        link(6, 5, 0, 11, 0, "IMAGE"),
        link(7, 5, 0, 8, 2, "IMAGE"),  # sprite -> encode (optional image slot if present)
        link(8, 6, 0, 10, 3, "IMAGE"),  # guide -> controlnet image
        link(9, 6, 1, 8, 1, "STRING"),  # edit_prompt -> positive text
        link(10, 6, 3, 7, 0, "IMAGE"),  # preview
        link(11, 8, 0, 10, 0, "CONDITIONING"),
        link(12, 9, 0, 10, 1, "CONDITIONING"),
        link(13, 10, 0, 13, 1, "CONDITIONING"),
        link(14, 10, 1, 13, 2, "CONDITIONING"),
        link(15, 11, 0, 12, 0, "IMAGE"),
        link(16, 3, 0, 12, 1, "VAE"),
        link(17, 3, 0, 14, 1, "VAE"),
        link(18, 12, 0, 13, 3, "LATENT"),
        link(19, 13, 0, 14, 0, "LATENT"),
        link(20, 14, 0, 15, 0, "IMAGE"),
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

    o = {i: outs(i) for i in range(1, 17)}

    # For CLIPTextEncode positive: clip=0, text=1, and optionally we won't use image on CLIPTextEncode
    # Remove link 7 to CLIPTextEncode image - CLIPTextEncode doesn't have image input!
    # Fix links: remove the bogus sprite->encode image link
    links = [L for L in links if L[0] != 7]
    # renumber? Keep gap is ok for ComfyUI. Better rebuild cleanly.

    links = [
        link(1, 1, 0, 13, 0, "MODEL"),
        link(2, 2, 0, 8, 0, "CLIP"),
        link(3, 2, 0, 9, 0, "CLIP"),
        link(4, 4, 0, 10, 2, "CONTROL_NET"),
        link(5, 5, 0, 6, 0, "IMAGE"),
        link(6, 5, 0, 11, 0, "IMAGE"),
        link(8, 6, 0, 10, 3, "IMAGE"),
        link(9, 6, 1, 8, 1, "STRING"),
        link(10, 6, 3, 7, 0, "IMAGE"),
        link(11, 8, 0, 10, 0, "CONDITIONING"),
        link(12, 9, 0, 10, 1, "CONDITIONING"),
        link(13, 10, 0, 13, 1, "CONDITIONING"),
        link(14, 10, 1, 13, 2, "CONDITIONING"),
        link(15, 11, 0, 12, 0, "IMAGE"),
        link(16, 3, 0, 12, 1, "VAE"),
        link(17, 3, 0, 14, 1, "VAE"),
        link(18, 12, 0, 13, 3, "LATENT"),
        link(19, 13, 0, 14, 0, "LATENT"),
        link(20, 14, 0, 15, 0, "IMAGE"),
    ]

    o = {i: outs(i) for i in range(1, 17)}

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
                        "name": "qwen_image_edit_2511_fp8_e4m3fn.safetensors",
                        "url": "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8_e4m3fn.safetensors",
                        "directory": "diffusion_models",
                    }
                ],
            },
            "widgets_values": ["qwen_image_edit_2511_fp8_e4m3fn.safetensors", "default"],
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
            "size": [360, 340],
            "flags": {},
            "order": 5,
            "mode": 0,
            "inputs": [
                {"name": "image", "type": "IMAGE", "link": inp(6, 0)},
                {"name": "source_pose", "type": "POSE", "link": None},
            ],
            "outputs": [
                {"name": "guide", "type": "IMAGE", "links": o[6].get(0), "slot_index": 0},
                {"name": "edit_prompt", "type": "STRING", "links": o[6].get(1), "slot_index": 1},
                {"name": "pose", "type": "POSE", "links": None, "slot_index": 2},
                {"name": "preview", "type": "IMAGE", "links": o[6].get(3), "slot_index": 3},
                {"name": "caption", "type": "STRING", "links": None, "slot_index": 4},
            ],
            "title": "Prep (caption + pose 3D + props)",
            "properties": {"Node name for S&R": "CP_PoseTransferPrep"},
            "widgets_values": [
                "idle",
                "SE",
                "sword",
                True,
                True,
                "none",
                0.0,
                0.0,
                0.0,
                False,
                "",
                "",
                0,
                0,
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
            "title": "Preview overlay",
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
                "CharacterPose — Qwen + ControlNet\n\n"
                "LoadImage → Prep → guide OpenPose\n"
                "→ ControlNetApplyAdvanced (strength ~1.4)\n"
                "Prompt auto depuis Prep.\n\n"
                "Si besoin d'un encodeur Qwen Edit natif\n"
                "(TextEncodeQwenImageEdit), remplace le\n"
                "CLIPTextEncode positive et rebranche\n"
                "edit_prompt + sprite + VAE."
            ],
        },
    ]

    data = {
        "last_node_id": 16,
        "last_link_id": 20,
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
                "version": "0.5.1",
                "description": "Fully wired Prep → ControlNet OpenPose → KSampler.",
            }
        },
        "models": [
            {
                "name": "qwen_image_edit_2511_fp8_e4m3fn.safetensors",
                "url": "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8_e4m3fn.safetensors",
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
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in data["nodes"]}
    links = {L[0]: L for L in data["links"]}
    errors = []
    for nid, n in nodes.items():
        for i, inp in enumerate(n.get("inputs") or []):
            lid = inp.get("link")
            if lid is None:
                continue
            if lid not in links:
                errors.append(f"node {nid}/{inp['name']}: missing link {lid}")
                continue
            L = links[lid]
            if L[3] != nid or L[4] != i:
                errors.append(
                    f"link {lid}: expected dst ({nid},{i}) got ({L[3]},{L[4]}) for {inp['name']}"
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
