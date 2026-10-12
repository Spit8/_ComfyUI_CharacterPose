"""Build ComfyUI frontend workflows for CharacterPose.

- pose_transfer_flux_klein.json — Prep + Klein batch
- workflow_mixamo.json — Mixamo export + pose library preview

    python workflows/_build_workflows.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
TARGET = OUT / "pose_transfer_flux_klein.json"
TARGET_MIXAMO = OUT / "workflow_mixamo.json"

MODELS = [
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
]


def link(lid, src, src_slot, dst, dst_slot, typ):
    return [lid, src, src_slot, dst, dst_slot, typ]


def _outs(links, nid):
    m: dict[int, list[int]] = {}
    for L in links:
        if L[1] == nid:
            m.setdefault(L[2], []).append(L[0])
    return m


def _inp(links, nid, slot):
    for L in links:
        if L[3] == nid and L[4] == slot:
            return L[0]
    return None


def _sock(name, typ, link=None, *, widget=None, shape=None, localized=None):
    d = {"name": name, "type": typ, "link": link}
    if localized:
        d["localized_name"] = localized
    if widget:
        d["widget"] = {"name": widget}
    if shape is not None:
        d["shape"] = shape
    return d


def _out(name, typ, links=None, *, localized=None, slot_index=None):
    d = {"name": name, "type": typ, "links": links}
    if localized:
        d["localized_name"] = localized
    if slot_index is not None:
        d["slot_index"] = slot_index
    return d


def write_flux_klein() -> Path:
    """Flux.2 Klein + Prep + sequential CP_KleinPoseBatch."""
    # Node ids: 1 UNET  2 CLIP  3 VAE  4 LoadImage  5 Prep
    # 6 Preview source  7 KleinBatch  8 Preview results  9 Save
    # 25 Preview guides  26 DWPose  27 Preview+guides  28 Note

    links = [
        link(1, 1, 0, 7, 0, "MODEL"),
        link(2, 2, 0, 7, 1, "CLIP"),
        link(3, 3, 0, 7, 2, "VAE"),
        link(6, 4, 0, 5, 0, "IMAGE"),
        link(33, 4, 0, 26, 0, "IMAGE"),
        # Prep optional sockets are indices 1/2 in the UI export (not 7/8).
        link(34, 26, 1, 5, 1, "POSE_KEYPOINT"),
        link(35, 26, 0, 5, 2, "IMAGE"),
        link(7, 5, 9, 7, 3, "IMAGE"),  # image_batch
        link(8, 5, 0, 7, 4, "IMAGE"),  # guide
        link(9, 5, 7, 7, 5, "STRING"),  # edit_prompts
        link(10, 5, 3, 6, 0, "IMAGE"),  # source_pose → preview
        link(32, 5, 4, 25, 0, "IMAGE"),  # preview_pose → preview
        link(11, 7, 0, 9, 0, "IMAGE"),  # results → Save
        link(12, 7, 0, 8, 0, "IMAGE"),  # results → Preview
        link(36, 7, 1, 27, 0, "IMAGE"),  # images_with_guides → Preview
    ]

    def o(nid):
        return _outs(links, nid)

    def i(nid, slot):
        return _inp(links, nid, slot)

    nodes = [
        {
            "id": 1,
            "type": "UNETLoader",
            "pos": [332.34463263067454, -121.82311683911603],
            "size": [360, 82],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [
                _sock("unet_name", "COMBO", widget="unet_name", localized="nom_unet"),
                _sock(
                    "weight_dtype", "COMBO", widget="weight_dtype", localized="dtype_poids"
                ),
            ],
            "outputs": [
                _out("MODEL", "MODEL", o(1).get(0), localized="MODÈLE", slot_index=0)
            ],
            "properties": {
                "Node name for S&R": "UNETLoader",
                "models": [MODELS[0]],
            },
            "widgets_values": ["flux-2-klein-4b-fp8.safetensors", "default"],
            "widgets_values_named": {
                "unet_name": "flux-2-klein-4b-fp8.safetensors",
                "weight_dtype": "default",
            },
        },
        {
            "id": 2,
            "type": "CLIPLoader",
            "pos": [335.2975903757215, 2.60630513433333],
            "size": [360, 106],
            "flags": {},
            "order": 1,
            "mode": 0,
            "inputs": [
                _sock("clip_name", "COMBO", widget="clip_name", localized="clip_name"),
                _sock("type", "COMBO", widget="type", localized="type"),
                _sock("device", "COMBO", widget="device", shape=7, localized="appareil"),
            ],
            "outputs": [
                _out("CLIP", "CLIP", o(2).get(0), localized="CLIP", slot_index=0)
            ],
            "properties": {
                "Node name for S&R": "CLIPLoader",
                "models": [MODELS[1]],
            },
            "widgets_values": ["qwen_3_4b.safetensors", "flux2", "default"],
            "widgets_values_named": {
                "clip_name": "qwen_3_4b.safetensors",
                "type": "flux2",
                "device": "default",
            },
        },
        {
            "id": 3,
            "type": "VAELoader",
            "pos": [335.2975903757214, 151.46535409892869],
            "size": [360, 58],
            "flags": {},
            "order": 2,
            "mode": 0,
            "inputs": [
                _sock("vae_name", "COMBO", widget="vae_name", localized="nom_vae"),
            ],
            "outputs": [_out("VAE", "VAE", o(3).get(0), localized="VAE", slot_index=0)],
            "properties": {
                "Node name for S&R": "VAELoader",
                "models": [MODELS[2]],
            },
            "widgets_values": ["flux2-vae.safetensors"],
            "widgets_values_named": {"vae_name": "flux2-vae.safetensors"},
        },
        {
            "id": 4,
            "type": "LoadImage",
            "pos": [-472.341156166366, 370.40501384634035],
            "size": [320, 314.0],
            "flags": {},
            "order": 3,
            "mode": 0,
            "inputs": [
                _sock("image", "COMBO", widget="image", localized="image"),
                _sock(
                    "upload",
                    "IMAGEUPLOAD",
                    widget="upload",
                    localized="choisissez le fichier à télécharger",
                ),
            ],
            "outputs": [
                _out("IMAGE", "IMAGE", o(4).get(0), localized="IMAGE", slot_index=0),
                _out("MASK", "MASK", None, localized="MASQUE", slot_index=1),
            ],
            "title": "Sprite de reference",
            "properties": {"Node name for S&R": "LoadImage"},
            "widgets_values": ["Example_Comfy.png", "image"],
            "widgets_values_named": {"image": "Example_Comfy.png", "upload": "image"},
        },
        {
            "id": 26,
            "type": "DWPreprocessor",
            "pos": [-68.04661426120819, 463.0844931461205],
            "size": [315, 222],
            "flags": {},
            "order": 5,
            "mode": 0,
            "inputs": [
                _sock("image", "IMAGE", i(26, 0), localized="image"),
                _sock(
                    "detect_hand", "COMBO", widget="detect_hand", shape=7, localized="detect_hand"
                ),
                _sock(
                    "detect_body", "COMBO", widget="detect_body", shape=7, localized="detect_body"
                ),
                _sock(
                    "detect_face", "COMBO", widget="detect_face", shape=7, localized="detect_face"
                ),
                _sock(
                    "resolution", "INT", widget="resolution", shape=7, localized="resolution"
                ),
                _sock(
                    "bbox_detector",
                    "COMBO",
                    widget="bbox_detector",
                    shape=7,
                    localized="bbox_detector",
                ),
                _sock(
                    "pose_estimator",
                    "COMBO",
                    widget="pose_estimator",
                    shape=7,
                    localized="pose_estimator",
                ),
                _sock(
                    "scale_stick_for_xinsr_cn",
                    "COMBO",
                    widget="scale_stick_for_xinsr_cn",
                    shape=7,
                    localized="scale_stick_for_xinsr_cn",
                ),
            ],
            "outputs": [
                _out("IMAGE", "IMAGE", o(26).get(0), localized="IMAGE", slot_index=0),
                _out(
                    "POSE_KEYPOINT",
                    "POSE_KEYPOINT",
                    o(26).get(1),
                    localized="POINT CLÉ DE POSE",
                    slot_index=1,
                ),
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
            "widgets_values_named": {
                "detect_hand": "enable",
                "detect_body": "enable",
                "detect_face": "enable",
                "resolution": 1024,
                "bbox_detector": "yolox_l.onnx",
                "pose_estimator": "dw-ll_ucoco_384.onnx",
                "scale_stick_for_xinsr_cn": "disable",
            },
        },
        {
            "id": 5,
            "type": "CP_PoseTransferPrep",
            "pos": [306.31039460963376, 368.59256027462476],
            "size": [360, 576],
            "flags": {},
            "order": 6,
            "mode": 0,
            "inputs": [
                _sock("image", "IMAGE", i(5, 0), localized="image"),
                _sock(
                    "pose_keypoint",
                    "POSE_KEYPOINT",
                    i(5, 1),
                    shape=7,
                    localized="pose_keypoint",
                ),
                _sock(
                    "dwpose_image", "IMAGE", i(5, 2), shape=7, localized="dwpose_image"
                ),
                _sock("source_pose", "POSE", None, shape=7, localized="source_pose"),
                _sock("animation", "COMBO", widget="animation", localized="animation"),
                _sock(
                    "orientation", "COMBO", widget="orientation", localized="orientation"
                ),
                _sock("frame", "COMBO", widget="frame", localized="frame"),
                _sock(
                    "align_to_source",
                    "BOOLEAN",
                    widget="align_to_source",
                    localized="align_to_source",
                ),
                _sock(
                    "auto_caption",
                    "BOOLEAN",
                    widget="auto_caption",
                    localized="auto_caption",
                ),
                _sock("caption", "STRING", widget="caption", shape=7, localized="caption"),
                _sock("extra", "STRING", widget="extra", shape=7, localized="extra"),
                _sock(
                    "pose_list", "STRING", widget="pose_list", shape=7, localized="pose_list"
                ),
                _sock("width", "INT", widget="width", shape=7, localized="width"),
                _sock("height", "INT", widget="height", shape=7, localized="height"),
            ],
            "outputs": [
                _out("guide", "IMAGE", o(5).get(0), localized="guide", slot_index=0),
                _out(
                    "edit_prompt", "STRING", None, localized="edit_prompt", slot_index=1
                ),
                _out("pose", "POSE", None, localized="pose", slot_index=2),
                _out(
                    "source_pose", "IMAGE", o(5).get(3), localized="source_pose", slot_index=3
                ),
                _out(
                    "preview_pose",
                    "IMAGE",
                    o(5).get(4),
                    localized="preview_pose",
                    slot_index=4,
                ),
                _out("caption", "STRING", None, localized="caption", slot_index=5),
                _out(
                    "caption_backend",
                    "STRING",
                    None,
                    localized="caption_backend",
                    slot_index=6,
                ),
                _out(
                    "edit_prompts",
                    "STRING",
                    o(5).get(7),
                    localized="edit_prompts",
                    slot_index=7,
                ),
                _out("pose_names", "STRING", None, localized="pose_names", slot_index=8),
                _out(
                    "image_batch",
                    "IMAGE",
                    o(5).get(9),
                    localized="image_batch",
                    slot_index=9,
                ),
                _out("count", "INT", None, localized="count", slot_index=10),
            ],
            "title": "Prep: filters -> guide batch",
            "properties": {"Node name for S&R": "CP_PoseTransferPrep"},
            "widgets_values": [
                "slow_run",
                "LEFT",
                "ALL",
                False,
                True,
                "",
                "",
                "",
                0,
                0,
            ],
            "widgets_values_named": {
                "animation": "slow_run",
                "orientation": "LEFT",
                "frame": "ALL",
                "align_to_source": False,
                "auto_caption": True,
                "caption": "",
                "extra": "",
                "pose_list": "",
                "width": 0,
                "height": 0,
            },
        },
        {
            "id": 6,
            "type": "PreviewImage",
            "pos": [861.6711943714672, 948.9017757350305],
            "size": [334.5496485720138, 274.42710440418705],
            "flags": {},
            "order": 7,
            "mode": 0,
            "inputs": [_sock("images", "IMAGE", i(6, 0), localized="images")],
            "outputs": [_out("images", "IMAGE", None, localized="images")],
            "title": "source_pose (base + squelette detecte)",
            "properties": {"Node name for S&R": "PreviewImage"},
        },
        {
            "id": 25,
            "type": "PreviewImage",
            "pos": [862.0769890501385, 642.7925749707717],
            "size": [334.1119140625, 260],
            "flags": {},
            "order": 8,
            "mode": 0,
            "inputs": [_sock("images", "IMAGE", i(25, 0), localized="images")],
            "outputs": [_out("images", "IMAGE", None, localized="images")],
            "title": "guides (squelettes seuls, live pose en cours)",
            "properties": {"Node name for S&R": "PreviewImage"},
        },
        {
            "id": 7,
            "type": "CP_KleinPoseBatch",
            "pos": [840, 80],
            "size": [360, 514],
            "flags": {},
            "order": 9,
            "mode": 0,
            "inputs": [
                _sock("model", "MODEL", i(7, 0), localized="model"),
                _sock("clip", "CLIP", i(7, 1), localized="clip"),
                _sock("vae", "VAE", i(7, 2), localized="vae"),
                _sock("image_batch", "IMAGE", i(7, 3), localized="image_batch"),
                _sock("guide", "IMAGE", i(7, 4), localized="guide"),
                # forceInput: socket only — no widget (avoids widgets_values shift)
                _sock("edit_prompts", "STRING", i(7, 5), localized="edit_prompts"),
                _sock("seed", "INT", widget="seed", localized="seed"),
                _sock("steps", "INT", widget="steps", localized="steps"),
                _sock("cfg", "FLOAT", widget="cfg", localized="cfg"),
                _sock(
                    "sampler_name", "COMBO", widget="sampler_name", localized="sampler_name"
                ),
                _sock("megapixels", "FLOAT", widget="megapixels", localized="megapixels"),
                _sock(
                    "pose_weight", "INT", widget="pose_weight", localized="pose_weight"
                ),
                _sock(
                    "filename_prefix",
                    "STRING",
                    widget="filename_prefix",
                    localized="filename_prefix",
                ),
            ],
            "outputs": [
                _out("images", "IMAGE", o(7).get(0), localized="images", slot_index=0),
                _out(
                    "images_with_guides",
                    "IMAGE",
                    o(7).get(1),
                    localized="images_with_guides",
                    slot_index=1,
                ),
                _out("count", "INT", None, localized="count", slot_index=2),
            ],
            "title": "Klein batch (resultats progressifs)",
            "properties": {"Node name for S&R": "CP_KleinPoseBatch"},
            "widgets_values": [42, 4, 1, "euler", 1, 2, "pose_transfer_flux_klein"],
            "widgets_values_named": {
                "seed": 42,
                "steps": 4,
                "cfg": 1,
                "sampler_name": "euler",
                "megapixels": 1,
                "pose_weight": 2,
                "filename_prefix": "pose_transfer_flux_klein",
            },
        },
        {
            "id": 8,
            "type": "PreviewImage",
            "pos": [1252.2842697008923, 162.41665639093395],
            "size": [360, 360],
            "flags": {},
            "order": 11,
            "mode": 0,
            "inputs": [_sock("images", "IMAGE", i(8, 0), localized="images")],
            "outputs": [_out("images", "IMAGE", None, localized="images")],
            "title": "Results (images seules)",
            "properties": {"Node name for S&R": "PreviewImage"},
        },
        {
            "id": 27,
            "type": "PreviewImage",
            "pos": [1254.7970793750599, 568.231079479891],
            "size": [360, 360],
            "flags": {},
            "order": 12,
            "mode": 0,
            "inputs": [_sock("images", "IMAGE", i(27, 0), localized="images")],
            "outputs": [_out("images", "IMAGE", None, localized="images")],
            "title": "Results + guides (QC pose)",
            "properties": {"Node name for S&R": "PreviewImage"},
        },
        {
            "id": 9,
            "type": "SaveImage",
            "pos": [1259.1964362185904, 106.83948958960399],
            "size": [360, 300],
            "flags": {"collapsed": True},
            "order": 10,
            "mode": 0,
            "inputs": [
                _sock("images", "IMAGE", i(9, 0), localized="images"),
                _sock(
                    "filename_prefix",
                    "STRING",
                    widget="filename_prefix",
                    localized="préfixe_du_nom_de_fichier",
                ),
            ],
            "outputs": [_out("images", "IMAGE", None, localized="images")],
            "properties": {"Node name for S&R": "SaveImage"},
            "widgets_values": ["pose_transfer_flux_klein_all"],
            "widgets_values_named": {
                "filename_prefix": "pose_transfer_flux_klein_all"
            },
        },
        {
            "id": 28,
            "type": "Note",
            "pos": [857.6469124072922, -132.31233208966685],
            "size": [323.7233686896046, 113.08174991360698],
            "flags": {},
            "order": 4,
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "properties": {},
            "widgets_values": [
                "[Steps]\n"
                "The number of steps computed for each picture\n"
                "Recommended value: 4 to 10\n\n"
                "[pose_weight]\n"
                "The weight balance between pose and original picture\n"
                "Recommended value: 2 to 4"
            ],
            "widgets_values_named": {
                "text": (
                    "[Steps]\n"
                    "The number of steps computed for each picture\n"
                    "Recommended value: 4 to 10\n\n"
                    "[pose_weight]\n"
                    "The weight balance between pose and original picture\n"
                    "Recommended value: 2 to 4"
                )
            },
            "color": "#432",
            "bgcolor": "#653",
        },
    ]

    data = {
        "id": "9c475d16-7ba3-4d4e-94aa-8be4b5f6fdd5",
        "revision": 0,
        "last_node_id": 31,
        "last_link_id": 36,
        "nodes": nodes,
        "links": links,
        "groups": [
            {
                "id": 1,
                "title": "Modeles Flux.2 Klein",
                "bounding": [
                    312.3446326306744,
                    -201.688236770825,
                    397.0470422549531,
                    424.2947762110972,
                ],
                "color": "#3f789e",
                "flags": {},
            },
            {
                "id": 2,
                "title": "CharacterPose Prep",
                "bounding": [
                    -500.6576477171255,
                    278.7931641110888,
                    1205.4915634542356,
                    673.4233217180927,
                ],
                "color": "#88A",
                "flags": {},
            },
            {
                "id": 3,
                "title": "Klein sequential batch",
                "bounding": [820, 0, 818.9272868483667, 1239.3749257655236],
                "color": "#3f789e",
                "flags": {},
            },
        ],
        "config": {},
        "extra": {
            "info": {
                "name": "CharacterPose — Flux.2 Klein Pose Transfer (Prep)",
                "version": "0.6.0",
                "description": (
                    "Prep filters → CP_KleinPoseBatch "
                    "(per-prompt sequential + progressive preview)."
                ),
            },
            "ds": {
                "scale": 0.8390545288824222,
                "offset": [933.0532459001683, 500.7379337698269],
            },
        },
        "extensions": {"models": MODELS},
        "version": 0.4,
    }

    TARGET.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {TARGET}")
    return TARGET


def write_mixamo() -> Path:
    """Mixamo FBX export + PoseLibraryLoad / ApplyPose preview."""
    # 2 PoseLibraryLoad  3 MixamoExport  6 ApplyPose
    # 7 Preview (library stick)  8 Preview (export contact sheet)

    links = [
        link(2, 2, 0, 6, 0, "POSE"),
        link(3, 6, 0, 7, 0, "IMAGE"),
        link(4, 3, 2, 8, 0, "IMAGE"),  # Mixamo preview → Preview
    ]

    def o(nid):
        return _outs(links, nid)

    def i(nid, slot):
        return _inp(links, nid, slot)

    nodes = [
        {
            "id": 2,
            "type": "CP_PoseLibraryLoad",
            "pos": [356.40303389938384, 349.50620953911255],
            "size": [270, 106],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [
                _sock("pose_file", "COMBO", widget="pose_file", localized="pose_file"),
                _sock("width", "INT", widget="width", shape=7, localized="width"),
                _sock("height", "INT", widget="height", shape=7, localized="height"),
            ],
            "outputs": [
                _out("pose", "POSE", o(2).get(0), localized="pose"),
            ],
            "properties": {"Node name for S&R": "CP_PoseLibraryLoad"},
            "widgets_values": ["head_hit_NO_03.pose", 1024, 1024],
            "widgets_values_named": {
                "pose_file": "head_hit_NO_03.pose",
                "width": 1024,
                "height": 1024,
            },
        },
        {
            "id": 3,
            "type": "CP_MixamoPoseExport",
            "pos": [354.0735430046062, 522.6579787074959],
            "size": [270, 218],
            "flags": {},
            "order": 1,
            "mode": 0,
            "inputs": [
                _sock("fbx", "COMBO", widget="fbx", localized="fbx"),
                _sock(
                    "orientations", "COMBO", widget="orientations", localized="orientations"
                ),
                _sock(
                    "frames_per_view",
                    "INT",
                    widget="frames_per_view",
                    localized="frames_per_view",
                ),
                _sock("out_dir", "STRING", widget="out_dir", shape=7, localized="out_dir"),
                _sock(
                    "write_png", "BOOLEAN", widget="write_png", shape=7, localized="write_png"
                ),
                _sock("size", "INT", widget="size", shape=7, localized="size"),
            ],
            "outputs": [
                _out("pose_list", "STRING", None, localized="pose_list"),
                _out("count", "INT", None, localized="count"),
                _out("preview", "IMAGE", o(3).get(2), localized="preview"),
            ],
            "properties": {"Node name for S&R": "CP_MixamoPoseExport"},
            # Portable name — resolved via package mixamo/ (see _resolve_fbx)
            "widgets_values": ["Head Hit.fbx", "8", 9, "poses", False, 1024],
            "widgets_values_named": {
                "fbx": "Head Hit.fbx",
                "orientations": "8",
                "frames_per_view": 9,
                "out_dir": "poses",
                "write_png": False,
                "size": 1024,
            },
        },
        {
            "id": 6,
            "type": "CP_ApplyPose",
            "pos": [811.6584509957235, 342.90130796699907],
            "size": [270, 126],
            "flags": {},
            "order": 2,
            "mode": 0,
            "inputs": [
                _sock("pose", "POSE", i(6, 0), localized="pose"),
                _sock("width", "INT", widget="width", shape=7, localized="width"),
                _sock("height", "INT", widget="height", shape=7, localized="height"),
                _sock(
                    "stick_width",
                    "INT",
                    widget="stick_width",
                    shape=7,
                    localized="stick_width",
                ),
            ],
            "outputs": [
                _out("skeleton", "IMAGE", o(6).get(0), localized="skeleton"),
                _out("pose", "POSE", None, localized="pose"),
            ],
            "properties": {"Node name for S&R": "CP_ApplyPose"},
            "widgets_values": [0, 0, 0],
            "widgets_values_named": {"width": 0, "height": 0, "stick_width": 0},
        },
        {
            "id": 7,
            "type": "PreviewImage",
            "pos": [1264.0413887997713, 340.7997687119954],
            "size": [210, 258],
            "flags": {},
            "order": 4,
            "mode": 0,
            "inputs": [_sock("images", "IMAGE", i(7, 0), localized="images")],
            "outputs": [_out("images", "IMAGE", None, localized="images")],
            "properties": {"Node name for S&R": "PreviewImage"},
        },
        {
            "id": 8,
            "type": "PreviewImage",
            "pos": [752.8391105415135, 560.649312671556],
            "size": [410.8676485847028, 487.6973322115944],
            "flags": {},
            "order": 3,
            "mode": 0,
            "inputs": [_sock("images", "IMAGE", i(8, 0), localized="images")],
            "outputs": [_out("images", "IMAGE", None, localized="images")],
            "properties": {"Node name for S&R": "PreviewImage"},
        },
    ]

    data = {
        "id": "b030ae23-f478-425d-8552-70b5976cf048",
        "revision": 0,
        "last_node_id": 9,
        "last_link_id": 5,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {
            "ds": {
                "scale": 0.9076742523543142,
                "offset": [-117.95075553589477, -231.00483260392738],
            }
        },
        "version": 0.4,
    }

    TARGET_MIXAMO.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {TARGET_MIXAMO}")
    return TARGET_MIXAMO


def validate(path: Path) -> None:
    """Sanity-check every link id referenced by nodes exists in links[]."""
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in data["nodes"]}
    links = {L[0]: L for L in data["links"]}
    errors: list[str] = []
    for nid, n in nodes.items():
        for inp in n.get("inputs") or []:
            lid = inp.get("link")
            if lid is None:
                continue
            if lid not in links:
                errors.append(f"node {nid}/{inp.get('name')}: missing link {lid}")
                continue
            L = links[lid]
            if L[3] != nid:
                errors.append(
                    f"link {lid}: expected dst node {nid} got {L[3]} for {inp.get('name')}"
                )
        for oi, out in enumerate(n.get("outputs") or []):
            for lid in out.get("links") or []:
                if lid not in links:
                    errors.append(
                        f"node {nid}/{out.get('name')}: missing out link {lid}"
                    )
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
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    write_flux_klein()
    validate(TARGET)
    write_mixamo()
    validate(TARGET_MIXAMO)
