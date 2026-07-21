"""Character encode / save / load nodes."""

from __future__ import annotations

from pathlib import Path

try:
    import folder_paths
except ImportError:
    folder_paths = None

import numpy as np
import torch

from ..formats.char_io import load_character, make_character, save_character
from ..utils import extract_palette, tensor_to_np


def _character_output_dir() -> Path:
    if folder_paths is not None:
        try:
            d = Path(folder_paths.get_output_directory()) / "characters"
            d.mkdir(parents=True, exist_ok=True)
            return d
        except Exception:
            pass
    d = Path(__file__).resolve().parent.parent / "characters"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _character_input_dirs() -> list[Path]:
    dirs = [_character_output_dir(), Path(__file__).resolve().parent.parent / "characters"]
    if folder_paths is not None:
        try:
            dirs.insert(0, Path(folder_paths.get_input_directory()) / "characters")
        except Exception:
            pass
    return dirs


def _list_characters() -> list[str]:
    names: list[str] = []
    for d in _character_input_dirs():
        if not d.exists():
            continue
        for p in sorted(d.glob("*.char")):
            if p.is_dir() and p.name not in names:
                names.append(p.name)
    return names or ["(none)"]


def _resolve_character(name: str) -> Path:
    for d in _character_input_dirs():
        cand = d / name
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Character not found: {name}")


def _encode_clip_vision(clip_vision, image_tensor: torch.Tensor) -> np.ndarray | None:
    """Encode with ComfyUI CLIP_VISION if provided. Returns flat float32 vector."""
    if clip_vision is None:
        return None
    try:
        # ComfyUI CLIPVision encode API
        out = clip_vision.encode_image(image_tensor)
        emb = out
        if hasattr(out, "image_embeds"):
            emb = out.image_embeds
        elif isinstance(out, dict) and "image_embeds" in out:
            emb = out["image_embeds"]
        elif isinstance(out, (tuple, list)):
            emb = out[0]
        if isinstance(emb, torch.Tensor):
            return emb.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
    except Exception:
        try:
            # Some versions: clip_vision.encode(image)
            out = clip_vision.encode(image_tensor)
            if isinstance(out, torch.Tensor):
                return out.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
        except Exception:
            return None
    return None


class CharacterEncode:
    """Encode an image into a CHARACTER identity object."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "name": ("STRING", {"default": "hero"}),
                "palette_colors": ("INT", {"default": 8, "min": 2, "max": 32}),
            },
            "optional": {
                "mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
                "caption": ("STRING", {"default": "", "multiline": True}),
                "prompt_template": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("CHARACTER",)
    RETURN_NAMES = ("character",)
    FUNCTION = "encode"
    CATEGORY = "CharacterPose/Character"

    def encode(
        self,
        image,
        name="hero",
        palette_colors=8,
        mask=None,
        clip_vision=None,
        caption="",
        prompt_template="",
    ):
        rgb = tensor_to_np(image)
        if mask is not None:
            m = mask[0].detach().cpu().numpy()
            if m.ndim == 3:
                m = m[:, :, 0]
            m = (m > 0.5).astype(np.uint8)
            # Composite on neutral gray background for cleaner identity
            bg = np.full_like(rgb, 127)
            m3 = m[:, :, None]
            rgb = np.where(m3 > 0, rgb, bg)

        ref = (rgb.astype(np.float32) / 255.0).clip(0, 1)
        palette = extract_palette(rgb, n_colors=int(palette_colors))
        embedding = _encode_clip_vision(clip_vision, image)

        meta = {"palette_colors": int(palette_colors)}
        if (caption or "").strip():
            meta["caption"] = caption.strip()
        if (prompt_template or "").strip():
            meta["prompt_template"] = prompt_template.strip()

        character = make_character(
            name=name or "hero",
            reference_image=ref,
            palette=palette,
            embedding=embedding,
            metadata=meta,
        )
        return (character,)


class SaveCharacter:
    """Persist a CHARACTER to output/characters/<name>.char"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character": ("CHARACTER",),
                "filename": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    CATEGORY = "CharacterPose/Character"
    OUTPUT_NODE = True

    def save(self, character, filename=""):
        name = (filename or character.get("name") or "character").strip()
        if name.endswith(".char"):
            name = name[:-5]
        path = _character_output_dir() / f"{name}.char"
        save_character(character, path)
        return (str(path),)


class LoadCharacter:
    """Load a CHARACTER from disk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_file": (_list_characters(),),
            },
        }

    RETURN_TYPES = ("CHARACTER",)
    RETURN_NAMES = ("character",)
    FUNCTION = "load"
    CATEGORY = "CharacterPose/Character"

    def load(self, character_file):
        if character_file == "(none)":
            raise ValueError("No .char files found. Encode and Save a character first.")
        path = _resolve_character(character_file)
        return (load_character(path),)


class CharacterToImage:
    """Expose the stored reference image from a CHARACTER (for IP-Adapter wiring)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"character": ("CHARACTER",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = "CharacterPose/Character"

    def convert(self, character):
        ref = character["reference_image"]
        t = torch.from_numpy(ref.astype(np.float32))[None, ...]
        return (t,)


NODE_CLASS_MAPPINGS = {
    "CP_CharacterEncode": CharacterEncode,
    "CP_SaveCharacter": SaveCharacter,
    "CP_LoadCharacter": LoadCharacter,
    "CP_CharacterToImage": CharacterToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CP_CharacterEncode": "Character Encode",
    "CP_SaveCharacter": "Save Character",
    "CP_LoadCharacter": "Load Character",
    "CP_CharacterToImage": "Character To Image",
}
