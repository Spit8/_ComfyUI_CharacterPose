"""Image-to-text caption + edit-prompt builders for pose transfer."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..utils import (
    content_bbox,
    crop_to_content,
    extract_palette,
    tensor_to_np,
)

ANTI_BONES = (
    "CRITICAL: image 2 / the pose guide is an abstract stick-figure only — "
    "NEVER draw bones, skeleton, x-ray, anatomical limbs, white bone overlays, "
    "or OpenPose sticks on the character. Solid clothed opaque skin and clothing only."
)

DEFAULT_FALLBACK_CAPTION = (
    "full-body 2D game character sprite, cartoon line art, clean flat colors, plain background"
)


def palette_hex_string(rgb: np.ndarray, n_colors: int = 8) -> str:
    """Dominant garment palette as '#RRGGBB, #…' (background whites dropped)."""
    try:
        pal = extract_palette(rgb, n_colors=int(n_colors))
    except Exception:
        return ""
    if pal is None or len(pal) == 0:
        return ""
    parts = []
    for c in pal:
        r, g, b = (int(round(float(x) * 255)) for x in c[:3])
        # Skip near-white / empty-canvas centers — not character colors
        if min(r, g, b) >= 235 and max(r, g, b) - min(r, g, b) < 25:
            continue
        parts.append(f"#{r:02X}{g:02X}{b:02X}")
    return ", ".join(parts)


def build_edit_prompt(
    caption: str,
    *,
    palette_hex: str = "",
    prop_hint: str = "",
    extra: str = "",
    include_anti_bones: bool = True,
) -> str:
    """Assemble an edit prompt that leads with appearance / color lock."""
    cap = (caption or "").strip() or DEFAULT_FALLBACK_CAPTION
    pal = (palette_hex or "").strip()

    parts: list[str] = []
    parts.append(
        "APPEARANCE LOCK — match this character exactly (face, hair, outfit, accessories, "
        f"line weight, shading style): {cap}."
    )
    if pal:
        parts.append(
            f"COLOR LOCK — dominant garment / skin / accessory colors must be exactly: {pal}. "
            "These are character colors only (ignore canvas white/empty background). "
            "Do not recolor, desaturate, wash out, or invent new dominant hues."
        )
    parts.append(
        "Keep the exact same art style and materials as the reference image "
        "(including isometric / RTS sprite look if present). "
        "Change ONLY the body pose to match the stick-figure pose guide. "
        "Identity and colors must stay identical to the description above."
    )
    hint = (prop_hint or "").strip()
    if hint:
        parts.append(hint.rstrip(".") + ".")
    extra_s = (extra or "").strip()
    if extra_s:
        parts.append(extra_s.rstrip(".") + ".")
    if include_anti_bones:
        parts.append(ANTI_BONES)
    parts.append(
        "Single character, clean art, plain white background, no extra limbs, no duplicate heads."
    )
    return " ".join(parts)


def _heuristic_caption(rgb: np.ndarray, palette_hex: str = "") -> str:
    """Palette-aware fallback when no VLM is available."""
    from ..utils import content_mask

    h, w, _ = rgb.shape
    pal = palette_hex or palette_hex_string(rgb, n_colors=6)
    mask = content_mask(rgb)
    fill = float(np.mean(mask)) if mask.size else 1.0

    def _name(hex_s: str) -> str:
        try:
            r = int(hex_s[1:3], 16)
            g = int(hex_s[3:5], 16)
            b = int(hex_s[5:7], 16)
        except Exception:
            return "mixed"
        if max(r, g, b) < 45:
            return "near-black"
        if min(r, g, b) > 210:
            return "near-white"
        if r > g + 45 and r > b + 45:
            return "red"
        if r > 160 and g > 90 and b < 90:
            return "orange/gold"
        if g > r + 35 and g > b + 35:
            return "green"
        if b > r + 35 and b > g + 35:
            return "blue"
        if r > 100 and g > 70 and b > 40 and abs(r - g) < 45:
            return "brown/tan"
        if abs(r - g) < 20 and abs(g - b) < 20:
            return "gray"
        return "mixed"

    color_bits = []
    if pal:
        for hx in pal.split(",")[:5]:
            hx = hx.strip()
            if hx.startswith("#"):
                name = _name(hx)
                if name == "near-white":
                    continue
                color_bits.append(f"{name} ({hx})")
    color_str = ", ".join(color_bits) if color_bits else "multicolor"

    box = content_bbox(rgb)
    if box is not None:
        x0, y0, x1, y1 = box
        ch, cw = max(1, y1 - y0), max(1, x1 - x0)
        aspect = ch / float(cw)
    else:
        aspect = h / max(1, w)

    body = "tall full-body" if aspect > 1.2 else "full-body" if aspect > 0.85 else "wide/bust"
    # Large empty canvas → typical Flux isometric RTS asset
    if fill < 0.35:
        style = (
            "isometric RTS game sprite, hand-painted or pre-rendered, "
            "centered on empty background, readable silhouette"
        )
    elif min(h, w) < 192:
        style = "pixel-art game sprite, limited palette, crisp pixels"
    elif min(h, w) < 512:
        style = "2D game character sprite, cartoon line art, flat or cel-shaded colors"
    else:
        style = "stylized 2D character illustration, clean outlines, game-ready sprite look"

    return (
        f"{body} character, {style}, clothing and details in {color_str}, "
        f"preserve exact garment colors and silhouette from the reference"
    )


_florence_cache: dict[str, Any] = {"model": None, "processor": None, "id": None}


def _resolve_florence_path(model_id: str) -> str:
    """Prefer a local ComfyUI models folder if present."""
    try:
        import folder_paths

        for sub in ("LLM", "florence2", "Florence2", "checkpoints"):
            try:
                base = folder_paths.get_folder_paths(sub)
            except Exception:
                base = []
            if not base and sub == "checkpoints":
                try:
                    base = [folder_paths.models_dir]
                except Exception:
                    base = []
            for root in base or []:
                from pathlib import Path

                cand = Path(root) / model_id.replace("/", "--")
                if cand.exists():
                    return str(cand)
                cand2 = Path(root) / model_id.split("/")[-1]
                if cand2.exists():
                    return str(cand2)
    except Exception:
        pass
    return model_id


def caption_with_florence(
    rgb: np.ndarray,
    *,
    model_id: str = "microsoft/Florence-2-base",
    max_tokens: int = 128,
    style_bias: str = "2D game sprite, cartoon",
    palette_hex: str = "",
) -> tuple[str, str]:
    """Return (caption, backend_note). Falls back to palette heuristic on failure."""
    pal = palette_hex or palette_hex_string(rgb)

    try:
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoProcessor
    except Exception as e:
        return _heuristic_caption(rgb, pal), f"fallback_heuristic (install transformers for Florence-2: {e})"

    path = _resolve_florence_path(model_id)
    try:
        if _florence_cache["model"] is None or _florence_cache["id"] != path:
            processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()
            _florence_cache["processor"] = processor
            _florence_cache["model"] = model
            _florence_cache["id"] = path

        processor = _florence_cache["processor"]
        model = _florence_cache["model"]
        # Crop away empty canvas so Florence focuses on the character
        crop = crop_to_content(rgb, pad=0.08)
        pil = Image.fromarray(crop)

        # Detailed caption focused on character appearance + garment colors
        task = "<MORE_DETAILED_CAPTION>"
        inputs = processor(text=task, images=pil, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in inputs.items()}

        with torch.inference_mode():
            generated = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=int(max_tokens),
                num_beams=3,
                do_sample=False,
            )
        raw = processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            raw, task=task, image_size=(pil.width, pil.height)
        )
        caption = ""
        if isinstance(parsed, dict):
            caption = str(
                parsed.get(task)
                or parsed.get("<DETAILED_CAPTION>")
                or next(iter(parsed.values()), "")
            )
        else:
            caption = str(parsed)
        caption = caption.strip()
        if not caption:
            return _heuristic_caption(rgb, pal), "fallback_heuristic (empty florence output)"

        # Enrich with style bias + palette if Florence omitted them
        bits = [caption.rstrip(".")]
        if style_bias and style_bias.strip().lower() not in caption.lower():
            bits.append(style_bias.strip())
        if pal and "palette" not in caption.lower() and "#" not in caption:
            bits.append(f"color palette {pal}")
        return ". ".join(bits), f"florence2:{path}"
    except Exception as e:
        return _heuristic_caption(rgb, pal), f"fallback_heuristic (florence error: {e})"


def describe_character(
    rgb: np.ndarray,
    *,
    model_id: str = "microsoft/Florence-2-base",
    max_tokens: int = 128,
    style_bias: str = "2D game sprite, cartoon",
    palette_colors: int = 8,
) -> tuple[str, str, str]:
    """Unified caption pipeline.

    Returns (caption, palette_hex, backend_note).
    Palette and Florence run on content-cropped pixels when possible.
    """
    crop = crop_to_content(rgb, pad=0.06)
    pal = palette_hex_string(crop, n_colors=palette_colors)
    caption, backend = caption_with_florence(
        crop,
        model_id=model_id,
        max_tokens=max_tokens,
        style_bias=style_bias,
        palette_hex=pal,
    )
    return caption, pal, backend


class CharacterCaption:
    """Describe a character sprite (Florence-2 + palette color lock)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_id": (
                    "STRING",
                    {"default": "microsoft/Florence-2-base"},
                ),
                "style_bias": ("STRING", {"default": "2D game sprite, cartoon line art"}),
                "max_tokens": ("INT", {"default": 128, "min": 32, "max": 256}),
                "palette_colors": ("INT", {"default": 8, "min": 3, "max": 16}),
            },
            "optional": {
                "character": ("CHARACTER",),
                "use_cached_caption": ("BOOLEAN", {"default": True}),
                "prop_hint": ("STRING", {"default": "", "multiline": True}),
                "extra": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "CHARACTER")
    RETURN_NAMES = ("caption", "edit_prompt", "palette_hex", "caption_backend", "character")
    FUNCTION = "caption"
    CATEGORY = "CharacterPose/Character"

    def caption(
        self,
        image,
        model_id="microsoft/Florence-2-base",
        style_bias="2D game sprite, cartoon line art",
        max_tokens=128,
        palette_colors=8,
        character=None,
        use_cached_caption=True,
        prop_hint="",
        extra="",
    ):
        rgb = tensor_to_np(image)
        cached = ""
        cached_pal = ""
        if character is not None and use_cached_caption:
            meta = character.get("metadata") or {}
            cached = str(meta.get("caption") or "").strip()
            cached_pal = str(meta.get("palette_hex") or "").strip()

        if cached:
            caption = cached
            pal = cached_pal or palette_hex_string(rgb, n_colors=int(palette_colors))
            backend = "cached"
        else:
            caption, pal, backend = describe_character(
                rgb,
                model_id=model_id,
                max_tokens=int(max_tokens),
                style_bias=style_bias,
                palette_colors=int(palette_colors),
            )

        edit_prompt = build_edit_prompt(
            caption, palette_hex=pal, prop_hint=prop_hint, extra=extra
        )

        out_char = character
        if out_char is None:
            from ..formats.char_io import make_character

            ref = (rgb.astype(np.float32) / 255.0).clip(0, 1)
            out_char = make_character(
                name="character",
                reference_image=ref,
                metadata={
                    "caption": caption,
                    "caption_backend": backend,
                    "palette_hex": pal,
                    "prompt_template": edit_prompt,
                },
            )
        else:
            meta = dict(out_char.get("metadata") or {})
            meta["caption"] = caption
            meta["caption_backend"] = backend
            meta["palette_hex"] = pal
            meta["prompt_template"] = edit_prompt
            out_char = dict(out_char)
            out_char["metadata"] = meta

        return (caption, edit_prompt, pal, backend, out_char)


class BuildEditPrompt:
    """Combine caption + palette + prop hints into one edit prompt string."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "palette_hex": ("STRING", {"default": ""}),
                "prop_hint": ("STRING", {"default": "", "multiline": True}),
                "extra": ("STRING", {"default": "", "multiline": True}),
                "include_anti_bones": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("edit_prompt",)
    FUNCTION = "build"
    CATEGORY = "CharacterPose/Character"

    def build(
        self,
        caption="",
        palette_hex="",
        prop_hint="",
        extra="",
        include_anti_bones=True,
    ):
        return (
            build_edit_prompt(
                caption,
                palette_hex=palette_hex,
                prop_hint=prop_hint,
                extra=extra,
                include_anti_bones=bool(include_anti_bones),
            ),
        )


NODE_CLASS_MAPPINGS = {
    "CP_CharacterCaption": CharacterCaption,
    "CP_BuildEditPrompt": BuildEditPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CP_CharacterCaption": "Character Caption",
    "CP_BuildEditPrompt": "Build Edit Prompt",
}
