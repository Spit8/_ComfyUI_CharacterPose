"""Image-to-text caption + edit-prompt builders for pose transfer."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..utils import tensor_to_np

ANTI_BONES = (
    "CRITICAL: the pose guide image is an abstract stick-figure only — "
    "NEVER draw bones, skeleton, x-ray, anatomical limbs, white bone overlays, "
    "or OpenPose sticks on the character. Solid clothed opaque skin and clothing only."
)

POSE_FOLLOW = (
    "Change ONLY the body pose to match the stick-figure pose guide. "
    "Keep the exact same character identity, outfit, colors, and art style."
)

DEFAULT_FALLBACK_CAPTION = (
    "full-body 2D game character sprite, cartoon line art, clean colors, plain background"
)


def build_edit_prompt(
    caption: str,
    *,
    prop_hint: str = "",
    extra: str = "",
    include_anti_bones: bool = True,
) -> str:
    """Assemble a ready-to-use edit prompt from caption + optional prop hints."""
    parts: list[str] = []
    cap = (caption or "").strip() or DEFAULT_FALLBACK_CAPTION
    parts.append(
        f"Full-body 2D game character sprite. Redraw the exact same character: {cap}."
    )
    parts.append(POSE_FOLLOW)
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


def _heuristic_caption(rgb: np.ndarray) -> str:
    """Cheap fallback when Florence-2 is unavailable: palette + size cues."""
    h, w, _ = rgb.shape
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    mean = pixels.mean(axis=0)
    # Dominant-ish via subsampled mode bins
    quantized = (pixels // 32).astype(np.int32)
    # Pack RGB bins
    keys = quantized[:, 0] * 10000 + quantized[:, 1] * 100 + quantized[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    top = vals[int(np.argmax(counts))]
    r = (top // 10000) * 32 + 16
    g = ((top // 100) % 100) * 32 + 16
    b = (top % 100) * 32 + 16

    def _color_name(rr: float, gg: float, bb: float) -> str:
        if max(rr, gg, bb) < 50:
            return "dark"
        if min(rr, gg, bb) > 200:
            return "light / pale"
        if rr > gg + 40 and rr > bb + 40:
            return "red / warm"
        if gg > rr + 30 and gg > bb + 30:
            return "green"
        if bb > rr + 30 and bb > gg + 30:
            return "blue"
        if rr > 150 and gg > 100 and bb < 80:
            return "gold / yellow"
        if rr > 100 and gg > 70 and bb > 40 and abs(rr - gg) < 40:
            return "brown / tan"
        return "multicolor"

    aspect = h / max(1, w)
    body = "tall full-body" if aspect > 1.2 else "full-body" if aspect > 0.85 else "wide / bust"
    style = "pixel-art style" if min(h, w) < 256 else "cartoon / stylized illustration"
    return (
        f"{body} character, {style}, dominant colors {_color_name(r, g, b)}, "
        f"average tone RGB({int(mean[0])},{int(mean[1])},{int(mean[2])}), "
        f"game sprite look"
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
    max_tokens: int = 64,
    style_bias: str = "game sprite",
) -> tuple[str, str]:
    """Return (caption, backend_note). Falls back to heuristic on any failure."""
    try:
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoProcessor
    except Exception as e:
        return _heuristic_caption(rgb), f"fallback_heuristic (transformers unavailable: {e})"

    path = _resolve_florence_path(model_id)
    try:
        if _florence_cache["model"] is None or _florence_cache["id"] != path:
            processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()
            _florence_cache["processor"] = processor
            _florence_cache["model"] = model
            _florence_cache["id"] = path

        processor = _florence_cache["processor"]
        model = _florence_cache["model"]
        pil = Image.fromarray(rgb)

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
            caption = str(parsed.get(task) or parsed.get("<DETAILED_CAPTION>") or next(iter(parsed.values()), ""))
        else:
            caption = str(parsed)
        caption = caption.strip()
        if style_bias and style_bias.strip().lower() not in caption.lower():
            caption = f"{caption.rstrip('.')}, {style_bias.strip()}"
        if not caption:
            return _heuristic_caption(rgb), "fallback_heuristic (empty florence output)"
        return caption, f"florence2:{path}"
    except Exception as e:
        return _heuristic_caption(rgb), f"fallback_heuristic (florence error: {e})"


class CharacterCaption:
    """Describe a character sprite (Florence-2 with heuristic fallback)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_id": (
                    "STRING",
                    {"default": "microsoft/Florence-2-base"},
                ),
                "style_bias": ("STRING", {"default": "2D game sprite, cartoon"}),
                "max_tokens": ("INT", {"default": 64, "min": 16, "max": 256}),
            },
            "optional": {
                "character": ("CHARACTER",),
                "use_cached_caption": ("BOOLEAN", {"default": True}),
                "prop_hint": ("STRING", {"default": "", "multiline": True}),
                "extra": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "CHARACTER")
    RETURN_NAMES = ("caption", "edit_prompt", "character")
    FUNCTION = "caption"
    CATEGORY = "CharacterPose/Character"

    def caption(
        self,
        image,
        model_id="microsoft/Florence-2-base",
        style_bias="2D game sprite, cartoon",
        max_tokens=64,
        character=None,
        use_cached_caption=True,
        prop_hint="",
        extra="",
    ):
        cached = ""
        if character is not None and use_cached_caption:
            meta = character.get("metadata") or {}
            cached = str(meta.get("caption") or "").strip()

        if cached:
            caption = cached
            backend = "cached"
        else:
            rgb = tensor_to_np(image)
            caption, backend = caption_with_florence(
                rgb,
                model_id=model_id,
                max_tokens=int(max_tokens),
                style_bias=style_bias,
            )

        edit_prompt = build_edit_prompt(caption, prop_hint=prop_hint, extra=extra)

        out_char = character
        if out_char is None:
            from ..formats.char_io import make_character

            ref = (tensor_to_np(image).astype(np.float32) / 255.0).clip(0, 1)
            out_char = make_character(
                name="character",
                reference_image=ref,
                metadata={"caption": caption, "caption_backend": backend, "prompt_template": edit_prompt},
            )
        else:
            meta = dict(out_char.get("metadata") or {})
            meta["caption"] = caption
            meta["caption_backend"] = backend
            meta["prompt_template"] = edit_prompt
            out_char = dict(out_char)
            out_char["metadata"] = meta

        return (caption, edit_prompt, out_char)


class BuildEditPrompt:
    """Combine caption + prop hints into one edit prompt string."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "prop_hint": ("STRING", {"default": "", "multiline": True}),
                "extra": ("STRING", {"default": "", "multiline": True}),
                "include_anti_bones": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("edit_prompt",)
    FUNCTION = "build"
    CATEGORY = "CharacterPose/Character"

    def build(self, caption="", prop_hint="", extra="", include_anti_bones=True):
        return (
            build_edit_prompt(
                caption,
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
