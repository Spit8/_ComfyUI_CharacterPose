"""Sequential Flux.2 Klein sampling — one pose at a time, per-prompt, live preview."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("characterpose.klein_batch")


def parse_edit_prompts(edit_prompts: str, n: int) -> list[str]:
    """Parse Prep `edit_prompts` JSON; pad/truncate to n. Pure helper (no Comfy)."""
    text = (edit_prompts or "").strip()
    if not text:
        raise ValueError("edit_prompts is empty — wire Prep.edit_prompts into CP_KleinPoseBatch.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"edit_prompts is not valid JSON: {e}") from e
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("edit_prompts must be a JSON list of strings.")
    if len(data) < n:
        data = data + [data[-1] if data else ""] * (n - len(data))
    return data[:n]


# Set from run() so nested V3 nodes can read cls.hidden.unique_id.
_NESTED_UNIQUE_ID: str | None = None


def _get_node_class(class_type: str):
    try:
        import nodes as comfy_nodes
    except ImportError as e:
        raise RuntimeError("ComfyUI `nodes` module required for CP_KleinPoseBatch.") from e
    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(class_type)
    if cls is None:
        raise RuntimeError(
            f"Missing ComfyUI node '{class_type}'. "
            "Install/update ComfyUI core (Flux.2 Klein graph nodes)."
        )
    return cls


def _hidden_stub(unique_id: str | None = None):
    """Minimal HiddenHolder so nested V3 `execute()` does not crash on cls.hidden."""
    uid = unique_id or _NESTED_UNIQUE_ID or "cp_klein_batch"
    try:
        from comfy_api.latest._io import HiddenHolder

        return HiddenHolder.from_dict({"UNIQUE_ID": uid})
    except Exception:

        class _Stub:
            def __init__(self, unique_id):
                self.unique_id = unique_id

            def __getattr__(self, _key):
                return None

        return _Stub(uid)


def _normalize_node_result(out) -> tuple:
    """Unwrap ComfyUI V3 NodeOutput / V1 tuple / dict into a plain tuple of values."""
    if out is None:
        return ()
    args = getattr(out, "args", None)
    if isinstance(args, tuple):
        return args
    if isinstance(out, dict) and "result" in out:
        result = out["result"]
        if isinstance(result, (tuple, list)):
            return tuple(result)
        return (result,)
    if isinstance(out, tuple):
        return out
    if isinstance(out, list):
        return tuple(out)
    return (out,)


def _is_v3_node(cls) -> bool:
    """True for ComfyUI 0.37+ io.ComfyNode classes (execute + hidden)."""
    if not (hasattr(cls, "execute") and callable(cls.execute)):
        return False
    # V1 nodes expose a string FUNCTION like "encode"; V3 uses EXECUTE_NORMALIZED.
    fn = getattr(cls, "FUNCTION", None)
    if callable(fn):  # classproperty
        try:
            fn = fn.__get__(cls, cls) if hasattr(fn, "__get__") else fn
        except Exception:
            pass
    if isinstance(fn, str) and fn not in (
        "EXECUTE_NORMALIZED",
        "EXECUTE_NORMALIZED_ASYNC",
        "execute",
    ):
        return False
    return True


def _call(class_type: str, **kwargs) -> tuple:
    """Invoke a registered Comfy node; unwrap NodeOutput; stub V3 hidden context."""
    cls = _get_node_class(class_type)

    if _is_v3_node(cls):
        prev = getattr(cls, "hidden", None)
        try:
            cls.hidden = _hidden_stub()
            return _normalize_node_result(cls.execute(**kwargs))
        finally:
            cls.hidden = prev

    inst = cls()
    fn_name = getattr(inst, "FUNCTION", "process")
    if fn_name in ("EXECUTE_NORMALIZED", "EXECUTE_NORMALIZED_ASYNC"):
        # Mis-detected V3 — retry with hidden stub
        prev = getattr(cls, "hidden", None)
        try:
            cls.hidden = _hidden_stub()
            return _normalize_node_result(cls.execute(**kwargs))
        finally:
            cls.hidden = prev
    out = getattr(inst, fn_name)(**kwargs)
    return _normalize_node_result(out)


def _scale_ref(image, megapixels: float):
    """Scale IMAGE to target megapixels (same math as ImageScaleToTotalPixels)."""
    import math

    import comfy.utils

    samples = image.movedim(-1, 1)
    total = float(megapixels) * 1024 * 1024
    resolution_steps = 64
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    width = max(
        resolution_steps,
        round(samples.shape[3] * scale_by / resolution_steps) * resolution_steps,
    )
    height = max(
        resolution_steps,
        round(samples.shape[2] * scale_by / resolution_steps) * resolution_steps,
    )
    s = comfy.utils.common_upscale(
        samples, int(width), int(height), "lanczos", "disabled"
    )
    return s.movedim(1, -1)


def _image_size(image) -> tuple[int, int]:
    """width, height from IMAGE tensor [B,H,W,C] — no GetImageSize node."""
    return int(image.shape[2]), int(image.shape[1])


def _tensor_to_pil(images: torch.Tensor):
    from PIL import Image

    t = images
    if t.ndim == 4:
        t = t[0]
    arr = (t.detach().cpu().float().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _prompt_id(prompt) -> str | None:
    if isinstance(prompt, dict) and len(prompt) == 1:
        return next(iter(prompt.keys()))
    return None


# Prep RETURN_NAMES: guide=0 … source_pose=3, preview_pose=4, …
_PREVIEW_POSE_SLOT = 4


def _find_guide_preview_nodes(prompt: Any) -> list[str]:
    """PreviewImage nodes whose images input is wired to Prep.preview_pose (slot 4)."""
    if not isinstance(prompt, dict):
        return []
    prep_ids = {
        str(nid)
        for nid, node in prompt.items()
        if isinstance(node, dict) and node.get("class_type") == "CP_PoseTransferPrep"
    }
    if not prep_ids:
        return []
    targets: list[str] = []
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") != "PreviewImage":
            continue
        images = (node.get("inputs") or {}).get("images")
        if not isinstance(images, (list, tuple)) or len(images) < 2:
            continue
        src, slot = str(images[0]), images[1]
        try:
            slot_i = int(slot)
        except (TypeError, ValueError):
            continue
        if src in prep_ids and slot_i == _PREVIEW_POSE_SLOT:
            targets.append(str(nid))
    return targets


def _overlay_guide_on_image(image: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
    """Composite OpenPose guide sticks onto an IMAGE tensor (result or sprite)."""
    from ..utils import np_to_tensor, tensor_to_np
    from .composer import _overlay_dwpose_image

    base = tensor_to_np(image)
    skel = tensor_to_np(guide)
    # Match spatial size if guide differs from result megapixel scale
    if base.shape[:2] != skel.shape[:2]:
        import cv2

        skel = cv2.resize(
            skel, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    out = _overlay_dwpose_image(base, skel)
    return np_to_tensor(out)


def _push_guide_preview(
    guide: torch.Tensor,
    *,
    filename_prefix: str,
    prompt: Any,
    prompt_id: str | None,
) -> None:
    """Update Prep preview_pose PreviewImage(s) with stick-only guide — not Klein UI."""
    targets = _find_guide_preview_nodes(prompt)
    if not targets:
        return

    saved_meta: list[Any] = []
    try:
        from nodes import PreviewImage

        saved = PreviewImage().save_images(guide, filename_prefix=filename_prefix)
        saved_meta = list(saved.get("ui", {}).get("images", []))
    except Exception as e:
        logger.warning("guide PreviewImage save failed: %s", e)
        return
    if not saved_meta:
        return

    try:
        from server import PromptServer

        for nid in targets:
            payload = {
                "node": str(nid),
                "display_node": str(nid),
                "output": {"images": list(saved_meta)},
            }
            if prompt_id:
                payload["prompt_id"] = prompt_id
            PromptServer.instance.send_sync("executed", payload)
    except Exception as e:
        logger.warning("guide executed push failed: %s", e)


def _push_live_preview(
    images: torch.Tensor,
    *,
    index: int,
    total: int,
    filename_prefix: str,
    ui_images: list,
    pbar,
    unique_id: str | None,
    prompt_id: str | None,
) -> None:
    """Save to disk + push WebSocket preview/executed so UI updates mid-node."""
    pil = _tensor_to_pil(images)
    preview_tuple = ("PNG", pil, None)

    # 1) Progress bar preview on the node
    if pbar is not None:
        try:
            if hasattr(pbar, "update_absolute"):
                pbar.update_absolute(index + 1, total, preview_tuple)
            else:
                pbar.update(1)
        except Exception as e:
            logger.warning("ProgressBar preview failed (pose %s): %s", index, e)

    # 2) Binary preview over WebSocket (latent-preview style)
    try:
        from server import PromptServer, BinaryEventTypes

        server = PromptServer.instance
        server.send_sync(
            BinaryEventTypes.UNENCODED_PREVIEW_IMAGE,
            preview_tuple,
            server.client_id,
        )
    except Exception as e:
        logger.warning("WebSocket PREVIEW_IMAGE failed (pose %s): %s", index, e)

    # 3) Persist to output/ immediately
    saved_meta: list[Any] = []
    try:
        from nodes import SaveImage

        saved = SaveImage().save_images(images, filename_prefix=filename_prefix)
        saved_meta = list(saved.get("ui", {}).get("images", []))
        ui_images.extend(saved_meta)
    except Exception as e:
        logger.error("SaveImage failed (pose %s, prefix=%s): %s", index, filename_prefix, e)

    # Fallback metadata via PreviewImage temp folder if SaveImage produced nothing
    if not saved_meta:
        try:
            from nodes import PreviewImage

            saved = PreviewImage().save_images(images, filename_prefix=filename_prefix)
            saved_meta = list(saved.get("ui", {}).get("images", []))
            ui_images.extend(saved_meta)
        except Exception as e:
            logger.error("PreviewImage fallback failed (pose %s): %s", index, e)

    # 4) executed event with cumulative images → node panel fills pose-by-pose
    if unique_id and ui_images:
        try:
            from server import PromptServer

            payload = {
                "node": str(unique_id),
                "display_node": str(unique_id),
                "output": {"images": list(ui_images)},
            }
            if prompt_id:
                payload["prompt_id"] = prompt_id
            PromptServer.instance.send_sync("executed", payload)
        except Exception as e:
            logger.warning("WebSocket executed failed (pose %s): %s", index, e)


class KleinPoseBatch:
    """Run Flux.2 Klein once per Prep batch item with that item's edit prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            from comfy.samplers import KSampler

            samplers = list(KSampler.SAMPLERS)
        except Exception:
            samplers = ["euler", "euler_ancestral", "heun", "dpmpp_2m", "uni_pc"]
        if "euler" not in samplers:
            samplers = ["euler"] + samplers

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image_batch": ("IMAGE",),
                "guide": ("IMAGE",),
                "edit_prompts": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "[]",
                        "forceInput": True,
                    },
                ),
                # Disable control_after_generate so widgets_values stay aligned
                # (seed, steps, cfg, sampler, megapixels, pose_weight, prefix).
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": False,
                    },
                ),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "cfg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "sampler_name": (samplers, {"default": "euler"}),
                "megapixels": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05},
                ),
                # How many times to apply ReferenceLatent(guide): 1 soft … 4 strong pose
                "pose_weight": ("INT", {"default": 2, "min": 1, "max": 4}),
                "filename_prefix": (
                    "STRING",
                    {"default": "pose_transfer_flux_klein"},
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("images", "images_with_guides", "count")
    FUNCTION = "run"
    CATEGORY = "characterpose"
    OUTPUT_NODE = True

    def run(
        self,
        model,
        clip,
        vae,
        image_batch,
        guide,
        edit_prompts: str,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        megapixels: float,
        pose_weight: int = 2,
        filename_prefix: str = "pose_transfer_flux_klein",
        unique_id=None,
        prompt=None,
    ):
        n_img = int(image_batch.shape[0])
        n_guide = int(guide.shape[0])
        if n_img != n_guide:
            raise ValueError(
                f"image_batch ({n_img}) and guide ({n_guide}) batch sizes differ."
            )
        n = n_img
        if n < 1:
            raise ValueError("Empty image_batch.")
        global _NESTED_UNIQUE_ID

        pw = int(max(1, min(4, int(pose_weight))))
        prompts = parse_edit_prompts(edit_prompts, n)
        prompt_id = _prompt_id(prompt)
        _NESTED_UNIQUE_ID = str(unique_id) if unique_id is not None else "cp_klein_batch"

        try:
            import comfy.utils

            pbar = comfy.utils.ProgressBar(n)
        except Exception:
            pbar = None

        outs: list[torch.Tensor] = []
        outs_guided: list[torch.Tensor] = []
        ui_images: list[Any] = []
        try:
            sampler = _call("KSamplerSelect", sampler_name=sampler_name)[0]

            for i in range(n):
                sprite_i = image_batch[i : i + 1]
                guide_i = guide[i : i + 1]
                prompt_text = prompts[i]

                # Stick-only live on Prep preview_pose Preview (source_pose stays base+detected)
                _push_guide_preview(
                    guide_i,
                    filename_prefix=f"{filename_prefix}_guide_{i:03d}",
                    prompt=prompt,
                    prompt_id=prompt_id,
                )

                scaled_sprite = _scale_ref(sprite_i, megapixels)
                scaled_guide = _scale_ref(guide_i, megapixels)
                width, height = _image_size(scaled_sprite)

                positive = _call("CLIPTextEncode", clip=clip, text=prompt_text)[0]
                negative = _call("ConditioningZeroOut", conditioning=positive)[0]

                lat1 = _call("VAEEncode", pixels=scaled_sprite, vae=vae)[0]
                lat2 = _call("VAEEncode", pixels=scaled_guide, vae=vae)[0]
                # Sprite once (identity/colors); guide pose_weight times (1 soft … 4 strong)
                positive = _call("ReferenceLatent", conditioning=positive, latent=lat1)[0]
                for _ in range(pw):
                    positive = _call(
                        "ReferenceLatent", conditioning=positive, latent=lat2
                    )[0]

                empty = _call(
                    "EmptyFlux2LatentImage",
                    width=width,
                    height=height,
                    batch_size=1,
                )[0]
                sigmas = _call(
                    "Flux2Scheduler",
                    steps=int(steps),
                    width=width,
                    height=height,
                )[0]
                noise = _call("RandomNoise", noise_seed=int(seed))[0]
                guider = _call(
                    "CFGGuider",
                    model=model,
                    positive=positive,
                    negative=negative,
                    cfg=float(cfg),
                )[0]
                sampled = _call(
                    "SamplerCustomAdvanced",
                    noise=noise,
                    guider=guider,
                    sampler=sampler,
                    sigmas=sigmas,
                    latent_image=empty,
                )[0]
                image = _call("VAEDecode", samples=sampled, vae=vae)[0]

                prefix = f"{filename_prefix}_{i:03d}"
                _push_live_preview(
                    image,
                    index=i,
                    total=n,
                    filename_prefix=prefix,
                    ui_images=ui_images,
                    pbar=pbar,
                    unique_id=unique_id,
                    prompt_id=prompt_id,
                )
                outs.append(image)
                try:
                    # Overlay sticks on result at result resolution
                    guided = _overlay_guide_on_image(image, scaled_guide)
                except Exception as e:
                    logger.warning("images_with_guides overlay failed (pose %s): %s", i, e)
                    guided = image
                outs_guided.append(guided)
        finally:
            _NESTED_UNIQUE_ID = None

        stacked = torch.cat(outs, dim=0)
        stacked_guided = torch.cat(outs_guided, dim=0)
        return {
            "ui": {"images": ui_images},
            "result": (stacked, stacked_guided, n),
        }


NODE_CLASS_MAPPINGS = {
    "CP_KleinPoseBatch": KleinPoseBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CP_KleinPoseBatch": "Klein Pose Batch (sequential)",
}
