"""CharacterRender and CharacterRepair — orchestrate SDXL + ControlNet + IP-Adapter."""

from __future__ import annotations

import numpy as np
import torch

from ..utils import np_to_tensor, tensor_to_np


def _common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0):
    """Call ComfyUI's shared ksampler."""
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import latent_preview

    latent_image = latent["samples"]
    noise = comfy.sample.prepare_noise(latent_image, seed)
    noise_mask = latent.get("noise_mask")

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=denoise,
        disable_noise=False,
        start_step=None,
        last_step=None,
        force_full_denoise=True,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    out = latent.copy()
    out["samples"] = samples
    return out


def _apply_controlnet(positive, negative, control_net, pose_image, strength, start=0.0, end=1.0):
    """Apply ControlNet using ComfyUI ControlNetApplyAdvanced logic."""
    if control_net is None or strength <= 0:
        return positive, negative

    # Prefer built-in node implementation when available
    try:
        from nodes import ControlNetApplyAdvanced

        node = ControlNetApplyAdvanced()
        return node.apply_controlnet(positive, negative, control_net, pose_image, strength, start, end)
    except Exception:
        pass

    # Fallback: manual conditioning dict update (best-effort)
    try:
        import comfy.utils

        control_hint = pose_image.movedim(-1, 1)
        cnets = {}

        def apply_to(conditioning):
            c = []
            for t in conditioning:
                d = t[1].copy()
                prev = d.get("control", None)
                key = (prev,) if prev is not None else None
                if key in cnets:
                    cn = cnets[key]
                else:
                    cn = control_net.copy().set_cond_hint(control_hint, strength, (start, end))
                    if prev is not None:
                        cn.set_previous_controlnet(prev)
                    cnets[key] = cn
                d["control"] = cn
                d["control_apply_to_uncond"] = False
                n = [t[0], d]
                c.append(n)
            return c

        return apply_to(positive), apply_to(negative)
    except Exception as e:
        raise RuntimeError(f"Failed to apply ControlNet: {e}") from e


def _is_ipadapter_node_class(cls) -> bool:
    """True if cls looks like a ComfyUI IP-Adapter apply node (not a torch custom class)."""
    import inspect

    if cls is None or not inspect.isclass(cls):
        return False
    # Avoid torch._classes proxies: they blow up on hasattr/getattr
    mod_name = getattr(cls, "__module__", "") or ""
    if mod_name.startswith("torch.") or "torch._classes" in mod_name:
        return False
    fn = getattr(cls, "FUNCTION", None)
    if fn != "apply_ipadapter":
        return False
    method = inspect.getattr_static(cls, "apply_ipadapter", None)
    return callable(method)


def _find_ipadapter_advanced():
    """Locate IPAdapterAdvanced from ComfyUI_IPAdapter_plus / comfyui_ipadapter_plus."""
    import importlib
    import sys

    # 1) Prefer ComfyUI's already-registered node map (safest)
    try:
        import nodes as comfy_nodes

        mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", None) or {}
        for key in ("IPAdapterAdvanced", "IPAdapter"):
            cls = mappings.get(key)
            if _is_ipadapter_node_class(cls):
                return cls
    except Exception:
        pass

    # 2) Direct import of known package module names
    candidates = [
        "comfyui_ipadapter_plus.IPAdapterPlus",
        "ComfyUI_IPAdapter_plus.IPAdapterPlus",
        "ipadapter.IPAdapterPlus",
    ]
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            for attr in ("IPAdapterAdvanced", "IPAdapterSimple", "IPAdapter"):
                cls = getattr(mod, attr, None)
                if _is_ipadapter_node_class(cls):
                    return cls
        except Exception:
            continue

    # 3) Scan loaded custom-node modules only (skip torch.*)
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name:
            continue
        if mod_name.startswith("torch") or "torch._classes" in mod_name:
            continue
        if "ipadapter" not in mod_name.lower():
            continue
        for attr in ("IPAdapterAdvanced", "IPAdapterSimple", "IPAdapter"):
            try:
                cls = getattr(mod, attr, None)
            except Exception:
                continue
            if _is_ipadapter_node_class(cls):
                return cls
    return None


def _apply_ipadapter(model, ipadapter, clip_vision, image, weight, start=0.0, end=1.0):
    """Apply IP-Adapter via ComfyUI_IPAdapter_plus if installed."""
    if ipadapter is None or weight <= 0:
        return model

    cls = _find_ipadapter_advanced()
    if cls is None:
        print(
            "[CharacterPose] IP-Adapter not applied — install ComfyUI_IPAdapter_plus "
            "(folder often named comfyui_ipadapter_plus) or wire IPAdapter manually. "
            "Proceeding without identity adapter."
        )
        return model

    node = cls()
    # Signature from cubiq/ComfyUI_IPAdapter_plus IPAdapterAdvanced.apply_ipadapter
    attempts = [
        dict(
            model=model,
            ipadapter=ipadapter,
            image=image,
            weight=weight,
            weight_type="linear",
            combine_embeds="concat",
            start_at=start,
            end_at=end,
            embeds_scaling="V only",
            clip_vision=clip_vision,
        ),
        dict(
            model=model,
            ipadapter=ipadapter,
            image=image,
            weight=weight,
            start_at=start,
            end_at=end,
            weight_type="linear",
            clip_vision=clip_vision,
        ),
    ]
    last_err = None
    for kwargs in attempts:
        try:
            result = node.apply_ipadapter(**kwargs)
            return result[0] if isinstance(result, (tuple, list)) else result
        except TypeError as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            break

    print(f"[CharacterPose] IP-Adapter apply failed ({last_err}); continuing without it.")
    return model


def _encode_prompt(clip, text: str):
    tokens = clip.tokenize(text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]


class CharacterRender:
    """Render a character into a new pose using SDXL + ControlNet + IP-Adapter."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character": ("CHARACTER",),
                "pose_image": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "control_net": ("CONTROL_NET",),
                "positive": ("STRING", {"multiline": True, "default": "same character, full body sprite, clean lines, consistent outfit, high quality"}),
                "negative": ("STRING", {"multiline": True, "default": "blurry, deformed hands, extra limbs, different character, watermark, text"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 6.5, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (cls._samplers(),),
                "scheduler": (cls._schedulers(),),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "controlnet_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01}),
                "ipadapter_weight": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "ipadapter": ("IPADAPTER",),
                "clip_vision": ("CLIP_VISION",),
            },
        }

    @staticmethod
    def _samplers():
        try:
            import comfy.samplers

            return comfy.samplers.KSampler.SAMPLERS
        except Exception:
            return ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde", "ddim"]

    @staticmethod
    def _schedulers():
        try:
            import comfy.samplers

            return comfy.samplers.KSampler.SCHEDULERS
        except Exception:
            return ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "render"
    CATEGORY = "CharacterPose/Render"

    def render(
        self,
        character,
        pose_image,
        model,
        clip,
        vae,
        control_net,
        positive,
        negative,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        width,
        height,
        controlnet_strength,
        ipadapter_weight,
        ipadapter=None,
        clip_vision=None,
    ):
        ref = torch.from_numpy(character["reference_image"].astype(np.float32))[None, ...]

        model = _apply_ipadapter(model, ipadapter, clip_vision, ref, ipadapter_weight)

        pos = _encode_prompt(clip, positive)
        neg = _encode_prompt(clip, negative)
        pos, neg = _apply_controlnet(pos, neg, control_net, pose_image, controlnet_strength)

        latent = {"samples": torch.zeros([1, 4, height // 8, width // 8])}
        # Use zeros on correct device via VAE encode empty path if needed
        try:
            import comfy.model_management as mm

            latent["samples"] = torch.zeros(
                [1, 4, height // 8, width // 8],
                device=mm.intermediate_device(),
            )
        except Exception:
            pass

        samples = _common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler, pos, neg, latent, denoise=1.0
        )
        image = vae.decode(samples["samples"])
        if len(image.shape) == 5:
            image = image.reshape(-1, image.shape[-3], image.shape[-2], image.shape[-1])
        # ComfyUI VAE decode returns BCHW sometimes depending on version — normalize to BHWC
        if image.ndim == 4 and image.shape[1] in (3, 4) and image.shape[-1] not in (3, 4):
            image = image.movedim(1, -1)
        return (image, samples)


class CharacterRepair:
    """Light img2img pass to fix warp/artifacts while keeping identity.

    Optionally locks the target pose via ControlNet OpenPose so the repair
    pass does not drift back toward the source pose.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "character": ("CHARACTER",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("STRING", {"multiline": True, "default": "same character, clean details, fix hands, sharp lines, consistent outfit"}),
                "negative": ("STRING", {"multiline": True, "default": "blurry, deformed, different character, extra limbs"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (CharacterRender._samplers(),),
                "scheduler": (CharacterRender._schedulers(),),
                "denoise": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ipadapter_weight": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "ipadapter": ("IPADAPTER",),
                "clip_vision": ("CLIP_VISION",),
                "mask": ("MASK",),
                "control_net": ("CONTROL_NET",),
                "pose_image": ("IMAGE",),
                "controlnet_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "repair"
    CATEGORY = "CharacterPose/Render"

    def repair(
        self,
        image,
        character,
        model,
        clip,
        vae,
        positive,
        negative,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        ipadapter_weight,
        ipadapter=None,
        clip_vision=None,
        mask=None,
        control_net=None,
        pose_image=None,
        controlnet_strength=0.85,
    ):
        if denoise <= 0:
            return (image,)

        ref = torch.from_numpy(character["reference_image"].astype(np.float32))[None, ...]
        model = _apply_ipadapter(model, ipadapter, clip_vision, ref, ipadapter_weight)

        pos = _encode_prompt(clip, positive)
        neg = _encode_prompt(clip, negative)

        if control_net is not None and pose_image is not None and controlnet_strength > 0:
            pos, neg = _apply_controlnet(
                pos, neg, control_net, pose_image, float(controlnet_strength)
            )

        # Encode input image to latent
        pixels = image
        if pixels.shape[-1] == 4:
            pixels = pixels[:, :, :, :3]
        latent_samples = vae.encode(pixels[:, :, :, :3])
        latent = {"samples": latent_samples}
        if mask is not None:
            latent["noise_mask"] = mask

        samples = _common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler, pos, neg, latent, denoise=denoise
        )
        out = vae.decode(samples["samples"])
        if len(out.shape) == 5:
            out = out.reshape(-1, out.shape[-3], out.shape[-2], out.shape[-1])
        if out.ndim == 4 and out.shape[1] in (3, 4) and out.shape[-1] not in (3, 4):
            out = out.movedim(1, -1)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "CP_CharacterRender": CharacterRender,
    "CP_CharacterRepair": CharacterRepair,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CP_CharacterRender": "Character Render",
    "CP_CharacterRepair": "Character Repair",
}
