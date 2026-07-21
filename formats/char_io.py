"""Character format (.char) — identity bundle (reference image + palette + metadata)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file


CHAR_VERSION = 1


def make_character(
    name: str,
    reference_image: np.ndarray,
    palette: np.ndarray | None = None,
    embedding: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a CHARACTER dict.

    reference_image: float32 HWC RGB in [0, 1]
    palette: float32 (K, 3) RGB in [0, 1]
    embedding: float32 vector (optional CLIP-Vision embedding)
    """
    if reference_image.dtype != np.float32:
        reference_image = reference_image.astype(np.float32)
    if reference_image.ndim != 3 or reference_image.shape[2] != 3:
        raise ValueError("reference_image must be HWC RGB")

    return {
        "version": CHAR_VERSION,
        "name": name or "character",
        "reference_image": reference_image,
        "palette": palette.astype(np.float32) if palette is not None else np.zeros((0, 3), dtype=np.float32),
        "embedding": embedding.astype(np.float32) if embedding is not None else np.zeros((0,), dtype=np.float32),
        "metadata": metadata or {},
    }


def save_character(character: dict[str, Any], path: str | Path) -> None:
    """Save CHARACTER as a .char directory or single sidecar pair.

    Layout:
      name.char/              (folder named *.char)
        meta.json
        tensors.safetensors   (reference_image flattened + palette + embedding)
    """
    path = Path(path)
    if path.suffix.lower() != ".char":
        path = path.with_suffix(".char")

    # Use a directory package named *.char
    if path.exists() and path.is_file():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)

    ref = character["reference_image"].astype(np.float32)
    h, w, c = ref.shape
    palette = character.get("palette")
    if palette is None:
        palette = np.zeros((0, 3), dtype=np.float32)
    else:
        palette = np.asarray(palette, dtype=np.float32)
        if palette.ndim == 1:
            palette = palette.reshape(-1, 3)

    embedding = character.get("embedding")
    if embedding is None:
        embedding = np.zeros((0,), dtype=np.float32)
    else:
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)

    tensors = {
        "reference_image": ref.reshape(-1),
        "palette": palette.reshape(-1),
        "embedding": embedding,
        "ref_shape": np.array([h, w, c], dtype=np.int64),
        "palette_shape": np.array(palette.shape, dtype=np.int64),
    }
    save_file(tensors, str(path / "tensors.safetensors"))

    meta = {
        "version": int(character.get("version", CHAR_VERSION)),
        "name": character.get("name", path.stem),
        "metadata": character.get("metadata", {}),
    }
    with (path / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_character(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Accept either folder.char/ or folder.char with files inside
    if path.is_file() and path.suffix == ".safetensors":
        # legacy / alternate: tensors beside meta.json
        tensor_path = path
        meta_path = path.with_name("meta.json")
        base_name = path.stem
    else:
        if path.is_file():
            raise ValueError(f".char path should be a directory: {path}")
        tensor_path = path / "tensors.safetensors"
        meta_path = path / "meta.json"
        base_name = path.stem

    tensors = load_file(str(tensor_path))
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    ref_shape = tuple(int(x) for x in tensors["ref_shape"].tolist())
    ref = tensors["reference_image"].reshape(ref_shape).astype(np.float32)

    palette_shape = tuple(int(x) for x in tensors["palette_shape"].tolist())
    palette = tensors["palette"].reshape(palette_shape).astype(np.float32) if palette_shape[0] > 0 or len(palette_shape) > 1 else np.zeros((0, 3), dtype=np.float32)
    if palette.size == 0:
        palette = np.zeros((0, 3), dtype=np.float32)

    embedding = tensors["embedding"].astype(np.float32)

    return {
        "version": int(meta.get("version", CHAR_VERSION)),
        "name": meta.get("name", base_name),
        "reference_image": ref,
        "palette": palette,
        "embedding": embedding,
        "metadata": meta.get("metadata", {}),
    }
