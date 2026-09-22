"""Image loading and downscaling.

JEV latency is dominated by visual prefill, which grows with the number of
image tokens. Inputs are downscaled before they reach the model.
"""

from __future__ import annotations

import base64
import binascii
import io
import math
from pathlib import Path
from typing import Any

from PIL import Image

from .config import ImageConfig


class ImageError(ValueError):
    pass


def load_image(source: Any) -> Image.Image:
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(source)))
    if isinstance(source, (str, Path)):
        text = str(source)
        if text.startswith("data:image/"):
            _, _, b64 = text.partition(",")
            return _open_b64(b64)
        path = Path(text)
        if path.exists():
            return Image.open(path)
        # last resort: raw base64 payload
        try:
            return _open_b64(text)
        except (binascii.Error, ValueError) as exc:
            raise ImageError(
                f"could not load image from {text[:60]!r} (not a path or base64)"
            ) from exc
    raise ImageError(f"unsupported image source type: {type(source)!r}")


def _open_b64(b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(b64, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ImageError("invalid base64 image payload") from exc
    return Image.open(io.BytesIO(raw))


def encode_images(sources: list[Any], cfg: ImageConfig) -> list[tuple[str, dict]]:
    return [encode_image(source, cfg) for source in sources]


def encode_image(source: Any, cfg: ImageConfig) -> tuple[str, dict]:
    """Downscale + encode an image, returning (data_url, info)."""
    img = load_image(source)
    original = img.size
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    scale = 1.0
    w, h = img.size
    if cfg.max_side and max(w, h) > cfg.max_side:
        scale = min(scale, cfg.max_side / max(w, h))
    if cfg.max_pixels and (w * h) > cfg.max_pixels:
        scale = min(scale, math.sqrt(cfg.max_pixels / (w * h)))
    if scale < 1.0:
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS
        )

    fmt = (cfg.format or "png").lower()
    buf = io.BytesIO()
    if fmt in ("jpg", "jpeg"):
        img.save(buf, format="JPEG", quality=92)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    raw = buf.getvalue()
    data_url = f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")
    info = {
        "original_width": original[0],
        "original_height": original[1],
        "width": img.size[0],
        "height": img.size[1],
        "scale": round(scale, 4),
        "bytes": len(raw),
        "mime": mime,
    }
    return data_url, info
