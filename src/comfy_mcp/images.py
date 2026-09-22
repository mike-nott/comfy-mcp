"""Client-side image handling: inputs, previews and saving. Nothing here logs."""

from __future__ import annotations

import base64
import io
import re
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

MAX_INPUT_BYTES = 64 * 1024 * 1024
_DATA_URL = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)
_BASE64_CHARS = re.compile(r"^[A-Za-z0-9+/=\s]+$")


def load_input(item: str) -> tuple[bytes, int, int]:
    """Accept a local path, file:// URL, data: URL or raw base64. Return (png_bytes, width, height).

    The image is re-encoded as PNG with orientation applied and metadata dropped.
    """
    item = item.strip()
    raw: bytes
    if _DATA_URL.match(item):
        raw = base64.b64decode(_DATA_URL.sub("", item), validate=False)
    elif item.startswith("file://"):
        raw = _read(Path(item[7:]))
    elif _looks_like_base64(item):
        raw = base64.b64decode(item, validate=False)
    else:
        raw = _read(Path(item).expanduser())
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("Reference image exceeds 64 MB")
    with Image.open(io.BytesIO(raw)) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        out = io.BytesIO()
        image.save(out, format="PNG", compress_level=1)
        return out.getvalue(), image.width, image.height


def _looks_like_base64(item: str) -> bool:
    if len(item) < 128 or "/" in item[:64] or not _BASE64_CHARS.match(item):
        return False
    try:
        return not Path(item).expanduser().is_file()
    except OSError:
        return True


def _read(path: Path) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"Reference image not found: {path}")
    return path.read_bytes()


def preview_jpeg(png: bytes, max_px: int, quality: int = 85) -> bytes:
    with Image.open(io.BytesIO(png)) as image:
        image = image.convert("RGB")
        image.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=quality, optimize=True)
        return out.getvalue()


def dimensions(png: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(png)) as image:
        return image.width, image.height


def output_name(seed: int, index: int, count: int, when: datetime | None = None) -> str:
    stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    suffix = f"-{index + 1}" if count > 1 else ""
    return f"{stamp}-{seed}{suffix}.png"


def save_png(png: bytes, output_dir: Path, seed: int, index: int, count: int, when: datetime | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / output_name(seed, index, count, when)
    counter = 0
    while path.exists():
        counter += 1
        path = output_dir / output_name(seed, index, count, when).replace(".png", f"-{counter}.png")
    path.write_bytes(png)
    return path
