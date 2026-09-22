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


AUDIO_SUFFIXES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".aac": "audio/aac"}
_AUDIO_DATA_URL = re.compile(r"^data:audio/([a-zA-Z0-9.+-]+);base64,", re.IGNORECASE)


def load_audio_input(item: str) -> tuple[bytes, str]:
    """Accept a local path, file:// URL, data:audio URL or raw base64 (assumed WAV). Returns (bytes, suffix).

    Audio is passed through untouched; ComfyUI's LoadAudio decodes it.
    """
    item = item.strip()
    match = _AUDIO_DATA_URL.match(item)
    if match:
        kind = match.group(1).lower()
        suffix = {"wav": ".wav", "x-wav": ".wav", "wave": ".wav", "mpeg": ".mp3", "mp3": ".mp3", "flac": ".flac", "mp4": ".m4a", "ogg": ".ogg", "aac": ".aac"}.get(kind, ".wav")
        raw = base64.b64decode(_AUDIO_DATA_URL.sub("", item), validate=False)
    elif item.startswith("file://"):
        path = Path(item[7:])
        raw, suffix = _read(path), path.suffix.lower()
    elif _looks_like_base64(item):
        raw, suffix = base64.b64decode(item, validate=False), ".wav"
    else:
        path = Path(item).expanduser()
        raw, suffix = _read(path), path.suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        raise ValueError(f"Unsupported audio type '{suffix}'; use one of {', '.join(AUDIO_SUFFIXES)}")
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("Audio clip exceeds 64 MB")
    return raw, suffix


def output_name(seed: int, index: int, count: int, when: datetime | None = None, ext: str = ".png") -> str:
    stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    suffix = f"-{index + 1}" if count > 1 else ""
    return f"{stamp}-{seed}{suffix}{ext}"


def save_bytes(data: bytes, output_dir: Path, seed: int, index: int, count: int, ext: str, when: datetime | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / output_name(seed, index, count, when, ext)
    counter = 0
    while path.exists():
        counter += 1
        path = output_dir / output_name(seed, index, count, when, ext).replace(ext, f"-{counter}{ext}")
    path.write_bytes(data)
    return path


def save_png(png: bytes, output_dir: Path, seed: int, index: int, count: int, when: datetime | None = None) -> Path:
    return save_bytes(png, output_dir, seed, index, count, ".png", when)
