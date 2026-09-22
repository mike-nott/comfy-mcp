import base64
import io
from datetime import datetime

from PIL import Image

from comfy_mcp import images


def _png(w=64, h=32, mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, (w, h), (200, 100, 50) if mode == "RGB" else (200, 100, 50, 128)).save(buf, format="PNG")
    return buf.getvalue()


def test_load_input_from_path_and_base64(tmp_path):
    path = tmp_path / "ref.png"
    path.write_bytes(_png())
    png, w, h = images.load_input(str(path))
    assert (w, h) == (64, 32) and png.startswith(b"\x89PNG")
    data_url = "data:image/png;base64," + base64.b64encode(_png(10, 10)).decode()
    _, w, h = images.load_input(data_url)
    assert (w, h) == (10, 10)
    _, w, h = images.load_input(base64.b64encode(_png(300, 200)).decode())
    assert (w, h) == (300, 200)


def test_load_input_missing():
    try:
        images.load_input("/nonexistent/file.png")
    except FileNotFoundError:
        return
    raise AssertionError


def test_preview_and_dimensions():
    preview = images.preview_jpeg(_png(2048, 1024), 512)
    with Image.open(io.BytesIO(preview)) as img:
        assert img.format == "JPEG" and img.size == (512, 256)
    assert images.dimensions(_png(2048, 1024)) == (2048, 1024)


def test_save_names(tmp_path):
    when = datetime(2026, 9, 22, 10, 30, 0)
    first = images.save_png(_png(), tmp_path, 42, 0, 1, when)
    assert first.name == "20260922-103000-42.png"
    second = images.save_png(_png(), tmp_path, 42, 0, 1, when)
    assert second.name == "20260922-103000-42-1.png"
    multi = images.save_png(_png(), tmp_path, 42, 1, 2, when)
    assert multi.name == "20260922-103000-42-2.png"


def test_load_audio_input(tmp_path):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 64)
    data, suffix = images.load_audio_input(str(wav))
    assert suffix == ".wav" and data.startswith(b"RIFF")
    data, suffix = images.load_audio_input("data:audio/mpeg;base64," + base64.b64encode(b"ID3" + b"\x00" * 10).decode())
    assert suffix == ".mp3"
    bad = tmp_path / "clip.txt"
    bad.write_bytes(b"x")
    try:
        images.load_audio_input(str(bad))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unsupported suffix")


def test_save_bytes_ext(tmp_path):
    when = datetime(2026, 9, 22, 10, 30, 0)
    path = images.save_bytes(b"\x00", tmp_path, 5, 0, 1, ".mp4", when)
    assert path.name == "20260922-103000-5.mp4"
