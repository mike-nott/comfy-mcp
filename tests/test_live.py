"""End-to-end against a real ComfyUI through the real MCP stdio transport.

Run with: COMFY_MCP_LIVE=1 uv run pytest tests/test_live.py -s
Optional: COMFYUI_URL, COMFY_MCP_OUTPUT_DIR (defaults to a temp dir here).
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from contextlib import asynccontextmanager

import pytest
from PIL import Image

pytestmark = pytest.mark.skipif(not os.environ.get("COMFY_MCP_LIVE"), reason="set COMFY_MCP_LIVE=1 to run against a real ComfyUI")


@asynccontextmanager
async def mcp_session(tmp_path, config_text: str | None = None):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = {**os.environ, "COMFY_MCP_OUTPUT_DIR": os.environ.get("COMFY_MCP_OUTPUT_DIR", str(tmp_path / "out"))}
    args = ["-m", "comfy_mcp.cli", "--debug"]
    if config_text is not None:
        cfg = tmp_path / "config.toml"
        cfg.write_text(config_text)
        args += ["--config", str(cfg)]
    params = StdioServerParameters(command=sys.executable, args=args, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()
            yield client


def _text(result) -> str:
    return "\n".join(block.text for block in result.content if block.type == "text")


def _images(result):
    return [block for block in result.content if block.type == "image"]


async def test_status_and_models(tmp_path):
  async with mcp_session(tmp_path) as client:
    status = json.loads(_text(await client.call_tool("server_status", {})))
    print("\nstatus:", status)
    assert status["comfyui_version"]
    models = json.loads(_text(await client.call_tool("list_models", {})))
    assert models["image_models"]["qwen21"]["ready"], models["image_models"]
    assert models["image_models"]["qwen21"]["default"] is True


async def test_generate(tmp_path):
  async with mcp_session(tmp_path) as client:
    result = await client.call_tool(
        "generate_image",
        {"prompt": "a red teapot and a blue cup on a wooden table, soft window light, photo", "model": "qwen21", "width": 768, "height": 768, "seed": 4242, "steps": 12},
    )
    text = _text(result)
    print("\n" + text)
    assert not result.is_error, text
    assert "seed 4242" in text and "saved:" in text
    previews = _images(result)
    assert len(previews) == 1
    with Image.open(io.BytesIO(base64.b64decode(previews[0].data))) as img:
        assert max(img.size) <= 512
    saved = [line.split("saved: ", 1)[1] for line in text.splitlines() if line.startswith("saved:")]
    with Image.open(saved[0]) as img:
        assert img.size == (768, 768) and img.format == "PNG"


async def test_edit_one_and_two_references(tmp_path):
  async with mcp_session(tmp_path) as client:
    ref1 = tmp_path / "ref1.png"
    ref2 = tmp_path / "ref2.png"
    Image.new("RGB", (640, 480), (30, 120, 200)).save(ref1)
    Image.new("RGB", (480, 480), (220, 180, 40)).save(ref2)

    single = await client.call_tool(
        "edit_image", {"prompt": "Turn this plain blue image into a calm sea under a cloudy sky", "reference_images": [str(ref1)], "seed": 7, "steps": 12}
    )
    text = _text(single)
    print("\n" + text)
    assert not single.is_error, text
    assert "640×480" in text

    double = await client.call_tool(
        "edit_image",
        {
            "instruction": "Compose a scene: use the colour of <image1> for the sky and the colour of <image2> for a sandy beach below it",
            "images": [str(ref1), str(ref2)],
            "seed": 8,
            "steps": 12,
            "width": 768,
            "height": 512,
        },
    )
    text = _text(double)
    print("\n" + text)
    assert not double.is_error, text
    assert "768×512" in text and len(_images(double)) == 1


async def test_job_flow_when_wait_is_short(tmp_path):
  async with mcp_session(tmp_path, "max_wait = 1\n") as client:
    result = await client.call_tool("generate_image", {"prompt": "a green apple on white", "width": 512, "height": 512, "seed": 99, "steps": 8})
    text = _text(result)
    print("\n" + text)
    assert not result.is_error, text
    if "saved:" in text:
        pytest.skip("finished within 1 s; server unexpectedly fast")
    pending = json.loads(text)
    assert pending["state"] in ("queued", "running") and pending["job_id"]
    status = json.loads(_text(await client.call_tool("job_status", {"job_id": pending["job_id"]})))
    assert status["job_id"] == pending["job_id"]
    done = await client.call_tool("wait_for_job", {"job_id": pending["job_id"], "timeout": 180})
    text = _text(done)
    print("\n" + text)
    assert not done.is_error, text
    assert "seed 99" in text and "saved:" in text and len(_images(done)) == 1
    path = [line for line in text.splitlines() if line.startswith("saved:")][0]
    # A second delivery (e.g. after the first caller's client timed out) returns the same file, not an error.
    again = await client.call_tool("fetch_result", {"job_id": pending["job_id"]})
    assert not again.is_error and path in _text(again) and len(_images(again)) == 1
    status = json.loads(_text(await client.call_tool("job_status", {"job_id": pending["job_id"]})))
    assert status["state"] == "done" and status["saved"] == [path.split("saved: ", 1)[1]]


async def test_bad_reference_is_a_clean_error(tmp_path):
  async with mcp_session(tmp_path) as client:
    result = await client.call_tool("edit_image", {"instruction": "x", "images": ["/nonexistent/ref.png"]})
    text = _text(result)
    print("\n" + text)
    assert result.is_error and "Reference image not found: /nonexistent/ref.png" in text and "Traceback" not in text
    result = await client.call_tool("generate_image", {"prompt": "x", "aspect": "cinema"})
    text = _text(result)
    assert result.is_error and "Unknown aspect" in text
    result = await client.call_tool("generate_image", {"prompt": "x", "model": "sdxl"})
    text = _text(result)
    assert result.is_error and "Unknown image model 'sdxl'" in text and "qwen21" in text


async def test_misconfigured_model_file_is_a_clear_error(tmp_path):
  async with mcp_session(tmp_path, '[qwen21]\nvae = "does_not_exist.safetensors"\n') as client:
    models = json.loads(_text(await client.call_tool("list_models", {})))
    assert models["image_models"]["qwen21"]["ready"] is False and models["image_models"]["qwen21"]["missing"] == ["vae = 'does_not_exist.safetensors'"]
    result = await client.call_tool("generate_image", {"prompt": "x"})
    text = _text(result)
    print("\n" + text)
    assert result.is_error and "does_not_exist.safetensors" in text and "[qwen21]" in text


def _video_info(path: str) -> dict:
    """Width/height/duration via ffprobe when available, else a minimal MP4 sanity check."""
    import shutil
    import subprocess

    head = open(path, "rb").read(12)
    assert head[4:8] == b"ftyp", "not an MP4"
    if shutil.which("ffprobe"):
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration", "-of", "json", path],
                capture_output=True, text=True, check=True, timeout=30,
            ).stdout
            data = json.loads(out)
            return {"width": data["streams"][0]["width"], "height": data["streams"][0]["height"], "duration": float(data["format"]["duration"])}
        except (subprocess.SubprocessError, ValueError, KeyError, IndexError) as error:
            print("ffprobe unavailable:", error.__class__.__name__)
    return {}


async def test_video_t2v_draft(tmp_path):
  async with mcp_session(tmp_path) as client:
    models = json.loads(_text(await client.call_tool("list_models", {})))
    assert models["video_models"]["minimax_h3"]["ready"], models["video_models"]
    started = time.monotonic()
    result = await client.call_tool(
        "generate_video",
        {"prompt": "A red fox trots across a snowy meadow at dawn, low sun, gentle camera pan following it, soft wind sound", "seconds": 5, "draft": True, "seed": 11},
    )
    text = _text(result)
    print("\n" + text)
    assert not result.is_error, text
    pending = json.loads(text)
    assert pending["media"] == "video" and pending["job_id"] and pending["seconds"] == 5.17
    done = await client.call_tool("wait_for_job", {"job_id": pending["job_id"], "timeout": 1500})
    text = _text(done)
    print("\n" + text + f"\n[wall {time.monotonic() - started:.0f} s]")
    assert not done.is_error, text
    assert "MiniMax H3 · t2v · 1280×704 · 5.17 s · seed 11 · 8 steps (draft)" in text
    saved = [line.split("saved: ", 1)[1] for line in text.splitlines() if line.startswith("saved:")]
    assert len(saved) == 1 and saved[0].endswith(".mp4")
    info = _video_info(saved[0])
    print("ffprobe:", info)
    if info:
        assert (info["width"], info["height"]) == (1280, 704) and 5.0 <= info["duration"] <= 5.4
    assert len(_images(done)) == 1  # poster frame
    again = await client.call_tool("fetch_result", {"job_id": pending["job_id"]})
    assert not again.is_error and saved[0] in _text(again)


async def _run_video(client, args: dict, expect: str, timeout: int = 1800) -> tuple[str, list[str]]:
    started = time.monotonic()
    result = await client.call_tool("generate_video", args)
    text = _text(result)
    assert not result.is_error, text
    pending = json.loads(text)
    done = await client.call_tool("wait_for_job", {"job_id": pending["job_id"], "timeout": timeout})
    text = _text(done)
    print("\n" + text + f"\n[wall {time.monotonic() - started:.0f} s]")
    assert not done.is_error, text
    assert expect in text, text
    saved = [line.split("saved: ", 1)[1] for line in text.splitlines() if line.startswith("saved:")]
    assert len(saved) == 1 and saved[0].endswith(".mp4") and len(_images(done)) == 1
    _video_info(saved[0])
    return text, saved


async def test_video_t2v_final(tmp_path):
  async with mcp_session(tmp_path) as client:
    await _run_video(client, {"prompt": "A red fox trots across a snowy meadow at dawn, low sun, gentle camera pan following it, soft wind sound", "seconds": 5, "seed": 11}, "t2v · 1280×704 · 5.17 s · seed 11 · 20 steps")


async def test_video_i2v_from_image(tmp_path):
  async with mcp_session(tmp_path) as client:
    still = await client.call_tool("generate_image", {"prompt": "a small wooden sailboat on a calm lake at sunset, photo", "width": 1280, "height": 704, "seed": 5, "steps": 12})
    path = [line.split("saved: ", 1)[1] for line in _text(still).splitlines() if line.startswith("saved:")][0]
    await _run_video(client, {"prompt": "The sailboat drifts slowly to the right as ripples spread, camera holds still, water lapping sounds", "mode": "i2v", "images": [path], "seconds": 5, "draft": True, "seed": 12}, "i2v · 1280×704")


async def test_video_r2v_and_refav(tmp_path):
  async with mcp_session(tmp_path) as client:
    still = await client.call_tool("generate_image", {"prompt": "portrait of a friendly robot with a round blue head, studio photo", "width": 704, "height": 704, "seed": 6, "steps": 12})
    path = [line.split("saved: ", 1)[1] for line in _text(still).splitlines() if line.startswith("saved:")][0]
    await _run_video(client, {"prompt": "The robot from the reference waves at the camera in a sunny park, birds chirping", "mode": "r2v", "images": [path], "seconds": 5, "seed": 13}, "r2v · 1280×704")
    import math
    import struct
    import wave

    clip = tmp_path / "tone.wav"
    with wave.open(str(clip), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / 16000))) for i in range(16000 * 3)))
    await _run_video(client, {"prompt": "The robot from the reference hums along to the tone while nodding", "mode": "refav", "images": [path], "audio": str(clip), "seconds": 5, "seed": 14}, "refav · 1280×704")


async def test_video_bench(tmp_path):
    """One measured H3 render with settings supplied by environment, for A/B runs.

    COMFY_MCP_BENCH_CONFIG: TOML text for the server config (e.g. a [minimax_h3.accel] block).
    COMFY_MCP_BENCH_ARGS: JSON for generate_video's arguments; seed should vary between runs so
    ComfyUI's execution cache cannot serve a cached sampler result.
    """
    cfg = os.environ.get("COMFY_MCP_BENCH_CONFIG")
    args = os.environ.get("COMFY_MCP_BENCH_ARGS")
    if not cfg or not args:
        pytest.skip("set COMFY_MCP_BENCH_CONFIG and COMFY_MCP_BENCH_ARGS")
    async with mcp_session(tmp_path, cfg) as client:
        text, saved = await _run_video(client, json.loads(args), "MiniMax H3 ·", timeout=2400)
        print("BENCH_RESULT", text.splitlines()[0], saved[0])
