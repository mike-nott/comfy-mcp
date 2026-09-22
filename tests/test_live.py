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
        "edit_image", {"instruction": "Turn this plain blue image into a calm sea under a cloudy sky", "images": [str(ref1)], "seed": 7, "steps": 12}
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
    gone = await client.call_tool("job_status", {"job_id": pending["job_id"]})
    assert gone.is_error


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
