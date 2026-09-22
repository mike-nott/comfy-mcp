import struct

import pytest

from comfy_mcp.comfy import EVENT_PREVIEW_IMAGE, FORMAT_JPEG, FORMAT_PNG, ComfyError, Frame


def test_parse_png_output_frame():
    frame = Frame.parse(struct.pack(">II", EVENT_PREVIEW_IMAGE, FORMAT_PNG) + b"\x89PNG...")
    assert frame.is_png_output
    assert frame.payload == b"\x89PNG..."


def test_sampler_preview_is_not_output():
    frame = Frame.parse(struct.pack(">II", EVENT_PREVIEW_IMAGE, FORMAT_JPEG) + b"\xff\xd8")
    assert not frame.is_png_output


def test_short_frame():
    with pytest.raises(ComfyError):
        Frame.parse(b"\x00\x01")


async def test_history_outputs_polls_and_filters():
    from comfy_mcp.comfy import ComfyClient

    client = ComfyClient("http://example.invalid")
    calls = {"n": 0}

    async def fake_json(method, path, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {}
        return {"p1": {"outputs": {"15": {"gifs": [{"filename": "mcp-x_00001.mp4", "subfolder": "", "type": "temp"}]}}, "status": {"status_str": "success"}}}

    client._json = fake_json  # type: ignore[method-assign]
    found = await client.history_outputs("p1", "15")
    assert found == [{"filename": "mcp-x_00001.mp4", "subfolder": "", "type": "temp"}] and calls["n"] == 2

    async def error_json(method, path, **kwargs):
        return {"p1": {"outputs": {}, "status": {"status_str": "error"}}}

    client._json = error_json  # type: ignore[method-assign]
    with pytest.raises(ComfyError):
        await client.history_outputs("p1", "15", attempts=2)
    await client.aclose()


def test_temp_siblings():
    from comfy_mcp.comfy import temp_siblings

    assert temp_siblings("mcp-ab_00001-audio.mp4") == ["mcp-ab_00001-audio.mp4", "mcp-ab_00001.mp4", "mcp-ab_00001.png"]
    assert temp_siblings("mcp-ab_00001.mp4") == ["mcp-ab_00001.mp4", "mcp-ab_00001.png"]
    assert temp_siblings("noext") == ["noext"]


async def test_loader_choices_both_schema_shapes():
    from comfy_mcp.comfy import ComfyClient

    client = ComfyClient("http://example.invalid")
    schemas = {
        "UNETLoader": {"UNETLoader": {"input": {"required": {"unet_name": [["a.safetensors", "b.safetensors"], {}]}}}},
        "UpscaleModelLoader": {"UpscaleModelLoader": {"input": {"required": {"model_name": ["COMBO", {"options": ["span.safetensors"]}]}}}},
        "Nothing": {},
    }

    async def fake_json(method, path, **kwargs):
        return schemas[path.rsplit("/", 1)[1]]

    client._json = fake_json  # type: ignore[method-assign]
    assert await client.loader_choices("UNETLoader", "unet_name") == ["a.safetensors", "b.safetensors"]
    assert await client.loader_choices("UpscaleModelLoader", "model_name") == ["span.safetensors"]
    assert await client.loader_choices("Nothing", "x") == []
    await client.aclose()
