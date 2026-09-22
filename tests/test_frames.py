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
