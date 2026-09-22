import pytest

from comfy_mcp.config import Qwen21Files
from comfy_mcp.recipes import qwen21

FILES = Qwen21Files()


def test_canvas_defaults_and_rounding():
    assert qwen21.canvas(None, None, aspect=None, max_pixels=2_097_152) == (1024, 1024)
    assert qwen21.canvas(1000, 700, aspect=None, max_pixels=2_097_152) == (992, 704)
    assert qwen21.canvas(None, None, aspect="landscape", max_pixels=2_097_152) == (1344, 768)
    w, h = qwen21.canvas(4000, 4000, aspect=None, max_pixels=2_097_152)
    assert w * h <= 2_097_152 + 32 * 32 * 4
    assert w % 32 == 0 and h % 32 == 0
    assert qwen21.canvas(None, None, aspect=None, max_pixels=2_097_152, fallback=(800, 600)) == (800, 608)
    assert qwen21.canvas(512, None, aspect=None, max_pixels=2_097_152, fallback=(1024, 512)) == (512, 256)
    with pytest.raises(ValueError):
        qwen21.canvas(None, None, aspect="cinema", max_pixels=1)


def test_fit_geometry_pads_centered():
    assert qwen21.fit_geometry(1000, 500, 1024, 1024) == (1024, 512, 0, 256, 0, 256)
    assert qwen21.fit_geometry(500, 1000, 1024, 1024) == (512, 1024, 256, 0, 256, 0)
    assert qwen21.fit_geometry(1024, 1024, 1024, 1024) == (1024, 1024, 0, 0, 0, 0)


def test_text_graph_shape():
    g = qwen21.text_graph(FILES, prompt="a fox", negative="", width=1024, height=768, seed=7, steps=25, cfg=1.0, count=2)
    assert g["6"]["class_type"] == "EmptyLatentImage"
    assert g["6"]["inputs"] == {"width": 1024, "height": 768, "batch_size": 2}
    assert g[qwen21.SAVE_NODE]["class_type"] == "SaveImageWebsocket"
    assert not any(node["class_type"] == "SaveImage" for node in g.values())
    assert g["7"]["inputs"]["seed"] == 7
    assert g["4"]["inputs"]["prompt"] == "a fox"
    assert g["1"]["inputs"]["unet_name"] == FILES.diffusion_model
    custom = qwen21.text_graph(Qwen21Files(diffusion_model="my.safetensors", text_encoder="enc.safetensors", vae="vae.safetensors"), prompt="x", negative="", width=1024, height=1024, seed=1, steps=1, cfg=1.0, count=1)
    assert custom["1"]["inputs"]["unet_name"] == "my.safetensors" and custom["2"]["inputs"]["clip_name"] == "enc.safetensors" and custom["3"]["inputs"]["vae_name"] == "vae.safetensors"
    assert "vae" not in g["4"]["inputs"]


def test_edit_graph_references():
    refs = [qwen21.Reference("mcp-a.png [temp]", 1000, 500), qwen21.Reference("mcp-b.png [temp]", 640, 640)]
    g = qwen21.edit_graph(FILES, instruction="swap", negative="", references=refs, width=1024, height=1024, seed=1, steps=25, cfg=1.0, count=1)
    enc = g["4"]["inputs"]
    assert enc["vae"] == ["3", 0] and enc["resolution"] == 0
    assert enc["images.image_1"] == ["22", 0]
    assert enc["images.image_2"] == ["26", 0]
    assert g["6"] == {"class_type": "RepeatLatentBatch", "inputs": {"samples": ["4", 2], "amount": 1}}
    assert g["20"]["inputs"]["image"] == "mcp-a.png [temp]"
    assert g["21"]["inputs"]["width"] == 1024 and g["21"]["inputs"]["height"] == 512
    assert g["22"]["inputs"]["top"] == 256 and g["22"]["inputs"]["bottom"] == 256
    assert g["26"]["class_type"] == "ImageScaleToTotalPixels"
    with pytest.raises(ValueError):
        qwen21.edit_graph(FILES, instruction="x", negative="", references=[], width=1024, height=1024, seed=1, steps=1, cfg=1.0, count=1)


def test_registry():
    from comfy_mcp.recipes import IMAGE_MODELS, image_model

    assert image_model("qwen21") is qwen21 and "qwen21" in IMAGE_MODELS
    with pytest.raises(ValueError, match="Unknown image model 'nope'. Available: qwen21"):
        image_model("nope")
