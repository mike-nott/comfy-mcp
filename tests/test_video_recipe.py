import pytest

from comfy_mcp.config import MinimaxH3Files
from comfy_mcp.recipes import VIDEO_MODELS, minimax_h3, video_model

FILES = MinimaxH3Files()
REF = minimax_h3.Reference("mcp-a.png [temp]", 1280, 704)


def test_frame_grid():
    assert minimax_h3.frames(5) == 124
    assert minimax_h3.frames(1) == 124
    assert minimax_h3.frames(10) == 243
    assert minimax_h3.frames(15) == 362
    assert minimax_h3.frames(60) == 362
    assert minimax_h3.seconds_for(124) == 5.17
    with pytest.raises(ValueError):
        minimax_h3.frames(0)


def test_canvas():
    assert minimax_h3.canvas("landscape") == (1280, 704)
    assert minimax_h3.canvas("portrait") == (704, 1280)
    with pytest.raises(ValueError):
        minimax_h3.canvas("square")


def _graph(**kw):
    base = dict(prompt="a fox runs", mode="t2v", images=[], audio_token=None, seconds=5, orientation="landscape", draft=False, seed=3)
    base.update(kw)
    return minimax_h3.graph(FILES, **base)


def test_t2v_final_graph():
    g, n = _graph()
    assert n == 124
    assert g["1"]["inputs"]["unet_name"] == FILES.diffusion_model
    assert g["6"]["class_type"] == "MiniMaxH3ImageToVideo" and g["6"]["inputs"]["length"] == 124
    assert g["8"]["inputs"]["model"] == ["1", 0] and g["10"]["inputs"]["steps"] == 20
    assert g["11"]["inputs"]["sampler"] == ["9", 0]
    vhs = g[minimax_h3.VIDEO_NODE]["inputs"]
    assert g[minimax_h3.VIDEO_NODE]["class_type"] == "VHS_VideoCombine"
    assert vhs["save_output"] is False and vhs["format"] == "video/h264-mp4" and vhs["audio"] == ["13", 0] and vhs["save_metadata"] is False
    assert vhs["filename_prefix"].startswith("mcp-") and "/" not in vhs["filename_prefix"]
    assert g["16"] == {"class_type": "ImageFromBatch", "inputs": {"image": ["12", 0], "batch_index": 0, "length": 1}}
    assert g[minimax_h3.POSTER_NODE] == {"class_type": "SaveImageWebsocket", "inputs": {"images": ["16", 0]}}
    assert not any(node["class_type"] in ("SaveImage", "SaveVideo") for node in g.values())
    assert "20" not in g and "40" not in g


def test_draft_wiring():
    g, _ = _graph(draft=True)
    assert g["20"]["class_type"] == "MiniMaxH3TurboLoRA" and g["20"]["inputs"]["lora_name"] == FILES.turbo_lora
    assert g["8"]["inputs"]["model"] == ["20", 0] and g["10"]["inputs"]["steps"] == 8
    assert g["11"]["inputs"]["sampler"] == ["21", 0] and g["21"]["class_type"] == "MiniMaxH3TurboSampler"
    with pytest.raises(ValueError):
        _graph(draft=True, mode="r2v", images=[REF])


def test_i2v_first_and_last():
    g, _ = _graph(mode="i2v", images=[REF])
    assert g["30"]["inputs"]["image"] == REF.token and g["6"]["inputs"]["first_frame"] == ["30", 0]
    assert "last_frame" not in g["6"]["inputs"]
    g, _ = _graph(mode="i2v", images=[REF, minimax_h3.Reference("mcp-b.png [temp]", 1, 1)])
    assert g["6"]["inputs"]["last_frame"] == ["31", 0]
    with pytest.raises(ValueError):
        _graph(mode="i2v", images=[])
    with pytest.raises(ValueError):
        _graph(mode="t2v", images=[REF])


def test_r2v_and_refav():
    g, _ = _graph(mode="r2v", images=[REF, REF], seconds=10, orientation="portrait")
    assert g["1"]["inputs"]["unet_name"] == FILES.reference_model
    assert g["6"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    assert g["6"]["inputs"]["audio_vae"] == ["4", 0] and g["6"]["inputs"]["ref_image_size"] == "match"
    assert g["6"]["inputs"]["ref_images.ref_image_0"] == ["30", 0] and g["6"]["inputs"]["ref_images.ref_image_1"] == ["31", 0]
    assert g["6"]["inputs"]["width"] == 704 and g["6"]["inputs"]["length"] == 243
    with pytest.raises(ValueError):
        _graph(mode="refav", images=[REF])
    g, _ = _graph(mode="refav", images=[REF], audio_token="mcp-c.wav [temp]")
    assert g["40"] == {"class_type": "LoadAudio", "inputs": {"audio": "mcp-c.wav [temp]"}}
    assert g["6"]["inputs"]["ref_audios.ref_audio_0"] == ["40", 0]
    with pytest.raises(ValueError):
        _graph(mode="t2v", audio_token="x [temp]")
    with pytest.raises(ValueError):
        _graph(mode="r2v", images=[REF] * 4)


def test_registry():
    assert video_model("minimax_h3") is minimax_h3 and "minimax_h3" in VIDEO_MODELS
    with pytest.raises(ValueError, match="Unknown video model"):
        video_model("sora")
