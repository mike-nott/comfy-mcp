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


def test_accel_chain_final_and_draft():
    from comfy_mcp.config import MinimaxH3Accel

    accel = MinimaxH3Accel(sol_attn=True, sage_patch=True, fused_modulation=True, chunk_feed_forward=True)
    g, _ = minimax_h3.graph(FILES, prompt="p", mode="t2v", images=[], audio_token=None, seconds=5, orientation="landscape", draft=False, seed=1, accel=accel)
    # order: loader -> Sage KJ -> Sage H3 -> Sol -> Fused -> Chunk FF -> FBC -> (guider + scheduler)
    assert g["22"]["class_type"] == "PathchSageAttentionKJ" and g["22"]["inputs"] == {"model": ["1", 0], "sage_attention": "auto"}
    assert g["23"] == {"class_type": "MiniMaxH3MemoryEfficientSageAttentionPatch", "inputs": {"model": ["22", 0]}}
    sol = g["24"]
    assert sol["class_type"] == "MiniMaxH3MemoryEfficientSolAttentionPatch" and sol["inputs"]["model"] == ["23", 0]
    assert sol["inputs"]["tau"] == 1.3 and sol["inputs"]["strict"] is True and sol["inputs"]["int8_qk"] is True and sol["inputs"]["sink_conditioning"] == "exact_kv_and_rows"
    assert g["26"] == {"class_type": "MiniMaxH3FusedModulation", "inputs": {"model": ["24", 0], "enabled": True}}
    assert g["27"]["class_type"] == "MiniMaxH3ChunkFeedForward" and g["27"]["inputs"]["model"] == ["26", 0] and g["27"]["inputs"]["chunks"] == 2
    fbc = g["25"]
    assert fbc["class_type"] == "ApplyMiniMaxH3FirstBlockCache" and fbc["inputs"]["model"] == ["27", 0]
    assert fbc["inputs"]["mode"] == "Custom — manual values" and fbc["inputs"]["threshold"] == 0.08 and fbc["inputs"]["start_percent"] == 0.15
    assert fbc["inputs"]["end_percent"] == 0.9 and fbc["inputs"]["max_consecutive_hits"] == 2 and fbc["inputs"]["temporal_guard"] is False
    assert g["8"]["inputs"]["model"] == ["25", 0] and g["10"]["inputs"]["model"] == ["25", 0]
    assert g["10"]["inputs"]["steps"] == 20 and g["6"]["inputs"]["width"] == 1280 and "28" not in g and "50" not in g
    assert minimax_h3.required_nodes(accel, draft=False)[-1] == "ApplyMiniMaxH3FirstBlockCache"
    assert minimax_h3.accel_summary(accel, draft=False) == ["sage", "sol-attn tau 1.3", "fused-mod", "chunk-ff", "fbc@0.08"]

    g, _ = minimax_h3.graph(FILES, prompt="p", mode="t2v", images=[], audio_token=None, seconds=5, orientation="landscape", draft=True, seed=1, accel=accel)
    # draft: turbo LoRA first, no cache, small canvas, SPAN upscale then rescale to the final canvas
    assert g["20"]["inputs"]["model"] == ["1", 0] and g["22"]["inputs"]["model"] == ["20", 0]
    assert "25" not in g and g["8"]["inputs"]["model"] == ["27", 0]
    assert g["6"]["inputs"]["width"] == 960 and g["6"]["inputs"]["height"] == 544 and g["10"]["inputs"]["steps"] == 8
    assert g["50"]["inputs"]["model_name"] == "2xNomosUni_span_multijpg.safetensors"
    assert g["51"]["inputs"]["image"] == ["12", 0] and g["52"]["inputs"] == {"image": ["51", 0], "upscale_method": "lanczos", "width": 1280, "height": 704, "crop": "disabled"}
    assert g[minimax_h3.VIDEO_NODE]["inputs"]["images"] == ["52", 0] and g["16"]["inputs"]["image"] == ["52", 0]
    assert "ImageUpscaleWithModel" in minimax_h3.required_nodes(accel, draft=True)
    assert minimax_h3.accel_summary(accel, draft=True)[-1] == "draft 960x544+upscale"

    g, _ = minimax_h3.graph(FILES, prompt="p", mode="t2v", images=[], audio_token=None, seconds=5, orientation="portrait", draft=True, seed=1, accel=accel)
    assert (g["6"]["inputs"]["width"], g["6"]["inputs"]["height"]) == (544, 960)


def test_accel_variants():
    from comfy_mcp.config import MinimaxH3Accel

    # Sage without the KJ node, and Spectrum instead of FBC
    accel = MinimaxH3Accel(sage_patch=True, sage_kj_node="", sol_attn=False, spectrum=True, first_block_cache=False)
    g, _ = minimax_h3.graph(FILES, prompt="p", mode="t2v", images=[], audio_token=None, seconds=5, orientation="landscape", draft=False, seed=1, accel=accel)
    assert "22" not in g and g["23"]["inputs"]["model"] == ["1", 0]
    assert g["28"]["class_type"] == "SpectrumApplyMiniMaxH3" and g["28"]["inputs"]["model"] == ["23", 0] and g["28"]["inputs"]["blend_weight"] == 0.5
    assert "25" not in g and g["8"]["inputs"]["model"] == ["28", 0]
    assert "SpectrumApplyMiniMaxH3" in minimax_h3.required_nodes(accel, draft=False) and "SpectrumApplyMiniMaxH3" not in minimax_h3.required_nodes(accel, draft=True)
    with pytest.raises(ValueError, match="spectrum and first_block_cache"):
        minimax_h3.graph(FILES, prompt="p", mode="t2v", images=[], audio_token=None, seconds=5, orientation="landscape", draft=False, seed=1, accel=MinimaxH3Accel(spectrum=True))


def test_accel_defaults_and_steps_override():
    from comfy_mcp.config import MinimaxH3Accel

    accel = MinimaxH3Accel()  # defaults: sage + sol + fbc on, fusion/spectrum off, upscale on
    g, _ = minimax_h3.graph(FILES, prompt="p", mode="t2v", images=[], audio_token=None, seconds=5, orientation="landscape", draft=False, seed=1, accel=accel, steps=14)
    assert g["22"]["inputs"]["model"] == ["1", 0] and g["24"]["inputs"]["model"] == ["23", 0] and "26" not in g and "28" not in g
    assert g["25"]["inputs"]["model"] == ["24", 0] and g["10"]["inputs"]["steps"] == 14
    off = MinimaxH3Accel(first_block_cache=False, draft_upscale=False, sol_attn=False, sage_patch=False)
    g, _ = minimax_h3.graph(FILES, prompt="p", mode="t2v", images=[], audio_token=None, seconds=5, orientation="landscape", draft=True, seed=1, accel=off)
    assert "25" not in g and "50" not in g and "22" not in g and g["6"]["inputs"]["width"] == 960 and g[minimax_h3.VIDEO_NODE]["inputs"]["images"] == ["12", 0]
    assert minimax_h3.required_nodes(off, draft=True) == minimax_h3.REQUIRED_NODES
    # no accel object: legacy behaviour
    g, _ = minimax_h3.graph(FILES, prompt="p", mode="t2v", images=[], audio_token=None, seconds=5, orientation="landscape", draft=False, seed=1)
    assert g["8"]["inputs"]["model"] == ["1", 0] and g["10"]["inputs"]["steps"] == 20
