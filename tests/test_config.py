from pathlib import Path

from comfy_mcp import config


def test_defaults_without_file(tmp_path, monkeypatch):
    monkeypatch.setenv(config.ENV_CONFIG, str(tmp_path / "missing.toml"))
    settings = config.load(env={})
    assert settings.comfyui_url == "http://127.0.0.1:8188"
    assert settings.output_dir == Path("~/Pictures/ComfyUI").expanduser()
    assert settings.preview_px == 512
    assert settings.source is None


def test_file_and_env_override(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('comfyui_url = "http://gpu-box:8188/"\noutput_dir = "/tmp/out"\npreview_px = 256\n[http]\nport = 9000\ntoken = "abc"\n[qwen21]\nvae = "custom_vae.safetensors"\n')
    settings = config.load(path, env={"COMFYUI_URL": "http://other:1", "COMFY_MCP_HTTP_TOKEN": "envtoken", "COMFY_MCP_MAX_WAIT": "25"})
    assert settings.max_wait == 25.0
    assert settings.comfyui_url == "http://other:1"
    assert settings.output_dir == Path("/tmp/out")
    assert settings.preview_px == 256
    assert settings.http.port == 9000
    assert settings.http.token == "envtoken"
    assert settings.qwen21.vae == "custom_vae.safetensors"
    assert settings.model_files("qwen21") is settings.qwen21
    assert settings.default_image_model == "qwen21"
    assert settings.qwen21.diffusion_model == "qwen_image_2.1_int8_convrot.safetensors"
    assert settings.source == path


def test_explicit_missing_file_raises(tmp_path):
    try:
        config.load(tmp_path / "nope.toml")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_write_example_refuses_overwrite(tmp_path):
    path = tmp_path / "c.toml"
    config.write_example(path)
    assert "comfyui_url" in path.read_text()
    try:
        config.write_example(path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected FileExistsError")
    config.write_example(path, force=True)


def test_video_section(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('max_video_seconds = 10\n[minimax_h3]\nturbo_lora = "my_turbo.safetensors"\n')
    settings = config.load(path, env={})
    assert settings.max_video_seconds == 10.0 and settings.default_video_model == "minimax_h3"
    assert settings.minimax_h3.turbo_lora == "my_turbo.safetensors"
    assert settings.minimax_h3.video_vae == "minimax_h3_video_vae_fp16.safetensors"
    assert settings.model_files("minimax_h3") is settings.minimax_h3
    assert settings.free_models_after == ("video",)
    path.write_text('free_models_after = ["video", "image"]\n')
    assert config.load(path, env={}).free_models_after == ("video", "image")
    path.write_text("free_models_after = []\n")
    assert config.load(path, env={}).free_models_after == ()


def test_accel_section(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[minimax_h3]\nturbo_lora = "t.safetensors"\n[minimax_h3.accel]\nfirst_block_cache = false\nsol_attn = true\nfbc_threshold = 0.1\ndraft_width = 832\nfbc_node = "MyFBC"\nsol_strict = false\n')
    settings = config.load(path, env={})
    a = settings.minimax_h3_accel
    assert a.first_block_cache is False and a.sol_attn is True and a.fbc_threshold == 0.1 and a.draft_width == 832 and a.fbc_node == "MyFBC"
    assert a.sage_patch is True and a.final_steps == 20 and a.sol_strict is False and a.fbc_node == "MyFBC"
    assert config.load(env={}).minimax_h3_accel.fbc_node == "ApplyMiniMaxH3FirstBlockCache"
    assert settings.minimax_h3.turbo_lora == "t.safetensors"
