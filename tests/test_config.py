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
    settings = config.load(path, env={"COMFYUI_URL": "http://other:1", "COMFY_MCP_HTTP_TOKEN": "envtoken"})
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
