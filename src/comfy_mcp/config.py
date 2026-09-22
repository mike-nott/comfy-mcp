"""Settings: defaults < TOML file < environment variables.

Nothing here is ever written except by the explicit `init` command.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path("~/.config/comfy-mcp/config.toml").expanduser()
ENV_CONFIG = "COMFY_MCP_CONFIG"

EXAMPLE = """\
# comfy-mcp configuration. Every key is optional; these are the defaults.

# ComfyUI HTTP endpoint. Env override: COMFYUI_URL
comfyui_url = "http://127.0.0.1:8188"

# Where full-resolution results are written on this machine. Env override: COMFY_MCP_OUTPUT_DIR
output_dir = "~/Pictures/ComfyUI"

# Long edge of the inline JPEG preview returned to the calling model.
preview_px = 512

# Seconds a generate/edit call blocks before returning a job id for polling. Keep this below your
# host's MCP tool-call timeout (see README "Host timeouts"). Env override: COMFY_MCP_MAX_WAIT
max_wait = 120

# Sampling defaults for Qwen Image 2.1.
default_steps = 25
default_cfg = 1.0

# Largest canvas accepted (width x height). 2_097_152 = 2 MP.
max_pixels = 2097152

# Seconds finished-but-unfetched results stay in memory.
job_ttl = 1800

# Image model used when a call does not pass `model`. Currently only "qwen21" is implemented.
default_image_model = "qwen21"

# Video model used when generate_video does not pass `model`, and the longest clip accepted (H3 max 15 s).
default_video_model = "minimax_h3"
max_video_seconds = 15

# Qwen Image 2.1 files as ComfyUI lists them (see `comfy-mcp check` or the list_models tool).
# Defaults are Comfy-Org's INT8 repackaged names; change them if you use another precision or renamed files.
[qwen21]
diffusion_model = "qwen_image_2.1_int8_convrot.safetensors"
text_encoder = "qwen3vl_8b_int8_convrot.safetensors"
vae = "qwen_image_2.1_vae_bf16.safetensors"

# MiniMax H3 video files as ComfyUI lists them. Video also needs the ComfyUI-VideoHelperSuite node pack.
[minimax_h3]
diffusion_model = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
reference_model = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
text_encoder = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
video_vae = "minimax_h3_video_vae_fp16.safetensors"
audio_vae = "minimax_h3_audio_vae_fp32.safetensors"
turbo_lora = "minimax_h3_turbo_v4_step600_ema.safetensors"

# Streamable HTTP mode (`comfy-mcp --http`). Token is required. Env override: COMFY_MCP_HTTP_TOKEN
[http]
host = "127.0.0.1"
port = 8765
token = ""
"""


@dataclass
class HttpSettings:
    host: str = "127.0.0.1"
    port: int = 8765
    token: str = ""


@dataclass
class Qwen21Files:
    diffusion_model: str = "qwen_image_2.1_int8_convrot.safetensors"
    text_encoder: str = "qwen3vl_8b_int8_convrot.safetensors"
    vae: str = "qwen_image_2.1_vae_bf16.safetensors"


@dataclass
class MinimaxH3Files:
    diffusion_model: str = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    reference_model: str = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    text_encoder: str = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    video_vae: str = "minimax_h3_video_vae_fp16.safetensors"
    audio_vae: str = "minimax_h3_audio_vae_fp32.safetensors"
    turbo_lora: str = "minimax_h3_turbo_v4_step600_ema.safetensors"


@dataclass
class Settings:
    comfyui_url: str = "http://127.0.0.1:8188"
    output_dir: Path = Path("~/Pictures/ComfyUI")
    preview_px: int = 512
    max_wait: float = 120.0
    default_steps: int = 25
    default_cfg: float = 1.0
    max_pixels: int = 2_097_152
    job_ttl: float = 1800.0
    default_image_model: str = "qwen21"
    default_video_model: str = "minimax_h3"
    max_video_seconds: float = 15.0
    qwen21: Qwen21Files = field(default_factory=Qwen21Files)
    minimax_h3: MinimaxH3Files = field(default_factory=MinimaxH3Files)
    http: HttpSettings = field(default_factory=HttpSettings)
    debug: bool = False
    source: Path | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir).expanduser()
        self.comfyui_url = self.comfyui_url.rstrip("/")

    def model_files(self, key: str):
        """The configured file names for a recipe key, e.g. settings.model_files("qwen21")."""
        return getattr(self, key)


def resolve_path(explicit: str | Path | None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    if os.environ.get(ENV_CONFIG):
        return Path(os.environ[ENV_CONFIG]).expanduser()
    return DEFAULT_PATH


def load(explicit: str | Path | None = None, *, debug: bool = False, env: dict[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    path = resolve_path(explicit)
    data: dict = {}
    if path.is_file():
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    elif explicit:
        raise FileNotFoundError(f"Config file not found: {path}")

    http_data = data.get("http", {}) or {}
    qwen_data = data.get("qwen21", {}) or {}
    h3_data = data.get("minimax_h3", {}) or {}
    settings = Settings(
        comfyui_url=str(data.get("comfyui_url", Settings.comfyui_url)),
        output_dir=Path(str(data.get("output_dir", Settings.output_dir))),
        preview_px=int(data.get("preview_px", Settings.preview_px)),
        max_wait=float(data.get("max_wait", Settings.max_wait)),
        default_steps=int(data.get("default_steps", Settings.default_steps)),
        default_cfg=float(data.get("default_cfg", Settings.default_cfg)),
        max_pixels=int(data.get("max_pixels", Settings.max_pixels)),
        job_ttl=float(data.get("job_ttl", Settings.job_ttl)),
        default_image_model=str(data.get("default_image_model", Settings.default_image_model)),
        default_video_model=str(data.get("default_video_model", Settings.default_video_model)),
        max_video_seconds=float(data.get("max_video_seconds", Settings.max_video_seconds)),
        qwen21=Qwen21Files(
            diffusion_model=str(qwen_data.get("diffusion_model", Qwen21Files.diffusion_model)),
            text_encoder=str(qwen_data.get("text_encoder", Qwen21Files.text_encoder)),
            vae=str(qwen_data.get("vae", Qwen21Files.vae)),
        ),
        minimax_h3=MinimaxH3Files(**{k: str(h3_data.get(k, getattr(MinimaxH3Files, k))) for k in MinimaxH3Files.__dataclass_fields__}),
        http=HttpSettings(
            host=str(http_data.get("host", HttpSettings.host)),
            port=int(http_data.get("port", HttpSettings.port)),
            token=str(http_data.get("token", HttpSettings.token)),
        ),
        debug=debug,
        source=path if path.is_file() else None,
    )
    if env.get("COMFYUI_URL"):
        settings.comfyui_url = env["COMFYUI_URL"].rstrip("/")
    if env.get("COMFY_MCP_OUTPUT_DIR"):
        settings.output_dir = Path(env["COMFY_MCP_OUTPUT_DIR"]).expanduser()
    if env.get("COMFY_MCP_HTTP_TOKEN"):
        settings.http.token = env["COMFY_MCP_HTTP_TOKEN"]
    if env.get("COMFY_MCP_MAX_WAIT"):
        settings.max_wait = float(env["COMFY_MCP_MAX_WAIT"])
    return settings


def write_example(path: Path | None = None, *, force: bool = False) -> Path:
    path = path or DEFAULT_PATH
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(EXAMPLE)
    return path
