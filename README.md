# comfy-mcp

One-shot, private image generation from any MCP host, backed by a self-hosted [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running **Qwen Image 2.1**. Ask Claude Code, Codex or any other MCP client for an image and get back a preview plus a full-size PNG on your own machine.

- Plain parameters, no node graphs: `generate_image`, `edit_image` (one reference or 2–4 references composed), `server_status`, `list_models`, job tools for long waits. Tools take a `model` key; Qwen Image 2.1 (`qwen21`) is the first model, and new ones slot in without changing the tool surface.
- Private by construction: results stream back over the websocket and are **never written to the ComfyUI server's disk**; reference images go to ComfyUI's throwaway temp folder; the prompt's history entry is deleted as soon as the result arrives; this server keeps no logs, cache or transcripts.
- Always queues politely behind other ComfyUI jobs and reports its position.
- stdio transport for CLI hosts; optional Streamable HTTP with a bearer token for browser chat UIs.

Developed against a DGX Spark; works with any ComfyUI ≥ 0.36 that has the Qwen Image 2.1 files installed.

## Requirements

- ComfyUI 0.36+ reachable over HTTP with Qwen Image 2.1 installed: a diffusion model, its Qwen3-VL text encoder and the 2.1 VAE. The defaults expect Comfy-Org's INT8 names (`qwen_image_2.1_int8_convrot`, `qwen3vl_8b_int8_convrot`, `qwen_image_2.1_vae_bf16`); other precisions or renamed files go in the `[qwen21]` section of the config.
  The bundled `SaveImageWebsocket` custom node (ships with ComfyUI in `custom_nodes/websocket_image_save.py`) must be enabled.
- [uv](https://docs.astral.sh/uv/) on the client machine. Python 3.12+ is fetched automatically.

## Install

```bash
uv tool install git+https://github.com/mike-nott/comfy-mcp
comfy-mcp init      # writes ~/.config/comfy-mcp/config.toml — set comfyui_url to your server
comfy-mcp check     # prints resolved settings and the ComfyUI status
```

Or run it ad hoc without installing: `uvx --from git+https://github.com/mike-nott/comfy-mcp comfy-mcp`.

## Configure

`~/.config/comfy-mcp/config.toml` (or `--config PATH`, or `$COMFY_MCP_CONFIG`). Every key is optional:

```toml
comfyui_url = "http://127.0.0.1:8188"   # your ComfyUI; env: COMFYUI_URL
output_dir  = "~/Pictures/ComfyUI"         # env: COMFY_MCP_OUTPUT_DIR
preview_px  = 512        # long edge of the inline preview
max_wait    = 120        # seconds a call blocks before handing back a job id
default_steps = 25
default_cfg   = 1.0
max_pixels    = 2097152  # 2 MP canvas cap
job_ttl       = 1800     # seconds unfetched results stay in memory
default_image_model = "qwen21"   # used when a call omits `model`

[qwen21]                 # the three files as ComfyUI lists them; defaults are Comfy-Org's INT8 names
diffusion_model = "qwen_image_2.1_int8_convrot.safetensors"
text_encoder    = "qwen3vl_8b_int8_convrot.safetensors"
vae             = "qwen_image_2.1_vae_bf16.safetensors"

[http]                   # only for `comfy-mcp --http`
host  = "127.0.0.1"
port  = 8765
token = ""               # required, ≥16 chars; env: COMFY_MCP_HTTP_TOKEN
```

## Add to a host

**Claude Code**

```bash
claude mcp add comfy -- comfy-mcp
# or without installing:
claude mcp add comfy -- uvx --from git+https://github.com/mike-nott/comfy-mcp comfy-mcp
```

**Codex** — `~/.codex/config.toml`:

```toml
[mcp_servers.comfy]
command = "comfy-mcp"
```

**Any stdio host (OMP, Cursor, LM Studio, …)** — the generic JSON shape:

```json
{ "mcpServers": { "comfy": { "command": "comfy-mcp", "args": [] } } }
```

**Streamable HTTP clients (browser chat UIs such as [vllm-chat](https://github.com/mike-nott/vllm-chat))** — run the server once, on the ComfyUI box or your workstation:

```bash
COMFY_MCP_HTTP_TOKEN=$(openssl rand -hex 24) comfy-mcp --http   # or set [http] token in the config
```

then point the client at `http://<host>:8765/mcp` with `Authorization: Bearer <token>`. Bind `[http] host = "0.0.0.0"` to serve the LAN.

## Tools

| Tool | What it does |
|---|---|
| `generate_image(prompt, model?, negative?, width?, height?, aspect?, seed?, steps?, cfg?, count?, save?)` | Text to image. `model` defaults to the configured default (`qwen21`). `aspect` presets: `square`, `landscape`, `portrait`, `wide`, `tall`. |
| `edit_image(instruction, images[], model?, negative?, width?, height?, seed?, steps?, cfg?, count?, save?)` | Edit one image or compose from 2–4. `images` are local paths or base64. First image = base and canvas. |
| `server_status()` | Version, GPU memory, queue counts, whether a foreign job is running, our active jobs. |
| `list_models()` | The `model` keys this server can drive, each with its configured files and readiness (what's missing), plus everything ComfyUI has installed. |
| `job_status(job_id)`, `wait_for_job(job_id, timeout?)`, `fetch_result(job_id)`, `cancel_job(job_id)` | Used when a call outlasts `max_wait` and returns a job id instead of images. |

Every result carries a text part (saved paths, seed, size, timing) so text-only clients work, followed by one JPEG preview per image for clients that render images. Pass `save=false` to get the full PNG as base64 in the text and write nothing.

The `image_prompt_guidance` prompt (optionally per `model`) and the `comfyui://models` resource are also exposed.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how a call flows through ComfyUI.

## Privacy notes

What this server does:

- Output images travel over the websocket (`SaveImageWebsocket`); no `SaveImage`, nothing in ComfyUI's `output/`.
- Reference images are uploaded with `type=temp` under random names and referenced as `name [temp]`. Nothing goes to `input/`. As soon as the job finishes, each temp file is overwritten with a 1×1 blank PNG (ComfyUI has no delete endpoint); the empty files vanish when ComfyUI restarts.
- `POST /history {"delete": [prompt_id]}` runs as soon as the prompt finishes, success or failure.
- No log file, no cache, no persisted job list. `--debug` prints terse diagnostics to stderr only.

What it cannot control: ComfyUI's own console log records that a prompt ran and how long it took (not its text). Live sampler previews are sent to this client only and discarded.

## Development

```bash
uv sync
uv run pytest                                  # unit tests, no ComfyUI needed
COMFY_MCP_LIVE=1 uv run pytest tests/test_live.py -s   # real renders against your ComfyUI
uv run mcp dev src/comfy_mcp/dev.py          # MCP Inspector
```

## Roadmap

v2 adds MiniMax H3 video (`generate_video`) and MiniMax Music 3 (`generate_music`) through the same job tools. Further image models such as Krea2, Ideogram 4 and FLUX.2 Klein arrive as new `model` keys on the existing tools rather than new tools.

## License

MIT
