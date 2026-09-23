# comfy-mcp

An MCP server that turns a self-hosted [ComfyUI](https://github.com/comfyanonymous/ComfyUI) into one-shot, private media generation for any MCP host. Ask Claude Code, Codex or any other MCP client for an image and get back a preview plus a full-size file on your own machine, with nothing left behind on the server.

- Plain parameters, no node graphs: `generate_image`, `edit_image` (one reference or 2–4 references composed), `generate_video` (text, first/last frame, references, references + audio), `server_status`, `list_models`, job tools for long renders.
- Model-agnostic tool surface: every generation tool takes a `model` key, and each model is a small self-contained recipe with its own config section. Bundled: **Qwen Image 2.1** (`qwen21`) for images and **MiniMax H3** (`minimax_h3`) for video with sound. Music is next (see roadmap).
- Private by construction: images stream back over the websocket and are **never written to the ComfyUI server's disk**; video goes through ComfyUI's throwaway temp folder and is blanked the moment it has been fetched; reference images and audio use the same temp folder; the prompt's history entry is deleted as soon as the result arrives; this server keeps no logs, cache or transcripts.
- Always queues politely behind other ComfyUI jobs and reports its position.
- stdio transport for CLI hosts; optional Streamable HTTP with a bearer token for browser chat UIs.

Developed against a DGX Spark; works with any ComfyUI ≥ 0.36 that has the files for at least one bundled model installed.

## Requirements

- ComfyUI 0.36+ reachable over HTTP, with the model files for at least one bundled model. For `qwen21` that is a Qwen Image 2.1 diffusion model, its Qwen3-VL text encoder and the 2.1 VAE; the defaults expect Comfy-Org's INT8 names (`qwen_image_2.1_int8_convrot`, `qwen3vl_8b_int8_convrot`, `qwen_image_2.1_vae_bf16`), and other precisions or renamed files go in the `[qwen21]` config section. `list_models` tells you what is ready.
- For video: the MiniMax H3 files (`[minimax_h3]` section: the fl2va and ref2va diffusion models, the Qwen3-VL 32B encoder, the video and audio VAEs and the turbo LoRA) plus the [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) custom node pack, whose Video Combine node is the only built-in-or-common node that can write a video to ComfyUI's temp folder instead of its output folder.
  The bundled `SaveImageWebsocket` custom node (ships with ComfyUI in `custom_nodes/websocket_image_save.py`) must be enabled.
- [uv](https://docs.astral.sh/uv/) on the client machine. Python 3.12+ is fetched automatically.

## Install

```bash
uv tool install git+https://github.com/mike-nott/comfy-mcp
comfy-mcp init      # writes ~/.config/comfy-mcp/config.toml — set comfyui_url to your server
comfy-mcp check     # prints resolved settings and the ComfyUI status
```

Or run it ad hoc without installing: `uvx --from git+https://github.com/mike-nott/comfy-mcp comfy-mcp`.

Upgrade with `uv tool upgrade comfy-mcp`. Every release bumps the version, which is what makes that command fetch the new code.

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
save_policy   = "default"  # "default" | "never" | "always"; env: COMFY_MCP_SAVE_POLICY
default_image_model = "qwen21"   # used when a call omits `model`
free_models_after = ["video"]    # unload ComfyUI's models after these job kinds; images stay warm

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
claude mcp add --scope user comfy-mcp -- comfy-mcp
# or without installing:
claude mcp add --scope user comfy-mcp -- uvx --from git+https://github.com/mike-nott/comfy-mcp comfy-mcp
```

**Codex** — `~/.codex/config.toml`:

```toml
[mcp_servers.comfy-mcp]
command = "comfy-mcp"
```

**Any stdio host (OMP, Cursor, LM Studio, …)** — the generic JSON shape:

```json
{ "mcpServers": { "comfy-mcp": { "command": "comfy-mcp", "args": [] } } }
```

**Streamable HTTP clients (browser chat UIs such as [vllm-chat](https://github.com/mike-nott/vllm-chat))** — run the server once, on the ComfyUI box or your workstation:

```bash
COMFY_MCP_HTTP_TOKEN=$(openssl rand -hex 24) comfy-mcp --http   # or set [http] token in the config
```

then point the client at `http://<host>:8765/mcp` with `Authorization: Bearer <token>`. Bind `[http] host = "0.0.0.0"` to serve the LAN.

**Delivering full-size files to a remote user.** When comfy-mcp runs on a server for a browser chat UI, `output_dir` is on the wrong machine. Set `save_policy = "never"` on that instance: nothing is ever written to disk, whatever `save` the model passes, and each full-size file is returned as a line of its own:

```
download: comfy://result/<handle>.png
```

The client's backend fetches it with the same bearer token as `/mcp`:

```
GET http://<host>:8765/dl/<handle>.png
Authorization: Bearer <token>
```

The response streams from memory with the right `Content-Type`, `Content-Disposition: attachment; filename="<timestamp>-<seed>.<ext>"` and `Cache-Control: no-store`. A handle stays fetchable until 60 s after its first complete download (a retry window) or until `job_ttl`, whichever comes first; `HEAD` and aborted transfers do not use it up. Fetching the same job again (`fetch_result` after `wait_for_job`) returns the same handle, or a note that the file was already downloaded, plus the preview. The client should strip `download:` lines before passing tool results to the model. In HTTP mode `save=false` under the default policy also returns handles instead of base64.

## Models

| Key | Model | Tools | Status |
|---|---|---|---|
| `qwen21` | Qwen Image 2.1 | `generate_image`, `edit_image` | bundled |
| `minimax_h3` | MiniMax H3 video (with sound) | `generate_video` | bundled; needs VideoHelperSuite |
| — | MiniMax Music 3 | `generate_music` | planned |
| — | Krea2, Ideogram 4, FLUX.2 Klein | `generate_image`, `edit_image` | candidates |

Adding a model is a recipe module plus a config section; see [ARCHITECTURE.md](ARCHITECTURE.md).

## Speed (MiniMax H3)

Video is compute-bound in the sampler, so the levers are the attention kernel, cross-step caching and how much you render. The `[minimax_h3.accel]` config section controls what comfy-mcp adds to the graph, in the order the packs require (Sage → Sol-Attn → fused modulation → chunked feed-forward → Spectrum or FirstBlockCache, with the patched model feeding both scheduler and guider):

- `first_block_cache` (default on, finals only): needs the [FirstBlockCache](https://github.com/duckyshell/ComfyUI-MiniMaxH3-FirstBlockCache) pack; settings default to the 20-step-safe values.
- `sol_attn` (default on): sparse attention via the [Sol-Attn](https://github.com/Saganaki22/ComfyUI-sol-attn) pack's H3 patch, run in strict mode so a fallback fails loudly instead of silently rendering dense.
- `sage_patch` (default on): node-scoped SageAttention through KJNodes, applied before Sol-Attn. Leave ComfyUI's global `--use-sage-attention` flag off with H3.
- `fused_modulation` and `chunk_feed_forward` (default off): kernel-fusion nodes from the Sol-Attn pack; the modulation patch is incompatible with ComfyUI 0.36's H3 block.
- `spectrum` (default off): step forecasting as an alternative to FirstBlockCache; the two must not be combined.
- Drafts render at `draft_width`×`draft_height` and are upscaled with a small SPAN model (`upscale_model` in `models/upscale_models`) back to the final canvas; caching is not applied to drafts because turbo schedules have little to reuse.
- `steps` on `generate_video` overrides the step count; `final_steps` / `draft_steps` set the defaults.

Node class names (`fbc_node`, `sol_node`, …) are overridable, and any option a node does not declare is dropped with a note in the result rather than failing the render. `list_models` reports which acceleration nodes are present.

Measured on a DGX Spark (5 s clip, 1280×704, ComfyUI 0.36, INT8 ConvRot files, seeds varied): plain attention 887 s; FirstBlockCache 683 s; plus Sol-Attn 496 s; plus the Sage patch 479 s. The draft profile with Sol-Attn renders in 197 s.

## Host timeouts

An image render takes 20–60 s, longer when the GPU is shared. Video is much slower (on a DGX Spark sharing memory with a 27B LLM, a 5 s MiniMax H3 clip took about 6 minutes as an 8-step draft and 12 minutes at the final 20 steps) which is why `generate_video` never blocks: it returns a `job_id` and `wait_for_job` does the waiting in chunks of whatever your host allows. For images, every MCP host applies its own timeout to a tool call, and if that is shorter than the render the host reports an unknown outcome even though the job completes and the file is saved. Two ways to avoid it:

- **Raise the host's timeout** so calls finish inline. OMP: add `"timeout": 180000` (milliseconds) to the server entry in `mcp.json`. Claude Code and Codex have their own settings for MCP tool timeouts; see their documentation.
- **Or lower comfy-mcp's wait** below the host's timeout, per host, with `COMFY_MCP_MAX_WAIT` in the server entry's `env`. The call then returns a `job_id` in time and the model finishes with `wait_for_job`.

Either way nothing is lost: a job that was already delivered to a caller that gave up can be fetched again with `fetch_result`, and `server_status` lists active jobs.

## Tools

| Tool | What it does |
|---|---|
| `generate_image(prompt, model?, negative?, width?, height?, aspect?, seed?, steps?, cfg?, count?, save?)` | Text to image. `model` defaults to the configured default (`qwen21`). `aspect` presets: `square`, `landscape`, `portrait`, `wide`, `tall`. |
| `edit_image(instruction, images[], model?, negative?, width?, height?, seed?, steps?, cfg?, count?, save?)` | Edit one image or compose from 2–4. `images` are local paths or base64. First image = base and canvas. |
| `generate_video(prompt, model?, mode?, images[]?, audio?, seconds?, orientation?, draft?, seed?, save?, wait?)` | 5–15 s clip with sound. `mode`: `t2v`, `i2v` (first frame, optional last), `r2v` (1–3 references), `refav` (references + audio). Async: returns a `job_id`; finish with `wait_for_job(job_id, timeout=600)`. `draft=true` is an 8-step turbo preview. |
| `server_status()` | Version, GPU memory, queue counts, whether a foreign job is running, our active jobs. |
| `list_models()` | The `model` keys this server can drive, each with its configured files and readiness (what's missing), plus everything ComfyUI has installed. |
| `job_status(job_id)`, `wait_for_job(job_id, timeout?)`, `fetch_result(job_id)`, `cancel_job(job_id)` | Used when a call outlasts `max_wait` and returns a job id instead of images. |

Every result carries a text part (saved paths, seed, size, timing) so text-only clients work, followed by a JPEG preview (each image, or the first frame of a video) for clients that render images. Pass `save=false` to get the full file as base64 in the text and write nothing.

The `image_prompt_guidance` and `video_prompt_guidance` prompts (optionally per `model`) and the `comfyui://models` resource are also exposed.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how a call flows through ComfyUI.

## Privacy notes

What this server does:

- Output images travel over the websocket (`SaveImageWebsocket`); no `SaveImage`, nothing in ComfyUI's `output/`.
- After a video job (configurable with `free_models_after`) the server asks ComfyUI to unload its models, so the multi-GB video encoder and model do not stay resident on a box shared with other services. Image models stay warm for fast iteration.
- Video is written by VideoHelperSuite to ComfyUI's `temp/` folder (never `output/`), fetched once, then every file it left (the MP4, the video-only intermediate and the first-frame PNG) is overwritten with a 1×1 blank. The first-frame preview comes over the websocket like an image. Video metadata embedding is off, so the prompt is not stored in the file.
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

Next: MiniMax Music 3 (`generate_music`) through the same job tools, then named multiple ComfyUI servers with a `server` key per model. Further image models arrive as new `model` keys on the existing tools rather than new tools.

## License

MIT
