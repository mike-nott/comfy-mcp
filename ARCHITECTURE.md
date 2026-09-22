# Architecture

comfy-mcp is a thin MCP server in front of a ComfyUI instance you already run. It knows a handful of hard-coded graphs, exposes them as tools with plain parameters, and takes care not to leave anything behind on the ComfyUI host.

## Modules

| Module | Role |
|---|---|
| `cli.py` | Entry point. `comfy-mcp` (stdio), `comfy-mcp --http`, `comfy-mcp init`, `comfy-mcp check`. |
| `config.py` | Settings: defaults < TOML file < environment. Never writes except in `init`. |
| `server.py` | Builds the `MCPServer`: tools, the `image_prompt_guidance` prompt, the `comfyui://models` resource. |
| `comfy.py` | ComfyUI client: HTTP calls, websocket result streaming, temp uploads and scrubbing, history deletion. |
| `recipes/__init__.py` | Registries of image and video models keyed by the `model` parameter (`IMAGE_MODELS`, `VIDEO_MODELS`). |
| `recipes/qwen21.py` | Qwen Image 2.1 graphs (text-to-image, reference edit/compose), canvas rules, prompt guidance. |
| `recipes/minimax_h3.py` | MiniMax H3 graphs (t2v, i2v, r2v, refav; draft via turbo LoRA), frame grid, prompt guidance. |
| `images.py` | Input decoding (path, `data:` URL, base64), JPEG previews, output naming and saving. |
| `jobs.py` | In-memory job registry for calls that outlast `max_wait`. |
| `http.py` | Streamable HTTP transport behind a static bearer token. |

## One call, end to end

1. The tool resolves `model` (default from config) to a recipe module, checks once a minute that the recipe's configured files exist on the server, validates parameters and resolves the canvas (multiples of 32, capped by `max_pixels`). A seed is chosen if none was given.
2. For `edit_image`, each reference is decoded locally, re-encoded as PNG without metadata, and uploaded to ComfyUI with `type=temp` under a random name. The graph refers to it as `name [temp]`.
3. A fresh `client_id` is generated and a websocket is opened to `/ws?clientId=…` **before** the prompt is submitted, so every message and binary frame for this prompt is routed to us.
4. `POST /prompt` submits the graph. The output node is `SaveImageWebsocket`, which emits each finished image as a binary frame (8-byte header: event type, image format, then PNG bytes) instead of writing a file.
5. The websocket loop relays `status`/`progress` events as MCP progress notifications, reports the queue position while waiting behind other jobs, and collects PNG frames while the save node is executing. Sampler previews arrive on the same channel as JPEG and are ignored.
6. When the prompt finishes, `POST /history {"delete": [prompt_id]}` removes it from ComfyUI's in-memory history, and each temp reference is overwritten with a 1×1 blank PNG (ComfyUI has no delete endpoint; the empty files disappear on its next restart).
7. Full-size PNGs are written to `output_dir` on the client machine (or returned as base64 when `save=false`). The tool result is one text block (paths, seed, size, timing) followed by one JPEG preview per image.

If the call outlasts `max_wait`, the job keeps running in the background and the tool returns JSON with a `job_id`. `wait_for_job`, `job_status`, `fetch_result` and `cancel_job` operate on that registry. Results live in memory until fetched or until `job_ttl` expires.

## Video

`generate_video` follows the same steps with two differences. It never blocks by default (renders take minutes), so the caller gets a `job_id` at once. And ComfyUI has no websocket save node for video, so the graph ends in VideoHelperSuite's `VHS_VideoCombine` with `save_output=false`: the MP4 (with the model's audio muxed in) lands in ComfyUI's temp folder, which is inside the container and cleared on restart. When the websocket reports the prompt finished, the client reads the file's name from `/history/{prompt_id}`, downloads it through `/view`, then overwrites every file VideoHelperSuite left (the reported MP4, the video-only intermediate and the first-frame PNG) with a 1×1 blank, and deletes the history entry. A parallel `ImageFromBatch` → `SaveImageWebsocket` branch streams the first frame as the preview so the client needs no video decoder. After the MP4 is saved locally its bytes are dropped from memory; the saved path and poster stay for the job's TTL so a second delivery works.

## Queueing

The server never interrupts or reorders anything. `server_status` reports how many prompts are running or pending and whether any of them belong to another client. Our own prompt's position in `queue_pending` is surfaced in progress messages and in the job status.

## Transports

- **stdio** is the default and what CLI hosts (Claude Code, Codex and others) expect. Nothing is written to stdout except protocol messages; `--debug` prints terse diagnostics to stderr.
- **Streamable HTTP** wraps the SDK's Starlette app in a bearer-token check. Sessions are stateful so clients that track `Mcp-Session-Id` work. DNS-rebinding protection is disabled because the server is meant to be bound to a LAN address behind the token.

## Adding a model

Tools are grouped by media type, not by model, so the tool list stays short for small calling models. An image model is a module under `recipes/` exposing the same interface as `qwen21.py` (`KEY`, `NAME`, `SAVE_NODE`, `canvas`, `text_graph`, `edit_graph`, `GUIDANCE`), registered in `IMAGE_MODELS`, with a matching `[key]` section in the config for its file names. It then appears in `list_models` and is selectable through the `model` parameter with no tool changes.

Video and audio are different media, so they get their own tools (`generate_video`, `generate_music`) with their own `model` keys and registries. A video recipe exposes `KEY`, `NAME`, `MODES`, `VIDEO_NODE`, `POSTER_NODE`, `REQUIRED_NODES`, `frames()`, `canvas()`, `graph()` and `GUIDANCE`; the tool checks `REQUIRED_NODES` against `/object_info` before submitting so a missing node pack produces a plain message.
