"""MCP surface: tools, one prompt and one resource, built around a settings object."""

from __future__ import annotations

import asyncio
import base64
import functools
import json
import secrets
import time
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image

from . import __version__
from .comfy import ComfyClient, ComfyError
from .config import Settings
from .images import dimensions, load_input, preview_jpeg, save_png
from .jobs import Job, JobRegistry
from .recipes import IMAGE_MODELS, image_model

INSTRUCTIONS = (
    "Private media generation on a self-hosted ComfyUI; call list_models to see which models are available. "
    "generate_image makes new images from text; edit_image changes or combines existing images given as file paths or base64. "
    "A call normally takes 20-60 seconds and blocks until the image is ready. If the server is busy the call may return a job_id "
    "instead; poll it with wait_for_job or fetch_result. Results include a small preview and the path of the saved full-size PNG. "
    "Nothing is stored on the ComfyUI server and this MCP keeps no logs."
)

MAX_SEED = 2**53 - 1

EXPECTED_ERRORS = (ComfyError, ValueError, KeyError, FileNotFoundError, OSError)


def friendly(func):
    """Turn anticipated failures into ToolError so the model sees the message, not a traceback."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ToolError:
            raise
        except EXPECTED_ERRORS as error:
            message = error.args[0] if error.args else error.__class__.__name__
            raise ToolError(str(message)) from error

    return wrapper


def _clamp_sampling(steps: int, cfg: float, count: int) -> None:
    if not 1 <= steps <= 60:
        raise ValueError("steps must be between 1 and 60")
    if not 0.0 <= cfg <= 20.0:
        raise ValueError("cfg must be between 0 and 20")
    if not 1 <= count <= 4:
        raise ValueError("count must be between 1 and 4")


def _seed(seed: int | None) -> int:
    if seed is None:
        return secrets.randbelow(MAX_SEED)
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_SEED}")
    return seed


def build_server(settings: Settings) -> MCPServer:
    client = ComfyClient(settings.comfyui_url, debug=settings.debug)
    registry = JobRegistry(settings.job_ttl)
    mcp = MCPServer(
        "comfy",
        title="ComfyUI",
        version=__version__,
        instructions=INSTRUCTIONS,
        log_level="DEBUG" if settings.debug else "ERROR",
    )

    # ---- readiness --------------------------------------------------------

    ready_until: dict[str, float] = {}

    async def inventory() -> tuple[list[str], list[str], list[str], list[str]]:
        return await asyncio.gather(
            client.loader_choices("UNETLoader", "unet_name"),
            client.loader_choices("CLIPLoader", "clip_name"),
            client.loader_choices("VAELoader", "vae_name"),
            client.loader_choices("LoraLoader", "lora_name"),
        )

    def missing_files(key: str, unets: list[str], clips: list[str], vaes: list[str]) -> list[str]:
        files = settings.model_files(key)
        return [
            f"{key} = {value!r}"
            for key, value, choices in (("diffusion_model", files.diffusion_model, unets), ("text_encoder", files.text_encoder, clips), ("vae", files.vae, vaes))
            if value not in choices
        ]

    async def ensure_ready(key: str) -> None:
        """Fail early with a readable message if the configured files are not on the server (cached 60 s)."""
        if time.monotonic() < ready_until.get(key, 0.0):
            return
        unets, clips, vaes, _ = await inventory()
        missing = missing_files(key, unets, clips, vaes)
        if missing:
            raise ComfyError(
                f"ComfyUI does not have the configured {image_model(key).NAME} file(s): "
                + "; ".join(missing)
                + f". Call list_models to see what is installed and set the [{key}] keys in the comfy-mcp config file."
            )
        ready_until[key] = time.monotonic() + 60.0

    def resolve_model(model: str | None):
        key = (model or settings.default_image_model).strip().lower()
        return key, image_model(key)

    # ---- job plumbing -----------------------------------------------------

    def start_job(kind: str, params: dict[str, Any], graph: dict, expected: int, save_node: str, temp_tokens: list[str] | None = None) -> Job:
        async def runner(job: Job) -> list[bytes]:
            job.started = time.monotonic()
            try:
                return await _run(job)
            finally:
                for token in temp_tokens or []:
                    await client.scrub_temp(token)

        async def _run(job: Job) -> list[bytes]:

            async def on_submitted(prompt_id: str) -> None:
                job.prompt_id = prompt_id
                job.cancel_hook = lambda: client.cancel(prompt_id)

            def on_progress(stage: str, data: dict[str, Any]) -> None:
                if stage == "queued":
                    position = data.get("position")
                    if position:
                        job.state, job.position = "queued", position
                        job.message = f"queued, {position} job(s) ahead"
                    else:
                        job.state, job.position, job.message = "running", 0, "running"
                elif stage == "running":
                    job.state, job.position, job.message = "running", 0, "running"
                elif stage == "progress":
                    job.state = "running"
                    job.progress = (int(data.get("value", 0)), int(data.get("max", 0)))
                    job.message = f"step {job.progress[0]}/{job.progress[1]}"
                elif stage == "image":
                    job.message = f"received image {data.get('index')}/{data.get('of')}"

            return await client.run(graph, save_node=save_node, expected=expected, on_progress=on_progress, on_submitted=on_submitted)

        return registry.create(kind, params, runner)

    async def report(ctx: Context, job: Job) -> None:
        try:
            if job.progress and job.state == "running":
                await ctx.report_progress(job.progress[0], job.progress[1], job.message)
            else:
                await ctx.report_progress(0, None, job.message)
        except Exception:  # noqa: BLE001 - progress is best effort
            pass

    async def await_job(ctx: Context, job: Job, timeout: float) -> bool:
        """Wait up to `timeout` seconds, relaying progress. True when the job has finished."""
        deadline = time.monotonic() + timeout
        last_message = None
        while job.task and not job.task.done():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            done, _ = await asyncio.wait({job.task}, timeout=min(1.0, remaining))
            if done:
                break
            if job.message != last_message:
                last_message = job.message
                await report(ctx, job)
        return True

    def finished_content(job: Job, save: bool) -> list[Any]:
        if job.state == "error":
            raise ComfyError(job.error or "generation failed")
        if job.state == "cancelled":
            raise ComfyError("the job was cancelled")
        if not job.results:
            raise ComfyError("the job finished without images")
        params = job.params
        width, height = dimensions(job.results[0])
        run_time = (job.finished or time.monotonic()) - (job.started or job.created)
        head = (
            f"{params['model']} · {len(job.results)} image{'s' if len(job.results) > 1 else ''} · {width}×{height} · seed {params['seed']}"
            f" · {params['steps']} steps · cfg {params['cfg']} · {run_time:.1f} s"
        )
        lines = [head]
        content: list[Any] = []
        for index, png in enumerate(job.results):
            if save:
                path = save_png(png, settings.output_dir, params["seed"], index, len(job.results))
                lines.append(f"saved: {path}")
            else:
                lines.append(f"image {index + 1} png_base64: {base64.b64encode(png).decode()}")
            content.append(Image(data=preview_jpeg(png, settings.preview_px), format="jpeg"))
        content.insert(0, "\n".join(lines))
        job.results = None if save else job.results
        if save:
            registry.forget(job.id)
        return content

    def pending_content(job: Job) -> str:
        data = job.public()
        data["next"] = "call wait_for_job(job_id) to block until done, or job_status(job_id) to peek; fetch_result(job_id) returns the images"
        return json.dumps(data, indent=2)

    # ---- tools ------------------------------------------------------------

    @mcp.tool(structured_output=False)
    @friendly
    async def generate_image(
        ctx: Context,
        prompt: str,
        model: str | None = None,
        negative: str = "",
        width: int | None = None,
        height: int | None = None,
        aspect: str | None = None,
        seed: int | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        count: int = 1,
        save: bool = True,
    ) -> list[Any]:
        """Generate new image(s) from a text prompt.

        model: which image model to use; omit for the configured default. Currently available: "qwen21"
        (Qwen Image 2.1). list_models shows what is installed and ready.
        Canvas: give width and height (multiples of 32, default 1024×1024) or an aspect preset:
        square, landscape (1344×768), portrait (768×1344), wide (1536×640), tall (640×1536).
        seed: omit for random; the seed used is always returned so a result can be reproduced.
        steps default 25, cfg default 1.0 (Qwen 2.1 works best at cfg 1.0; a negative prompt only
        matters when cfg > 1). count 1-4 images in one run.
        save=true writes full-size PNGs to the configured output folder and returns their paths plus
        a small preview of each; save=false returns base64 PNG in the text instead of writing files.
        Blocks until done (typically 20-60 s). If it returns JSON with a job_id, the server was busy:
        use wait_for_job(job_id).
        """
        steps = steps or settings.default_steps
        cfg = settings.default_cfg if cfg is None else cfg
        _clamp_sampling(steps, cfg, count)
        if not prompt.strip():
            raise ValueError("prompt is required")
        key, recipe = resolve_model(model)
        w, h = recipe.canvas(width, height, aspect=aspect, max_pixels=settings.max_pixels)
        await ensure_ready(key)
        chosen = _seed(seed)
        graph = recipe.text_graph(settings.model_files(key), prompt=prompt, negative=negative, width=w, height=h, seed=chosen, steps=steps, cfg=cfg, count=count)
        params = {"model": recipe.NAME, "model_key": key, "seed": chosen, "width": w, "height": h, "steps": steps, "cfg": cfg, "count": count}
        job = start_job("generate", params, graph, count, recipe.SAVE_NODE)
        if await await_job(ctx, job, settings.max_wait):
            return finished_content(job, save)
        return [pending_content(job)]

    @mcp.tool(structured_output=False)
    @friendly
    async def edit_image(
        ctx: Context,
        instruction: str,
        images: list[str],
        model: str | None = None,
        negative: str = "",
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        count: int = 1,
        save: bool = True,
    ) -> list[Any]:
        """Edit one image, or compose a new image from 2-4 reference images.

        model: which image model to use; omit for the configured default. Currently available: "qwen21".
        images: 1-4 local file paths (or data:/base64 strings). ORDER MATTERS: the first image is
        the base/subject and sets the output canvas (unless width/height are given); it is fitted and
        padded, never stretched. Later images are supporting references at ~1 MP.
        instruction: start with the operation ("Replace the background with…", "Put the person from
        <image1> into the scene from <image2>…"). Say what comes from each reference, what must stay
        recognizable and what must change. For placing a subject into a scene: subject FIRST, scene
        SECOND, with explicit <image1>/<image2> roles.
        Other parameters and the return value match generate_image.
        """
        steps = steps or settings.default_steps
        cfg = settings.default_cfg if cfg is None else cfg
        _clamp_sampling(steps, cfg, count)
        if not instruction.strip():
            raise ValueError("instruction is required")
        if not images:
            raise ValueError("at least one reference image is required")
        key, recipe = resolve_model(model)
        if len(images) > recipe.MAX_REFERENCES:
            raise ValueError(f"at most {recipe.MAX_REFERENCES} reference images for {recipe.NAME}")
        await ensure_ready(key)
        loaded = [await asyncio.to_thread(load_input, item) for item in images]
        first_w, first_h = loaded[0][1], loaded[0][2]
        w, h = recipe.canvas(width, height, aspect=None, max_pixels=settings.max_pixels, fallback=(first_w, first_h))
        references = []
        for png, rw, rh in loaded:
            token = await client.upload_temp(png)
            references.append(recipe.Reference(token, rw, rh))
        chosen = _seed(seed)
        graph = recipe.edit_graph(
            settings.model_files(key),
            instruction=instruction, negative=negative, references=references, width=w, height=h, seed=chosen, steps=steps, cfg=cfg, count=count
        )
        params = {"model": recipe.NAME, "model_key": key, "seed": chosen, "width": w, "height": h, "steps": steps, "cfg": cfg, "count": count, "references": len(references)}
        job = start_job("edit", params, graph, count, recipe.SAVE_NODE, temp_tokens=[ref.token for ref in references])
        if await await_job(ctx, job, settings.max_wait):
            return finished_content(job, save)
        return [pending_content(job)]

    @mcp.tool()
    @friendly
    async def server_status() -> dict[str, Any]:
        """Check the ComfyUI server: reachability, version, GPU memory, queue length and whether
        someone else's job is running (this MCP always queues behind it)."""
        status = await client.status()
        ours = {job.prompt_id for job in registry.active() if job.prompt_id}
        running = status.pop("running_prompt_ids", [])
        pending = status.pop("pending_prompt_ids", [])
        status["foreign_job_running"] = any(pid not in ours for pid in running)
        status["foreign_jobs_pending"] = sum(1 for pid in pending if pid not in ours)
        status["our_active_jobs"] = [job.public() for job in registry.active()]
        status["output_dir"] = str(settings.output_dir)
        return status

    @mcp.tool()
    @friendly
    async def list_models() -> dict[str, Any]:
        """List the image models this server can drive (keys for the `model` parameter) with their readiness,
        plus everything ComfyUI currently has installed."""
        unets, clips, vaes, loras = await inventory()
        image_models = {}
        for key, recipe in IMAGE_MODELS.items():
            files = settings.model_files(key)
            missing = missing_files(key, unets, clips, vaes)
            image_models[key] = {
                "name": recipe.NAME,
                "default": key == settings.default_image_model,
                "supports": ["generate_image", "edit_image"],
                "configured_files": {"diffusion_model": files.diffusion_model, "text_encoder": files.text_encoder, "vae": files.vae},
                "ready": not missing,
                "missing": missing,
                "hint": None if not missing else f"Set the [{key}] keys in the comfy-mcp config file to filenames from the installed lists.",
            }
        return {
            "image_models": image_models,
            "installed": {"diffusion_models": unets, "text_encoders": clips, "vaes": vaes, "loras": loras},
        }

    @mcp.tool()
    @friendly
    async def job_status(job_id: str) -> dict[str, Any]:
        """Peek at a job returned by generate_image/edit_image without waiting."""
        return registry.get(job_id).public()

    @mcp.tool(structured_output=False)
    @friendly
    async def wait_for_job(ctx: Context, job_id: str, timeout: float = 120.0, save: bool = True) -> list[Any]:
        """Block up to `timeout` seconds for a job to finish. Returns the images (preview + saved path)
        when done, or the job status JSON if still running."""
        job = registry.get(job_id)
        if await await_job(ctx, job, timeout):
            return finished_content(job, save)
        return [pending_content(job)]

    @mcp.tool(structured_output=False)
    @friendly
    async def fetch_result(job_id: str, save: bool = True) -> list[Any]:
        """Return the images of a finished job (preview + saved path). Fails if the job is not done yet."""
        job = registry.get(job_id)
        if job.state in ("queued", "running"):
            raise ComfyError(f"job {job_id} is still {job.state}: {job.message}")
        return finished_content(job, save)

    @mcp.tool()
    @friendly
    async def cancel_job(job_id: str) -> dict[str, Any]:
        """Cancel one of this server's own jobs: removes it from the ComfyUI queue or interrupts it if running."""
        job = await registry.cancel(job_id)
        return job.public()

    # ---- prompt + resource -----------------------------------------------

    @mcp.prompt(name="image_prompt_guidance", description="How to write prompts and edit instructions for the available image models")
    def image_prompt_guidance(model: str | None = None) -> str:
        if model:
            return image_model(model.strip().lower()).GUIDANCE
        return "\n\n".join(recipe.GUIDANCE for recipe in IMAGE_MODELS.values())

    @mcp.resource("comfyui://models", name="models", description="Live model inventory from ComfyUI", mime_type="application/json")
    async def models_resource() -> str:
        return json.dumps(await list_models(), indent=2)

    return mcp
