"""Minimal ComfyUI client: submit a graph, stream results over the websocket, clean up.

Privacy: outputs arrive as websocket frames (never written on the server), reference
uploads go to ComfyUI's temp folder, and the history entry is deleted once the prompt
has finished. This module never logs; `debug` writes terse lines to stderr only.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import struct
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
import websockets

EVENT_PREVIEW_IMAGE = 1
FORMAT_JPEG = 1
FORMAT_PNG = 2

ProgressFn = Callable[[str, dict[str, Any]], Awaitable[None] | None]

MAX_FILE_BYTES = 512 * 1024 * 1024
HISTORY_OUTPUT_KEYS = ("gifs", "video", "videos", "audio", "images")


@dataclass
class RunResult:
    """What a prompt produced: PNGs streamed over the websocket, and/or files fetched from history."""

    images: list[bytes]
    files: list[tuple[str, bytes]]  # (filename as ComfyUI named it, bytes)

# A 1x1 opaque black PNG used to overwrite uploaded references once a job is done.
BLANK_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108020000009077053d"
    "0000000c4944415408d763f8cfc0000003010100c9fe92ef0000000049454e44ae426082"
)


class ComfyError(RuntimeError):
    pass


def temp_siblings(name: str) -> list[str]:
    """Every temp file VideoHelperSuite leaves for one output: the reported file (e.g. `x_00001-audio.mp4`),
    the video-only intermediate (`x_00001.mp4`) and the first-frame PNG (`x_00001.png`)."""
    stem, dot, ext = name.rpartition(".")
    if not dot:
        return [name]
    base = stem[: -len("-audio")] if stem.endswith("-audio") else stem
    names = [name, f"{base}.{ext}", f"{base}.png"]
    return list(dict.fromkeys(names))


@dataclass(frozen=True)
class Frame:
    event: int
    fmt: int
    payload: bytes

    @classmethod
    def parse(cls, data: bytes) -> Frame:
        if len(data) < 8:
            raise ComfyError("Short binary websocket frame")
        event, fmt = struct.unpack(">II", data[:8])
        return cls(event, fmt, data[8:])

    @property
    def is_png_output(self) -> bool:
        return self.event == EVENT_PREVIEW_IMAGE and self.fmt == FORMAT_PNG


class ComfyClient:
    def __init__(self, base_url: str, *, debug: bool = False, idle_timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.debug = debug
        self.idle_timeout = idle_timeout
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(30.0, read=120.0))

    async def aclose(self) -> None:
        await self._http.aclose()

    def _log(self, *parts: Any) -> None:
        if self.debug:
            print("[comfy-mcp]", *parts, file=sys.stderr, flush=True)

    @property
    def ws_url(self) -> str:
        scheme = "wss" if self.base_url.startswith("https") else "ws"
        return scheme + "://" + self.base_url.split("://", 1)[1] + "/ws"

    # ---- plain HTTP -------------------------------------------------------

    async def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = await self._http.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise ComfyError(f"ComfyUI unreachable at {self.base_url}: {error.__class__.__name__}") from error
        if response.status_code >= 400:
            detail = response.text[:500]
            raise ComfyError(f"ComfyUI {method} {path} failed ({response.status_code}): {detail}")
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return response.text

    async def system_stats(self) -> dict:
        return await self._json("GET", "/system_stats")

    async def queue(self) -> dict:
        return await self._json("GET", "/queue")

    async def object_info(self, node: str) -> dict:
        data = await self._json("GET", f"/object_info/{node}")
        return data.get(node, {}) if isinstance(data, dict) else {}

    async def loader_choices(self, node: str, field: str) -> list[str]:
        info = await self.object_info(node)
        try:
            return list(info["input"]["required"][field][0])
        except (KeyError, IndexError, TypeError):
            return []

    async def status(self) -> dict[str, Any]:
        stats, queue = await asyncio.gather(self.system_stats(), self.queue())
        running = queue.get("queue_running", [])
        pending = queue.get("queue_pending", [])
        devices = stats.get("devices", [])
        gpu = devices[0] if devices else {}
        return {
            "comfyui_version": stats.get("system", {}).get("comfyui_version"),
            "python": stats.get("system", {}).get("python_version", "").split()[0] if stats.get("system") else None,
            "gpu": gpu.get("name"),
            "vram_free_gb": round(gpu.get("vram_free", 0) / 2**30, 1) if gpu else None,
            "vram_total_gb": round(gpu.get("vram_total", 0) / 2**30, 1) if gpu else None,
            "queue_running": len(running),
            "queue_pending": len(pending),
            "running_prompt_ids": [row[1] for row in running if len(row) > 1],
            "pending_prompt_ids": [row[1] for row in pending if len(row) > 1],
        }

    async def queue_position(self, prompt_id: str) -> int | None:
        """0 = running now, n = n jobs ahead; None = not in the queue."""
        queue = await self.queue()
        for row in queue.get("queue_running", []):
            if len(row) > 1 and row[1] == prompt_id:
                return 0
        for index, row in enumerate(queue.get("queue_pending", [])):
            if len(row) > 1 and row[1] == prompt_id:
                return index + 1
        return None

    async def upload_temp(self, data: bytes, suffix: str = ".png", mime: str = "image/png") -> str:
        """Upload a file into ComfyUI's temp folder under a random name. Returns the `name [temp]` token
        that LoadImage / LoadAudio accept."""
        name = f"mcp-{secrets.token_hex(8)}{suffix}"
        result = await self._json(
            "POST",
            "/upload/image",
            data={"type": "temp", "subfolder": "", "overwrite": "true"},
            files={"image": (name, data, mime)},
        )
        if not isinstance(result, dict) or result.get("type") != "temp" or not result.get("name"):
            raise ComfyError("ComfyUI did not accept the upload into its temp folder")
        return f"{result['name']} [temp]"

    async def scrub_temp(self, token: str, subfolder: str = "") -> None:
        """Overwrite a temp file (an uploaded reference or a fetched output) with a 1x1 blank PNG.

        ComfyUI has no delete endpoint; its temp folder is only cleared on restart.
        Re-uploading under the same name with overwrite=true replaces the bytes.
        Accepts a `name [temp]` token or a bare filename.
        """
        name = token.removesuffix(" [temp]")
        try:
            await self._json(
                "POST",
                "/upload/image",
                data={"type": "temp", "subfolder": subfolder, "overwrite": "true"},
                files={"image": (name, BLANK_PNG, "image/png")},
            )
        except ComfyError as error:
            self._log("temp scrub failed:", error)

    async def fetch_view(self, filename: str, subfolder: str, kind: str) -> bytes:
        """Download one file through /view (streamed, capped at MAX_FILE_BYTES)."""
        data = bytearray()
        try:
            async with self._http.stream("GET", "/view", params={"filename": filename, "subfolder": subfolder, "type": kind}) as response:
                if response.status_code != 200:
                    raise ComfyError(f"ComfyUI /view returned {response.status_code} for {filename}")
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > MAX_FILE_BYTES:
                        raise ComfyError(f"{filename} exceeds the {MAX_FILE_BYTES // 2**20} MB transfer limit")
        except httpx.HTTPError as error:
            raise ComfyError(f"download of {filename} failed: {error.__class__.__name__}") from error
        return bytes(data)

    async def history_outputs(self, prompt_id: str, node: str, attempts: int = 20) -> list[dict]:
        """Descriptors ({filename, subfolder, type}) a node wrote, from /history. Polls briefly because
        ComfyUI records history a moment after it reports execution finished."""
        for attempt in range(attempts):
            entry = await self._json("GET", f"/history/{prompt_id}")
            outputs = (entry or {}).get(prompt_id, {}).get("outputs", {}) if isinstance(entry, dict) else {}
            found = [d for key in HISTORY_OUTPUT_KEYS for d in outputs.get(node, {}).get(key, []) if isinstance(d, dict) and d.get("filename")]
            if found:
                return found
            status = (entry or {}).get(prompt_id, {}).get("status", {}) if isinstance(entry, dict) else {}
            if status.get("status_str") == "error":
                raise ComfyError("ComfyUI recorded an execution error for this prompt")
            await asyncio.sleep(0.25 * (attempt + 1))
        raise ComfyError(f"ComfyUI history has no output for node {node}")

    async def delete_history(self, prompt_id: str) -> None:
        try:
            await self._json("POST", "/history", json={"delete": [prompt_id]})
        except ComfyError as error:
            self._log("history delete failed:", error)

    async def cancel(self, prompt_id: str) -> str:
        position = await self.queue_position(prompt_id)
        if position is None:
            return "not queued"
        if position == 0:
            await self._json("POST", "/interrupt")
            return "interrupted"
        await self._json("POST", "/queue", json={"delete": [prompt_id]})
        return "removed from queue"

    # ---- run a graph ------------------------------------------------------

    async def run(
        self,
        graph: dict,
        *,
        save_node: str,
        expected: int,
        on_progress: ProgressFn | None = None,
        on_submitted: Callable[[str], Awaitable[None] | None] | None = None,
        history_node: str | None = None,
    ) -> RunResult:
        """Submit `graph`. Returns the PNGs `save_node` streamed over the websocket (exactly `expected`
        of them, or none if `save_node` is None) and, when `history_node` is given, the files that node
        wrote to ComfyUI's temp folder, fetched through /view and then blanked on the server."""
        client_id = uuid.uuid4().hex
        prompt_id: str | None = None

        async def emit(stage: str, **data: Any) -> None:
            if on_progress:
                result = on_progress(stage, data)
                if asyncio.iscoroutine(result):
                    await result

        try:
            async with websockets.connect(f"{self.ws_url}?clientId={client_id}", max_size=None, ping_interval=20, ping_timeout=60) as socket:
                submitted = await self._json("POST", "/prompt", json={"prompt": graph, "client_id": client_id})
                prompt_id = submitted.get("prompt_id") if isinstance(submitted, dict) else None
                if not prompt_id:
                    raise ComfyError(f"ComfyUI did not return a prompt id: {str(submitted)[:300]}")
                if on_submitted:
                    result = on_submitted(prompt_id)
                    if asyncio.iscoroutine(result):
                        await result
                position = await self.queue_position(prompt_id)
                await emit("queued", prompt_id=prompt_id, position=position)
                images = await self._collect(socket, prompt_id, save_node, expected, emit, wait_for_finish=history_node is not None)
                files: list[tuple[str, bytes]] = []
                if history_node:
                    await emit("fetching", prompt_id=prompt_id)
                    for descriptor in await self.history_outputs(prompt_id, history_node):
                        name, sub, kind = descriptor["filename"], descriptor.get("subfolder", ""), descriptor.get("type", "temp")
                        files.append((name, await self.fetch_view(name, sub, kind)))
                        if kind == "temp":
                            for sibling in temp_siblings(name):
                                await self.scrub_temp(sibling, sub)
                return RunResult(images=images, files=files)
        except websockets.exceptions.WebSocketException as error:
            raise ComfyError(f"Websocket to ComfyUI failed: {error}") from error
        except OSError as error:
            raise ComfyError(f"Could not connect to ComfyUI websocket at {self.ws_url}: {error}") from error
        finally:
            if prompt_id:
                await self.delete_history(prompt_id)

    async def _collect(
        self, socket: Any, prompt_id: str, save_node: str | None, expected: int, emit: Callable[..., Awaitable[None]], wait_for_finish: bool = False
    ) -> list[bytes]:
        images: list[bytes] = []
        current_node: str | None = None
        started = time.monotonic()
        last_status_check = 0.0
        while True:
            try:
                message = await asyncio.wait_for(socket.recv(), timeout=self.idle_timeout)
            except asyncio.TimeoutError:
                position = await self.queue_position(prompt_id)
                if position is None:
                    raise ComfyError("ComfyUI stopped reporting on this prompt and it is no longer queued")
                await emit("queued" if position else "running", prompt_id=prompt_id, position=position)
                continue

            if isinstance(message, bytes):
                frame = Frame.parse(message)
                if frame.is_png_output and save_node is not None and current_node == save_node:
                    images.append(frame.payload)
                    await emit("image", index=len(images), of=expected)
                    if len(images) >= expected and not wait_for_finish:
                        return images
                continue

            try:
                event = json.loads(message)
            except ValueError:
                continue
            kind, data = event.get("type"), event.get("data") or {}
            if kind == "status":
                now = time.monotonic()
                if current_node is None and now - last_status_check > 2.0:
                    last_status_check = now
                    position = await self.queue_position(prompt_id)
                    if position:
                        await emit("queued", prompt_id=prompt_id, position=position)
                continue
            if data.get("prompt_id") != prompt_id:
                continue
            if kind == "execution_start":
                await emit("running", prompt_id=prompt_id, position=0)
            elif kind == "executing":
                current_node = data.get("node")
                if current_node is None:
                    break
            elif kind == "progress":
                await emit("progress", value=data.get("value", 0), max=data.get("max", 0), node=data.get("node"))
            elif kind == "execution_success":
                break
            elif kind == "execution_error":
                node = data.get("node_type") or data.get("node_id")
                detail = data.get("exception_message") or "unknown error"
                raise ComfyError(f"ComfyUI execution error in {node}: {detail}")
            elif kind == "execution_interrupted":
                raise ComfyError("ComfyUI execution was interrupted")
            elif kind == "execution_cached":
                continue
        if len(images) < expected:
            self._log(f"finished after {time.monotonic() - started:.1f}s with {len(images)}/{expected} images")
            raise ComfyError(f"ComfyUI finished but only {len(images)} of {expected} images arrived over the websocket")
        return images
