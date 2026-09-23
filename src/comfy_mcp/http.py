"""Streamable HTTP transport behind a static bearer token."""

from __future__ import annotations

import asyncio
import hmac
from typing import Any

import uvicorn
from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import Settings
from .downloads import DownloadStore

CHUNK = 1024 * 1024


class BearerToken:
    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.token = token.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        header = dict(scope.get("headers") or {}).get(b"authorization", b"")
        supplied = header[7:] if header[:7].lower() == b"bearer " else b""
        if not supplied or not hmac.compare_digest(supplied, self.token):
            await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"text/plain"), (b"www-authenticate", b"Bearer")]})
            await send({"type": "http.response.body", "body": b"unauthorized"})
            return
        await self.app(scope, receive, send)


class Downloads:
    """GET/HEAD /dl/<handle>: stream a result from memory. Sits behind BearerToken, so it needs the
    same Authorization header as /mcp. Only a fully sent GET starts the handle's short grace window
    (after which the bytes are dropped); HEAD and detected aborts leave it untouched."""

    def __init__(self, app: ASGIApp, store: DownloadStore) -> None:
        self.app = app
        self.store = store

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith("/dl/"):
            await self.app(scope, receive, send)
            return
        method = scope.get("method", "GET")
        handle = scope["path"][len("/dl/"):]
        if method not in ("GET", "HEAD"):
            await _plain(send, 405, b"method not allowed")
            return
        entry = self.store.get(handle)
        if entry is None:
            await _plain(send, 404, b"not found, expired or already downloaded")
            return
        data = entry.data
        headers = [
            (b"content-type", entry.mime.encode()),
            (b"content-length", str(len(data)).encode()),
            (b"content-disposition", f'attachment; filename="{entry.filename}"'.encode()),
            (b"cache-control", b"no-store"),
        ]
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        if method == "HEAD":
            await send({"type": "http.response.body", "body": b""})
            return
        aborted = False

        async def watch() -> None:
            nonlocal aborted
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    aborted = True
                    return

        watcher = asyncio.create_task(watch())
        try:
            for start in range(0, len(data), CHUNK):
                await asyncio.sleep(0)  # let the disconnect watcher run; send() may not suspend
                if aborted:
                    return
                await send({"type": "http.response.body", "body": data[start : start + CHUNK], "more_body": True})
            await asyncio.sleep(0)
            if aborted:
                return
            # Check before the final message: once the response is complete, uvicorn answers the
            # watcher's receive() with http.disconnect, which would look like an abort.
            await send({"type": "http.response.body", "body": b""})
            self.store.complete(handle)
        except OSError:
            return
        finally:
            watcher.cancel()


async def _plain(send: Send, status: int, body: bytes) -> None:
    await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"text/plain")]})
    await send({"type": "http.response.body", "body": body})


def build_app(mcp: MCPServer, settings: Settings) -> ASGIApp:
    app = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        host=settings.http.host,
    )
    store = getattr(mcp, "comfy_downloads", None)
    if store is not None:
        app = Downloads(app, store)
    return BearerToken(app, settings.http.token)


def serve(mcp: MCPServer, settings: Settings) -> None:
    token = settings.http.token
    if not token or len(token) < 16:
        raise SystemExit("HTTP mode needs [http] token (at least 16 characters) in the config file or COMFY_MCP_HTTP_TOKEN")
    config: dict[str, Any] = {"host": settings.http.host, "port": settings.http.port, "log_level": "debug" if settings.debug else "warning", "access_log": settings.debug}
    uvicorn.run(build_app(mcp, settings), **config)
