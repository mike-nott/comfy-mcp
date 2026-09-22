"""Streamable HTTP transport behind a static bearer token."""

from __future__ import annotations

import hmac
from typing import Any

import uvicorn
from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import Settings


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


def serve(mcp: MCPServer, settings: Settings) -> None:
    token = settings.http.token
    if not token or len(token) < 16:
        raise SystemExit("HTTP mode needs [http] token (at least 16 characters) in the config file or COMFY_MCP_HTTP_TOKEN")
    app = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        host=settings.http.host,
    )
    config: dict[str, Any] = {"host": settings.http.host, "port": settings.http.port, "log_level": "debug" if settings.debug else "warning", "access_log": settings.debug}
    uvicorn.run(BearerToken(app, token), **config)
