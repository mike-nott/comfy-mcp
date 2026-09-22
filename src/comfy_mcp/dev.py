"""Entry point for the MCP Inspector: `uv run mcp dev src/comfy_mcp/dev.py`."""

from .config import load
from .server import build_server

mcp = build_server(load(debug=True))
