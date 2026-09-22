"""`comfy-mcp` entry point."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import DEFAULT_PATH, load, write_example


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="comfy-mcp", description="MCP server for private one-shot media generation on your own ComfyUI.")
    parser.add_argument("--config", help=f"settings file (default: $COMFY_MCP_CONFIG or {DEFAULT_PATH})")
    parser.add_argument("--http", action="store_true", help="serve Streamable HTTP with a bearer token instead of stdio")
    parser.add_argument("--debug", action="store_true", help="terse diagnostics on stderr (never a file)")
    parser.add_argument("--version", action="version", version=f"comfy-mcp {__version__}")
    sub = parser.add_subparsers(dest="command")
    init = sub.add_parser("init", help="write a commented example config file")
    init.add_argument("--path", help=f"where to write it (default {DEFAULT_PATH})")
    init.add_argument("--force", action="store_true", help="overwrite an existing file")
    sub.add_parser("check", help="print the resolved settings and ComfyUI status, then exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        from pathlib import Path

        try:
            path = write_example(Path(args.path).expanduser() if args.path else None, force=args.force)
        except FileExistsError as error:
            print(error, file=sys.stderr)
            return 1
        print(f"Wrote {path}")
        return 0

    try:
        settings = load(args.config, debug=args.debug)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1

    if args.command == "check":
        import asyncio
        import json

        from .comfy import ComfyClient, ComfyError

        print(f"config: {settings.source or '(defaults)'}")
        print(f"comfyui_url: {settings.comfyui_url}")
        print(f"output_dir: {settings.output_dir}")
        q = settings.qwen21
        print(f"qwen21: {q.diffusion_model} / {q.text_encoder} / {q.vae}")
        async def probe() -> int:
            client = ComfyClient(settings.comfyui_url, debug=args.debug)
            try:
                print(json.dumps(await client.status(), indent=2))
                return 0
            except ComfyError as error:
                print(error, file=sys.stderr)
                return 2
            finally:
                await client.aclose()

        return asyncio.run(probe())

    from .server import build_server

    mcp = build_server(settings)
    if args.http:
        from .http import serve

        serve(mcp, settings)
    else:
        mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
