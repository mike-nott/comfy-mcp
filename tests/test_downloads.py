import io
import re

import httpx
import pytest
from PIL import Image

from comfy_mcp import comfy as comfy_mod
from comfy_mcp.config import Settings
from comfy_mcp.downloads import DownloadStore
from comfy_mcp.http import build_app
from comfy_mcp.server import build_server

TOKEN = "test-token-0123456789abcdef"


def _png(w=64, h=48, colour=(10, 120, 200)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="PNG")
    return buf.getvalue()


def test_store_lifecycle(monkeypatch):
    store = DownloadStore(ttl=60, grace=0)
    handle = store.issue(b"abc", "20260923-1-5.png", "image/png")
    assert re.match(r"^[A-Za-z0-9_-]{24,64}\.png$", handle)
    assert store.get(handle).data == b"abc" and store.status(handle) == "ready"
    assert store.get("../etc/passwd") is None and store.get("nope.png") is None
    store.complete(handle)
    assert store.get(handle) is None and store.status(handle) == "delivered"
    graced = DownloadStore(ttl=60, grace=30)
    kept = graced.issue(b"retry", "c.png", "image/png")
    graced.complete(kept)
    assert graced.get(kept).data == b"retry" and graced.status(kept) == "delivered"  # retry window
    other = store.issue(b"x", "a.mp4", "video/mp4")
    assert other.endswith(".mp4") and other != handle
    store.ttl = 0
    expired = store.issue(b"y", "b.png", "image/png")
    assert store.get(expired) is None and store.status(expired) == "expired"


@pytest.fixture
def fake_comfy(monkeypatch):
    """Stand-in ComfyUI: every loader lists the configured files, every node exists, renders return fixed bytes."""
    s = Settings()
    names = {s.qwen21.diffusion_model, s.qwen21.text_encoder, s.qwen21.vae, *vars(s.minimax_h3).values()}

    async def loader_choices(self, node, field):
        return sorted(names)

    async def object_info(self, node):
        return {"input": {"required": {}}}

    async def run(self, graph, *, save_node, expected, on_progress=None, on_submitted=None, history_node=None):
        if history_node:
            return comfy_mod.RunResult(images=[_png(32, 18)], files=[("mcp-x_00001-audio.mp4", b"\x00\x00\x00\x18ftypmp42" + b"v" * 1000)])
        return [_png(colour=(i * 40, 90, 160)) for i in range(expected)]

    monkeypatch.setattr(comfy_mod.ComfyClient, "loader_choices", loader_choices)
    monkeypatch.setattr(comfy_mod.ComfyClient, "object_info", object_info)
    monkeypatch.setattr(comfy_mod.ComfyClient, "run", run)


def _server(tmp_path, policy="never", http_mode=True):
    s = Settings(output_dir=tmp_path / "out", save_policy=policy, http_mode=http_mode, max_wait=30)
    s.http.token = TOKEN
    return s, build_server(s)


def _text(result):
    return "\n".join(b.text for b in result.content if b.type == "text")


def _handles(text):
    return [line.split("download: comfy://result/", 1)[1] for line in text.splitlines() if line.startswith("download: comfy://result/")]


async def test_never_policy_images_and_download_route(tmp_path, fake_comfy):
    settings, mcp = _server(tmp_path)
    mcp.comfy_downloads.grace = 0
    result = await mcp.call_tool("generate_image", {"prompt": "a fox", "count": 2, "seed": 5, "save": True})
    text = _text(result)
    assert not result.is_error, text
    handles = _handles(text)
    assert len(handles) == 2 and len(set(handles)) == 2 and all(h.endswith(".png") for h in handles)
    assert "saved:" not in text and "png_base64" not in text
    assert sum(1 for b in result.content if b.type == "image") == 2
    assert not (tmp_path / "out").exists()  # nothing written, even though the model passed save=true

    app = build_app(mcp, settings)
    auth = {"Authorization": f"Bearer {TOKEN}"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        assert (await http.get(f"/dl/{handles[0]}")).status_code == 401
        head = await http.head(f"/dl/{handles[0]}", headers=auth)
        assert head.status_code == 200 and head.headers["content-type"] == "image/png"
        got = await http.get(f"/dl/{handles[0]}", headers=auth)  # HEAD did not burn it
        assert got.status_code == 200 and got.content.startswith(b"\x89PNG")
        assert re.match(r'attachment; filename="\d{8}-\d{6}-5-1\.png"', got.headers["content-disposition"])
        assert got.headers["cache-control"] == "no-store"
        assert (await http.get(f"/dl/{handles[0]}", headers=auth)).status_code == 404  # one-shot
        assert (await http.get("/dl/not-a-handle", headers=auth)).status_code == 404

    assert text.splitlines()[0].startswith("Qwen Image 2.1 · 2 images")


async def test_fetch_twice_same_handles(tmp_path, fake_comfy):
    settings, mcp = _server(tmp_path)
    settings.max_wait = 0.0  # force the job_id path
    mcp.comfy_downloads.grace = 0
    first = await mcp.call_tool("generate_image", {"prompt": "x", "seed": 7})
    import json

    job_id = json.loads(_text(first))["job_id"]
    waited = await mcp.call_tool("wait_for_job", {"job_id": job_id, "timeout": 30})
    fetched = await mcp.call_tool("fetch_result", {"job_id": job_id})
    h1, h2 = _handles(_text(waited)), _handles(_text(fetched))
    assert h1 == h2 and len(h1) == 1  # same live link, not a second copy
    mcp.comfy_downloads.complete(h1[0])
    after = _text(await mcp.call_tool("fetch_result", {"job_id": job_id}))
    assert _handles(after) == [] and "already downloaded" in after
    assert sum(1 for b in (await mcp.call_tool("fetch_result", {"job_id": job_id})).content if b.type == "image") == 1


async def test_video_never_policy(tmp_path, fake_comfy):
    settings, mcp = _server(tmp_path)
    import json

    pending = json.loads(_text(await mcp.call_tool("generate_video", {"prompt": "fox", "seconds": 5, "seed": 9})))
    done = await mcp.call_tool("wait_for_job", {"job_id": pending["job_id"], "timeout": 30})
    text = _text(done)
    handles = _handles(text)
    assert len(handles) == 1 and handles[0].endswith(".mp4") and "saved:" not in text
    assert sum(1 for b in done.content if b.type == "image") == 1  # poster
    entry = mcp.comfy_downloads.get(handles[0])
    assert entry.mime == "video/mp4" and entry.filename.endswith("-9.mp4")
    assert not (tmp_path / "out").exists()


async def test_policies_default_and_always(tmp_path, fake_comfy):
    # default + stdio + save=false keeps the legacy base64 path; default + http gives handles instead
    _, stdio = _server(tmp_path, policy="default", http_mode=False)
    assert "png_base64" in _text(await stdio.call_tool("generate_image", {"prompt": "x", "save": False}))
    _, http_default = _server(tmp_path, policy="default", http_mode=True)
    text = _text(await http_default.call_tool("generate_image", {"prompt": "x", "save": False}))
    assert _handles(text) and "png_base64" not in text
    saved = _text(await http_default.call_tool("generate_image", {"prompt": "x", "save": True}))
    assert "saved:" in saved and not _handles(saved)
    _, always = _server(tmp_path, policy="always", http_mode=True)
    assert "saved:" in _text(await always.call_tool("generate_image", {"prompt": "x", "save": False}))


def test_config_policy(tmp_path):
    from comfy_mcp import config

    path = tmp_path / "c.toml"
    path.write_text('save_policy = "never"\n')
    assert config.load(path, env={}).save_policy == "never"
    assert config.load(path, env={"COMFY_MCP_SAVE_POLICY": "always"}).save_policy == "always"
    path.write_text('save_policy = "sometimes"\n')
    with pytest.raises(ValueError):
        config.load(path, env={})
