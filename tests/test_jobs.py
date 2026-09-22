import asyncio

import pytest

from comfy_mcp.jobs import JobRegistry


async def test_job_lifecycle():
    registry = JobRegistry(ttl=60)

    async def runner(job):
        job.state = "running"
        await asyncio.sleep(0.01)
        return [b"png"]

    job = registry.create("generate", {"seed": 1}, runner)
    await job.task
    assert job.state == "done" and job.results == [b"png"]
    assert registry.get(job.id).public()["images"] == 1


async def test_job_error_and_cancel():
    registry = JobRegistry(ttl=60)

    async def failing(job):
        raise RuntimeError("boom")

    job = registry.create("generate", {}, failing)
    with pytest.raises(RuntimeError):
        await job.task
    assert job.state == "error" and job.error == "boom"

    hooked = []

    async def slow(job):
        await asyncio.sleep(10)

    slow_job = registry.create("generate", {}, slow)

    async def hook():
        hooked.append(True)

    slow_job.cancel_hook = hook
    await asyncio.sleep(0)
    await registry.cancel(slow_job.id)
    assert slow_job.state == "cancelled" and hooked == [True]


async def test_sweep_expires():
    registry = JobRegistry(ttl=0)

    async def runner(job):
        return [b"x"]

    job = registry.create("generate", {}, runner)
    await job.task
    await asyncio.sleep(0.01)
    with pytest.raises(KeyError):
        registry.get(job.id)
