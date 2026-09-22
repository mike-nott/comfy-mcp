"""In-memory job registry. Results live in RAM until fetched or expired; nothing touches disk."""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Job:
    id: str
    kind: str
    params: dict[str, Any]
    created: float = field(default_factory=time.monotonic)
    started: float | None = None
    finished: float | None = None
    state: str = "queued"  # queued | running | done | error | cancelled
    prompt_id: str | None = None
    position: int | None = None
    progress: tuple[int, int] | None = None
    message: str = "submitting"
    results: list[bytes] | None = None
    error: str | None = None
    task: asyncio.Task | None = field(default=None, repr=False)
    cancel_hook: Callable[[], Awaitable[None]] | None = field(default=None, repr=False)

    @property
    def elapsed(self) -> float:
        end = self.finished or time.monotonic()
        return end - self.created

    def public(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "job_id": self.id,
            "kind": self.kind,
            "state": self.state,
            "message": self.message,
            "elapsed_s": round(self.elapsed, 1),
        }
        if self.position is not None and self.state == "queued":
            data["queue_position"] = self.position
        if self.progress and self.state == "running":
            data["progress"] = {"step": self.progress[0], "of": self.progress[1]}
        if self.error:
            data["error"] = self.error
        if self.results is not None:
            data["images"] = len(self.results)
        data.update({k: v for k, v in self.params.items() if k in ("seed", "width", "height", "count", "steps", "cfg", "model")})
        return data


class JobRegistry:
    def __init__(self, ttl: float) -> None:
        self.ttl = ttl
        self._jobs: dict[str, Job] = {}

    def create(self, kind: str, params: dict[str, Any], runner: Callable[[Job], Awaitable[list[bytes]]]) -> Job:
        self.sweep()
        job = Job(id=secrets.token_hex(6), kind=kind, params=params)
        job.task = asyncio.create_task(self._run(job, runner))
        self._jobs[job.id] = job
        return job

    async def _run(self, job: Job, runner: Callable[[Job], Awaitable[list[bytes]]]) -> list[bytes]:
        try:
            job.results = await runner(job)
            job.state = "done"
            job.message = "finished"
            return job.results
        except asyncio.CancelledError:
            job.state = "cancelled"
            job.message = "cancelled"
            raise
        except Exception as error:  # noqa: BLE001 - surfaced to the caller as job.error
            job.state = "error"
            job.error = str(error) or error.__class__.__name__
            job.message = "failed"
            raise
        finally:
            job.finished = time.monotonic()

    def get(self, job_id: str) -> Job:
        self.sweep()
        try:
            return self._jobs[job_id]
        except KeyError as error:
            raise KeyError(f"Unknown or expired job id: {job_id}") from error

    async def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.state in ("done", "error", "cancelled"):
            return job
        if job.cancel_hook:
            try:
                await job.cancel_hook()
            except Exception:  # noqa: BLE001 - best effort; the task cancel below still applies
                pass
        if job.task and not job.task.done():
            job.task.cancel()
            try:
                await job.task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        job.state = "cancelled"
        job.message = "cancelled"
        job.finished = job.finished or time.monotonic()
        return job

    def forget(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)

    def sweep(self) -> None:
        now = time.monotonic()
        for job_id, job in list(self._jobs.items()):
            if job.finished is not None and now - job.finished > self.ttl:
                job.results = None
                self._jobs.pop(job_id, None)

    def active(self) -> list[Job]:
        return [job for job in self._jobs.values() if job.state in ("queued", "running")]
