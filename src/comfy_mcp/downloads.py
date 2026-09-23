"""In-memory store for full-size results handed out as one-shot download handles.

Nothing here touches disk. A handle is `<random>.<ext>`. The bytes live until `grace` seconds after
the first completed authenticated fetch (a short retry window, because a server cannot fully know the
client stored what it sent) or until the TTL, whichever comes first.
"""

from __future__ import annotations

import re
import secrets
import time
from dataclasses import dataclass, field

HANDLE_RE = re.compile(r"^[A-Za-z0-9_-]{24,64}\.[a-z0-9]{2,5}$")
URI_PREFIX = "comfy://result/"


@dataclass
class Entry:
    data: bytes
    filename: str
    mime: str
    expires: float
    delivered_at: float | None = None


@dataclass
class DownloadStore:
    ttl: float
    grace: float = 60.0
    _entries: dict[str, Entry] = field(default_factory=dict)

    def issue(self, data: bytes, filename: str, mime: str) -> str:
        self.sweep()
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
        handle = f"{secrets.token_urlsafe(24)}.{ext}"
        self._entries[handle] = Entry(data=data, filename=filename, mime=mime, expires=time.monotonic() + self.ttl)
        return handle

    def get(self, handle: str) -> Entry | None:
        self.sweep()
        if not HANDLE_RE.match(handle):
            return None
        entry = self._entries.get(handle)
        return entry if entry and entry.data else None

    def complete(self, handle: str) -> None:
        """A full response was sent: start the grace window, after which the bytes are dropped.
        The handle stays known (as delivered) until the TTL."""
        entry = self._entries.get(handle)
        if entry and entry.delivered_at is None:
            entry.delivered_at = time.monotonic()
        self.sweep()

    def status(self, handle: str) -> str:
        self.sweep()
        entry = self._entries.get(handle)
        if entry is None:
            return "expired"
        return "ready" if entry.delivered_at is None else "delivered"

    def sweep(self) -> None:
        now = time.monotonic()
        for handle, entry in list(self._entries.items()):
            if entry.expires <= now:
                entry.data = b""
                del self._entries[handle]
            elif entry.delivered_at is not None and now - entry.delivered_at >= self.grace:
                entry.data = b""
