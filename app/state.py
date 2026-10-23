"""下载意图的持久化。

任务本身存在内存里，进程重启就没了；但「这个文件名当初是从哪个 URL 下的」值得记下来，
否则页面上那个 .part 就只能删掉——明明还能续。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

MAX_ENTRIES = 200


class DownloadIndex:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return {k: v for k, v in raw.items() if isinstance(v, dict)}
        except (OSError, ValueError):
            pass
        return {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._entries, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            pass

    def remember(self, filename: str, url: str, sha256: str | None = None) -> None:
        with self._lock:
            self._entries[filename] = {
                "url": url,
                "sha256": sha256 or "",
                "updated": time.time(),
            }
            if len(self._entries) > MAX_ENTRIES:
                oldest = sorted(self._entries, key=lambda k: self._entries[k].get("updated", 0))
                for key in oldest[: len(self._entries) - MAX_ENTRIES]:
                    self._entries.pop(key, None)
            self._save()

    def lookup(self, filename: str) -> dict | None:
        with self._lock:
            entry = self._entries.get(filename)
            return dict(entry) if entry else None
