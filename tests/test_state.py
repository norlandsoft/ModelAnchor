"""下载意图持久化的测试。"""

from __future__ import annotations

from pathlib import Path

from app.state import DownloadIndex


def test_remember_and_lookup_roundtrip(tmp_path: Path):
    path = tmp_path / "state" / "downloads.json"
    index = DownloadIndex(path)
    index.remember("a.safetensors", "https://example.com/a", "0" * 64)

    again = DownloadIndex(path)  # 模拟进程重启
    entry = again.lookup("a.safetensors")
    assert entry is not None
    assert entry["url"] == "https://example.com/a"
    assert entry["sha256"] == "0" * 64
    assert again.lookup("missing.safetensors") is None


def test_index_keeps_entries_bounded(tmp_path: Path):
    index = DownloadIndex(tmp_path / "d.json")
    for i in range(205):
        index.remember(f"f{i}.safetensors", f"https://example.com/{i}")
    assert index.lookup("f0.safetensors") is None
    assert index.lookup("f204.safetensors") is not None
