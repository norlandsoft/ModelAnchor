"""分段并行、断点续传、代理禁用的端到端测试。"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from app.config import Settings
from app.downloader import DownloadManager

PAYLOAD = bytes(range(256)) * 12288 + b"tail-bytes"  # 3 MiB + 10 字节


class RangeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    support_range = True
    ranges: ClassVar[list[tuple[int, int]]] = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        rng = self.headers.get("Range")
        if not rng or not self.support_range:
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD)
            return

        spec = rng.split("=", 1)[1]
        raw_start, _, raw_end = spec.partition("-")
        start = int(raw_start or 0)
        end = int(raw_end) if raw_end else len(PAYLOAD) - 1
        end = min(end, len(PAYLOAD) - 1)
        if start >= len(PAYLOAD):
            self.send_response(416)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        type(self).ranges.append((start, end))
        body = PAYLOAD[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(PAYLOAD)}")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.fixture()
def server():
    RangeHandler.ranges = []
    RangeHandler.support_range = True
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def settings_for(tmp_path: Path, **overrides) -> Settings:
    values = {
        "model_dir": tmp_path,
        "model_dir_source": "test",
        "download_connections": 4,
        "download_chunk_bytes": 1024 * 1024,
        "segment_min_bytes": 0,
        "segment_mode": "always",
    }
    values.update(overrides)
    return Settings(**values)


def wait_for(job, timeout: float = 20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if job.state in {"done", "failed", "canceled"}:
            return job
        time.sleep(0.05)
    raise AssertionError(f"任务没有结束：{job.state}")


def test_segmented_download_matches_bytes(tmp_path: Path, server):
    manager = DownloadManager(settings_for(tmp_path))
    try:
        job = manager.start(f"{server}/model.safetensors")
        wait_for(job)
        assert job.state == "done", job.error
        assert (tmp_path / "model.safetensors").read_bytes() == PAYLOAD
        # 应该走的是多段 Range，而不是一条整文件请求
        assert len(RangeHandler.ranges) >= 4
        assert not list(tmp_path.glob("*.part*"))
    finally:
        manager.shutdown()


def test_resume_skips_completed_chunks(tmp_path: Path, server):
    settings = settings_for(tmp_path)
    chunk = settings.download_chunk_bytes
    total = len(PAYLOAD)
    count = (total + chunk - 1) // chunk

    dest = tmp_path / "model.safetensors"
    part = dest.with_name(dest.name + ".part")
    # 前两个分片已下好，其余是空洞
    part.write_bytes(PAYLOAD[: 2 * chunk] + b"\x00" * (total - 2 * chunk))
    (tmp_path / "model.safetensors.part.json").write_text(
        json.dumps({"total": total, "chunk": chunk, "done": [0, 1]}), encoding="utf-8"
    )

    manager = DownloadManager(settings)
    try:
        job = manager.start(f"{server}/model.safetensors")
        wait_for(job)
        assert job.state == "done", job.error
        assert dest.read_bytes() == PAYLOAD
        # 忽略 0-0 的探测请求，只统计真正的分片
        chunks_requested = [
            (start, end) for start, end in RangeHandler.ranges if not (start == 0 and end == 0)
        ]
        assert chunks_requested
        assert all(start >= 2 * chunk for start, _end in chunks_requested), chunks_requested
        assert count > 2
    finally:
        manager.shutdown()


def test_ignores_proxy_environment(tmp_path: Path, server, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    # 去掉 no_proxy，否则即使走了代理环境也会被 127.0.0.1 豁免，测不出问题
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)

    manager = DownloadManager(settings_for(tmp_path))
    try:
        job = manager.start(f"{server}/model.safetensors")
        wait_for(job)
        assert job.state == "done", job.error
        assert (tmp_path / "model.safetensors").read_bytes() == PAYLOAD
    finally:
        manager.shutdown()


def test_falls_back_when_server_lacks_range(tmp_path: Path, server):
    RangeHandler.support_range = False
    manager = DownloadManager(settings_for(tmp_path))
    try:
        job = manager.start(f"{server}/model.safetensors")
        wait_for(job)
        assert job.state == "done", job.error
        assert (tmp_path / "model.safetensors").read_bytes() == PAYLOAD
    finally:
        manager.shutdown()


def test_repo_download_parallel_and_keeps_total(tmp_path: Path, server):
    files = [
        {"path": f"part/{i}/model.safetensors", "url": f"{server}/f{i}", "size": len(PAYLOAD)}
        for i in range(5)
    ]
    settings = settings_for(tmp_path)
    manager = DownloadManager(settings)
    try:
        job = manager.start_repo("owner/repo", files)
        wait_for(job, timeout=30)
        assert job.state == "done", job.error
        assert job.files_done == 5
        assert job.total_bytes == len(PAYLOAD) * 5
        for i in range(5):
            assert (tmp_path / "owner_repo" / "part" / str(i) / "model.safetensors").read_bytes() == PAYLOAD
    finally:
        manager.shutdown()


def test_repo_download_accepts_unknown_sizes(tmp_path: Path, server):
    files = [{"path": f"f{i}.bin", "url": f"{server}/f{i}", "size": None} for i in range(3)]
    manager = DownloadManager(settings_for(tmp_path))
    try:
        job = manager.start_repo("owner/unknown", files)
        wait_for(job, timeout=30)
        assert job.state == "done", job.error
        assert job.files_done == 3
        assert job.total_bytes is None
    finally:
        manager.shutdown()


def test_legacy_contiguous_part_resumes_without_meta(tmp_path: Path, server):
    """旧版本留下的连续 .part 没有分片元数据，也应该从断点续。"""
    settings = settings_for(tmp_path)
    chunk = settings.download_chunk_bytes
    dest = tmp_path / "legacy.safetensors"
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(PAYLOAD[: 2 * chunk])

    manager = DownloadManager(settings)
    try:
        job = manager.start(f"{server}/legacy.safetensors")
        wait_for(job)
        assert job.state == "done", job.error
        assert dest.read_bytes() == PAYLOAD
        requested = [start for start, _end in RangeHandler.ranges if not (start == 0 and _end == 0)]
        assert requested and all(start >= 2 * chunk for start in requested), requested
    finally:
        manager.shutdown()


def test_sparse_part_without_meta_restarts_safely(tmp_path: Path, server):
    """稀疏文件（预分配过但没写完）没有元数据时不能瞎续，必须重下整份。"""
    settings = settings_for(tmp_path)
    dest = tmp_path / "sparse.safetensors"
    part = dest.with_name(dest.name + ".part")
    with open(part, "wb") as fh:
        fh.truncate(len(PAYLOAD))  # 稀疏空洞

    manager = DownloadManager(settings)
    try:
        job = manager.start(f"{server}/sparse.safetensors")
        wait_for(job)
        assert job.state == "done", job.error
        assert dest.read_bytes() == PAYLOAD
        requested = [start for start, _end in RangeHandler.ranges if not (start == 0 and _end == 0)]
        assert requested and min(requested) == 0
    finally:
        manager.shutdown()
