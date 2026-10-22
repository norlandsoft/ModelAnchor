"""下载引擎的端到端测试：本地 HTTP 服务，不依赖外网。"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from app.config import Settings
from app.downloader import DownloadManager


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    calls = 0
    payload = b"model-bytes-" * 4096  # 48 KiB

    def log_message(self, *args):
        pass

    def do_GET(self):
        type(self).calls += 1
        if type(self).calls < 3:
            body = b"rate limited"
            self.send_response(429)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        start = 0
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            start = int(rng[6:].split("-")[0] or 0)
            if start >= len(self.payload):
                self.send_response(416)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        chunk = self.payload[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(chunk)))
        if start:
            self.send_header("Content-Range", f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}")
        self.end_headers()
        self.wfile.write(chunk)


@pytest.fixture()
def server():
    Handler.calls = 0
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def wait_for(job, timeout: float = 10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if job.state in {"done", "failed", "canceled"}:
            return job
        time.sleep(0.05)
    raise AssertionError(f"任务没有结束：{job.state}")


def test_download_retries_on_429(tmp_path: Path, server, monkeypatch):
    monkeypatch.setattr("app.downloader._RETRY_DELAYS", (0.01, 0.01, 0.01))
    settings = Settings(model_dir=tmp_path, model_dir_source="test")
    manager = DownloadManager(settings)
    try:
        job = manager.start(f"{server}/model.safetensors")
        wait_for(job)
        assert job.state == "done", job.error
        assert Handler.calls >= 3
        assert (tmp_path / "model.safetensors").read_bytes() == Handler.payload
    finally:
        manager.shutdown()


def test_download_reports_rate_limit_after_retries(tmp_path: Path, server, monkeypatch):
    monkeypatch.setattr("app.downloader._RETRY_DELAYS", (0.01,))
    Handler.calls = 0
    # 让服务永远 429
    monkeypatch.setattr(Handler, "calls", -10, raising=False)
    settings = Settings(model_dir=tmp_path, model_dir_source="test")
    manager = DownloadManager(settings)
    try:
        job = manager.start(f"{server}/nope.safetensors", filename="x.safetensors")
        wait_for(job)
        assert job.state == "failed"
        assert "429" in (job.error or "")
    finally:
        manager.shutdown()


def test_duplicate_download_conflicts(tmp_path: Path, server, monkeypatch):
    monkeypatch.setattr("app.downloader._RETRY_DELAYS", (0.01,))
    Handler.calls = 10
    settings = Settings(model_dir=tmp_path, model_dir_source="test")
    manager = DownloadManager(settings)
    try:
        (tmp_path / "exists.safetensors").write_bytes(b"x")
        from app.downloader import FileExists

        with pytest.raises(FileExists):
            manager.start(f"{server}/model.safetensors", filename="exists.safetensors")
    finally:
        manager.shutdown()
