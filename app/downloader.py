"""下载引擎：单文件与整目录（diffusers）任务、断点续传、进度、取消。

用线程而不是 async：下载是长 IO，线程里阻塞写盘不会拖住 FastAPI 的事件循环，
而且 httpx 的同步流式接口在这里更直观。
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from .config import Settings
from .library import resolve_target

CHUNK_SIZE = 1024 * 1024
USER_AGENT = "ModelAnchor/0.1 (local model manager for DrawThings)"
_SPEED_WINDOW = 6.0
_BAD_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


class DownloadError(Exception):
    """可预期的下载失败，消息直接给用户看。"""


class FileExists(DownloadError):
    pass


class JobConflict(DownloadError):
    pass


def sanitize_filename(raw: str, *, strict: bool = False) -> str:
    """从 URL 或用户输入得到安全文件名。strict=True 时显式拒绝路径分隔符，而不是改写它。"""
    if "://" in raw:
        name = unquote(urlparse(raw).path).rsplit("/", 1)[-1]
    else:
        name = raw
        if strict and ("/" in name or "\\" in name):
            raise DownloadError("文件名不能包含路径分隔符")
    name = _BAD_CHARS.sub("_", name).strip().strip(".")
    if not name or name in {".", ".."}:
        raise DownloadError("无法从链接推断文件名，请手动填写")
    if len(name) > 180:
        raise DownloadError("文件名过长（超过 180 字符）")
    return name


def slug_for_repo(repo_id: str) -> str:
    return sanitize_filename(repo_id.replace("/", "_"), strict=True)


@dataclass
class DownloadJob:
    id: str
    filename: str
    url: str = ""
    dest: Path = field(default_factory=Path)
    kind: str = "file"  # file | repo
    sha256_expected: str | None = None
    state: str = "queued"
    bytes_done: int = 0
    total_bytes: int | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    speed_bps: float = 0.0
    # 目录任务
    files: list[dict] = field(default_factory=list)
    files_done: int = 0
    files_total: int = 1
    current_file: str = ""
    cancel_event: threading.Event = field(default_factory=threading.Event)
    _samples: deque = field(default_factory=deque)

    def to_dict(self) -> dict:
        percent = None
        if self.total_bytes:
            percent = round(min(self.bytes_done / self.total_bytes, 1.0) * 100, 1)
        eta = None
        if self.total_bytes and self.speed_bps > 1:
            remaining = max(self.total_bytes - self.bytes_done, 0)
            eta = round(remaining / self.speed_bps, 1)
        return {
            "id": self.id,
            "kind": self.kind,
            "filename": self.filename,
            "url": self.url,
            "state": self.state,
            "bytes_done": self.bytes_done,
            "total_bytes": self.total_bytes,
            "percent": percent,
            "speed_bps": round(self.speed_bps, 1),
            "eta_seconds": eta,
            "error": self.error,
            "files_done": self.files_done,
            "files_total": self.files_total,
            "current_file": self.current_file,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class DownloadManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._jobs: dict[str, DownloadJob] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, settings.max_concurrent_downloads),
            thread_name_prefix="modelanchor-dl",
        )

    # ---------------------------------------------------------------- 生命周期
    def start(
        self,
        url: str,
        filename: str | None = None,
        sha256: str | None = None,
    ) -> DownloadJob:
        if not url.lower().startswith(("http://", "https://")):
            raise DownloadError("只支持 http / https 链接")

        name = sanitize_filename(filename, strict=True) if filename else sanitize_filename(url)
        target = resolve_target(self.settings.model_dir, name)
        part = target.with_name(target.name + ".part")

        with self._lock:
            for existing in self._jobs.values():
                if existing.kind == "file" and existing.filename == name and existing.state in {
                    "queued",
                    "running",
                    "verifying",
                }:
                    raise JobConflict(f"“{name}”已经在下载中")
            if target.exists() and not part.exists():
                raise FileExists(f"“{name}”已经存在于模型目录")

        if sha256 and not re.fullmatch(r"[0-9a-f]{64}", sha256.lower()):
            raise DownloadError("sha256 应为 64 位十六进制")

        job = DownloadJob(
            id=uuid.uuid4().hex[:12],
            filename=name,
            url=url,
            dest=target,
            sha256_expected=sha256.lower() if sha256 else None,
        )
        with self._lock:
            self._jobs[job.id] = job

        self.settings.model_dir.mkdir(parents=True, exist_ok=True)
        self._pool.submit(self._run_file, job)
        return job

    def start_repo(self, repo_id: str, files: list[dict], source: str = "") -> DownloadJob:
        """下载整个仓库目录（diffusers 结构），按文件顺序串行，进度按字节聚合。"""
        if not files:
            raise DownloadError("仓库里没有可下载的文件")
        slug = slug_for_repo(repo_id)
        dest = resolve_target(self.settings.model_dir, slug)
        if dest.exists() and not dest.is_dir():
            raise FileExists(f"“{slug}”已存在且不是目录")

        with self._lock:
            for existing in self._jobs.values():
                if existing.kind == "repo" and existing.filename == slug and existing.state in {
                    "queued",
                    "running",
                }:
                    raise JobConflict(f"“{repo_id}”已经在下载中")

        total = 0
        known = True
        for item in files:
            size = item.get("size")
            if size is None:
                known = False
            else:
                total += int(size)

        job = DownloadJob(
            id=uuid.uuid4().hex[:12],
            kind="repo",
            filename=slug,
            url="",
            dest=dest,
            files=files,
            files_total=len(files),
            total_bytes=total if known else None,
            current_file="",
        )
        with self._lock:
            self._jobs[job.id] = job

        self.settings.model_dir.mkdir(parents=True, exist_ok=True)
        self._pool.submit(self._run_repo, job)
        return job

    def get(self, job_id: str) -> DownloadJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[DownloadJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at)

    def cancel(self, job_id: str) -> DownloadJob:
        job = self.get(job_id)
        if job is None:
            raise DownloadError("任务不存在")
        if job.state in {"queued", "running"}:
            job.cancel_event.set()
        return job

    def forget(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise DownloadError("任务不存在")
            if job.state in {"queued", "running", "verifying"}:
                raise DownloadError("任务未结束，先取消它")
            del self._jobs[job_id]

    def shutdown(self) -> None:
        for job in self.list():
            job.cancel_event.set()
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ 单文件
    def _run_file(self, job: DownloadJob) -> None:
        part = job.dest.with_name(job.dest.name + ".part")
        try:
            if job.cancel_event.is_set():
                raise _Cancelled
            job.state = "running"
            job.started_at = time.time()

            with _client() as client:
                self._fetch(client, job, job.url, part)

            if job.cancel_event.is_set():
                raise _Cancelled

            if job.sha256_expected:
                job.state = "verifying"
                digest = _sha256_of(part)
                if digest != job.sha256_expected:
                    part.unlink(missing_ok=True)
                    raise DownloadError(
                        f"校验失败：期望 {job.sha256_expected[:12]}…，实际 {digest[:12]}…"
                    )

            job.dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(part, job.dest)
            job.state = "done"
        except _Cancelled:
            job.state = "canceled"
        except DownloadError as exc:
            job.state = "failed"
            job.error = str(exc)
        except httpx.HTTPError as exc:
            job.state = "failed"
            job.error = f"网络错误：{exc}"
        except OSError as exc:
            job.state = "failed"
            job.error = f"写盘失败：{exc}"
        except Exception as exc:  # noqa: BLE001 —— 线程里兜底，避免任务永远停在 running
            job.state = "failed"
            job.error = f"未知错误：{exc}"
        finally:
            job.finished_at = time.time()

    # ------------------------------------------------------------------ 整目录
    def _run_repo(self, job: DownloadJob) -> None:
        try:
            if job.cancel_event.is_set():
                raise _Cancelled
            job.state = "running"
            job.started_at = time.time()
            job.dest.mkdir(parents=True, exist_ok=True)

            done_bytes = 0
            for item in job.files:
                target = self._repo_target(job, item["path"])
                part = target.with_name(target.name + ".part")
                size = item.get("size")
                if target.exists() and (size is None or target.stat().st_size == size):
                    done_bytes += target.stat().st_size
                elif part.exists():
                    done_bytes += part.stat().st_size
            job.bytes_done = done_bytes

            with _client() as client:
                for index, item in enumerate(job.files):
                    if job.cancel_event.is_set():
                        raise _Cancelled
                    target = self._repo_target(job, item["path"])
                    part = target.with_name(target.name + ".part")
                    job.current_file = item["path"]
                    job.files_done = index

                    size = item.get("size")
                    if target.exists() and size is not None and target.stat().st_size == size:
                        continue

                    target.parent.mkdir(parents=True, exist_ok=True)
                    self._fetch(client, job, item["url"], part)
                    os.replace(part, target)
                    job.files_done = index + 1

            job.state = "done"
        except _Cancelled:
            job.state = "canceled"
        except DownloadError as exc:
            job.state = "failed"
            job.error = str(exc)
        except httpx.HTTPError as exc:
            job.state = "failed"
            job.error = f"网络错误：{exc}"
        except OSError as exc:
            job.state = "failed"
            job.error = f"写盘失败：{exc}"
        except Exception as exc:  # noqa: BLE001
            job.state = "failed"
            job.error = f"未知错误：{exc}"
        finally:
            job.current_file = ""
            job.finished_at = time.time()

    def _repo_target(self, job: DownloadJob, rel_path: str) -> Path:
        rel = rel_path.replace("\\", "/").lstrip("/")
        if ".." in rel.split("/"):
            raise DownloadError(f"非法文件路径：{rel_path}")
        target = (job.dest / rel).resolve()
        root = job.dest.resolve()
        if target != root and root not in target.parents:
            raise DownloadError(f"路径逃逸：{rel_path}")
        return target

    # ------------------------------------------------------------------ 传输
    def _fetch(self, client: httpx.Client, job: DownloadJob, url: str, part: Path) -> int:
        """带退避的重试：镜像站偶尔会整站 429。"""
        for attempt in range(len(_RETRY_DELAYS) + 1):
            try:
                return self._fetch_once(client, job, url, part)
            except _RetryableStatus as exc:
                if attempt >= len(_RETRY_DELAYS):
                    raise DownloadError(
                        f"服务器限流或暂时不可用（HTTP {exc.status}），请稍后再试"
                    ) from exc
                end = time.monotonic() + _RETRY_DELAYS[attempt]
                while time.monotonic() < end:
                    if job.cancel_event.is_set():
                        raise _Cancelled from exc
                    time.sleep(0.05)
        raise DownloadError("下载失败")

    def _fetch_once(self, client: httpx.Client, job: DownloadJob, url: str, part: Path) -> int:
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
        if offset:
            headers["Range"] = f"bytes={offset}-"

        if job.kind == "file" and job.bytes_done < offset:
            job.bytes_done = offset

        with client.stream("GET", url, headers=headers) as response:
            if response.status_code == 416 and offset:
                raise DownloadError("续传范围无效，请删除 .part 文件后重试")
            if offset and response.status_code == 200:
                job.bytes_done -= offset  # 服务器不认 Range，从头重下
                offset = 0
            if response.status_code in _RETRYABLE_STATUS:
                raise _RetryableStatus(response.status_code)
            if response.status_code not in (200, 206):
                raise DownloadError(f"服务器返回 HTTP {response.status_code}")

            length = response.headers.get("content-length")
            remaining = int(length) if length and length.isdigit() else None
            if job.kind == "file":
                job.total_bytes = (remaining + offset) if remaining is not None else None
            if remaining is not None:
                self._ensure_space(job, remaining)

            mode = "ab" if offset else "wb"
            part.parent.mkdir(parents=True, exist_ok=True)
            with open(part, mode) as handle:
                for chunk in response.iter_bytes(CHUNK_SIZE):
                    if job.cancel_event.is_set():
                        raise _Cancelled
                    if not chunk:
                        continue
                    handle.write(chunk)
                    self._tick(job, len(chunk))
        return offset

    def _tick(self, job: DownloadJob, nbytes: int) -> None:
        now = time.monotonic()
        job.bytes_done += nbytes
        job._samples.append((now, job.bytes_done))
        while job._samples and now - job._samples[0][0] > _SPEED_WINDOW:
            job._samples.popleft()
        if len(job._samples) >= 2:
            first_t, first_b = job._samples[0]
            dt = now - first_t
            if dt > 0.2:
                job.speed_bps = (job.bytes_done - first_b) / dt

    def _ensure_space(self, job: DownloadJob, needed: int) -> None:
        try:
            usage = os.statvfs(self.settings.model_dir)
            free = usage.f_bavail * usage.f_frsize
        except OSError:
            return
        if free < needed:
            raise DownloadError(
                f"磁盘空间不足：还需 {needed / 2**30:.1f} GiB，剩余 {free / 2**30:.1f} GiB"
            )


class _Cancelled(Exception):
    pass


class _RetryableStatus(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status


_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# 第 n 次失败后等待 _RETRY_DELAYS[n] 秒；测试里会把它改小
_RETRY_DELAYS = (2.0, 4.0, 8.0)


def _client() -> httpx.Client:
    return httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(30.0, read=120.0),
        headers={"User-Agent": USER_AGENT},
    )


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
