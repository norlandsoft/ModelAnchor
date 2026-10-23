"""下载引擎：分段并行、多源择优、断点续传、进度、取消。

提速思路：
1. 强制直连（trust_env=False），任何 HTTP(S)_PROXY 环境变量都不生效；
2. 同一份文件有多个源时（HF 镜像 / ModelScope），先小段测速再挑最快；
3. 支持 Range 且文件够大时切分片，用多个连接并行拉，每个分片独立落盘；
4. 复用长连接、加大读缓冲，减少重定向与握手开销。

用线程而不是 async：下载是长 IO，线程里阻塞写盘不会拖住 FastAPI 的事件循环。
"""

from __future__ import annotations

import hashlib
import json
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
from .state import DownloadIndex

CHUNK_SIZE = 1024 * 1024
USER_AGENT = "ModelAnchor/0.1 (local model manager for DrawThings)"
_SPEED_WINDOW = 6.0
_BAD_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_RETRY_DELAYS = (2.0, 4.0, 8.0)
_PROBE_TIMEOUT = httpx.Timeout(6.0, connect=6.0)
_SCORE_BYTES = 512 * 1024
_SCORE_SECONDS = 2.0
_PAR_SPAN = 256 * 1024
_PAR_SECONDS = 1.0


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
    urls: list[str] = field(default_factory=list)
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
    source: str = ""
    file_progress: dict[str, int] = field(default_factory=dict)
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
            "source": self.source,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class DownloadManager:
    def __init__(self, settings: Settings, index: DownloadIndex | None = None) -> None:
        self.settings = settings
        self.index = index or DownloadIndex(settings.download_index_path)
        self._jobs: dict[str, DownloadJob] = {}
        self._lock = threading.Lock()
        self._progress_lock = threading.Lock()
        self._shared_client = _client()
        self._parallel_pref: dict[str, bool] = {}
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
        alternates: list[str] | None = None,
    ) -> DownloadJob:
        if not url.lower().startswith(("http://", "https://")):
            raise DownloadError("只支持 http / https 链接")

        name = sanitize_filename(filename, strict=True) if filename else sanitize_filename(url)
        if name.lower().endswith(".part"):
            # 从「未完成」行点继续时传进来的可能是 x.safetensors.part，目标是 x.safetensors
            name = name[: -len(".part")]
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

        urls = [url, *[u for u in (alternates or []) if u and u != url]]
        job = DownloadJob(
            id=uuid.uuid4().hex[:12],
            filename=name,
            url=url,
            dest=target,
            urls=urls,
            sha256_expected=sha256.lower() if sha256 else None,
        )
        with self._lock:
            self._jobs[job.id] = job
        self.index.remember(name, url, job.sha256_expected)

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
        self._shared_client.close()

    # ------------------------------------------------------------------ 单文件
    def _run_file(self, job: DownloadJob) -> None:
        part = job.dest.with_name(job.dest.name + ".part")
        try:
            if job.cancel_event.is_set():
                raise _Cancelled
            job.state = "running"
            job.started_at = time.time()

            self._download_file(self._shared_client, job, job.urls, part, job.total_bytes, prior=0)

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

            pending: list[dict] = []
            for item in job.files:
                target = self._repo_target(job, item["path"])
                part = target.with_name(target.name + ".part")
                size = item.get("size")
                if target.exists() and size is not None and target.stat().st_size == size:
                    job.file_progress[item["path"]] = target.stat().st_size
                    job.files_done += 1
                else:
                    pending.append(item)
                    job.file_progress[item["path"]] = part.stat().st_size if part.exists() else 0
            job.bytes_done = sum(job.file_progress.values())

            workers = max(1, min(4, len(pending), self.settings.download_connections))
            errors: list[Exception] = []

            def run_one(item: dict) -> None:
                if job.cancel_event.is_set():
                    return
                try:
                    target = self._repo_target(job, item["path"])
                    part = target.with_name(target.name + ".part")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    job.current_file = item["path"]
                    urls = item.get("urls") or [item["url"]]
                    self._download_file(
                        self._shared_client,
                        job,
                        urls,
                        part,
                        item.get("size"),
                        progress_key=item["path"],
                    )
                    os.replace(part, target)
                except Exception as exc:  # noqa: BLE001 —— 收集后在主线程抛出
                    errors.append(exc)

                with self._progress_lock:
                    job.files_done += 1

            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="repo-file") as pool:
                list(pool.map(run_one, pending))

            if errors:
                raise errors[0]
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

    # ------------------------------------------------------------------ 调度
    def _download_file(
        self,
        client: httpx.Client,
        job: DownloadJob,
        urls: list[str],
        part: Path,
        expected_total: int | None,
        prior: int = 0,
        progress_key: str | None = None,
    ) -> None:
        """把一份文件下到 part：优先分段并行，失败时换源重试。"""
        ordered = self._rank_sources(client, urls, expected_total)
        last_error: Exception | None = None

        for index, url in enumerate(ordered):
            job.source = urlparse(url).netloc
            try:
                if expected_total is not None and expected_total < self.settings.segment_min_bytes:
                    # 小文件没必要探测/分段，直接顺流
                    self._stream(client, job, url, part, prior, expected_total, progress_key)
                    return

                total, supports_range = self._probe(client, url)
                if supports_range and total and total >= self.settings.segment_min_bytes:
                    if job.kind == "file":
                        job.total_bytes = total
                    if self._should_parallel(client, url, total):
                        self._segmented(client, job, ordered[index:], part, total, prior, progress_key)
                        return
                self._stream(client, job, url, part, prior, total or expected_total, progress_key)
                return
            except DownloadError as exc:
                last_error = exc
                if index + 1 >= len(ordered):
                    raise
                continue
        if last_error:
            raise last_error

    def _rank_sources(
        self, client: httpx.Client, urls: list[str], expected_total: int | None
    ) -> list[str]:
        """多源时先各测 1MiB，快的排前面。小文件不值得测速，直接用首选。"""
        unique = [u for i, u in enumerate(urls) if u and u not in urls[:i]]
        if len(unique) < 2:
            return unique
        if expected_total is not None and expected_total < self.settings.segment_min_bytes:
            return unique

        # 两个源同时测，别串行等
        def score(url: str) -> float:
            return self._score_source(client, url)

        with ThreadPoolExecutor(max_workers=len(unique)) as pool:
            speeds = list(pool.map(score, unique))
        scored = list(zip(speeds, unique))
        best_speed, best_url = max(scored, key=lambda item: item[0])
        primary_speed = scored[0][0]
        if best_url != unique[0] and best_speed > primary_speed * 1.2:
            rest = [url for _speed, url in sorted(scored, key=lambda item: item[0], reverse=True) if url != best_url]
            return [best_url, *rest]
        return [url for _speed, url in scored]

    def _score_source(self, client: httpx.Client, url: str) -> float:
        started = time.monotonic()
        received = 0
        try:
            with client.stream(
                "GET",
                url,
                headers={"Range": f"bytes=0-{_SCORE_BYTES - 1}"},
                timeout=_PROBE_TIMEOUT,
            ) as response:
                if response.status_code not in {200, 206}:
                    return 0.0
                for chunk in response.iter_bytes(64 * 1024):
                    received += len(chunk)
                    if received >= _SCORE_BYTES or time.monotonic() - started > _SCORE_SECONDS:
                        break
        except httpx.HTTPError:
            return 0.0
        elapsed = max(time.monotonic() - started, 0.001)
        return received / elapsed

    def _probe(self, client: httpx.Client, url: str) -> tuple[int | None, bool]:
        try:
            with client.stream(
                "GET", url, headers={"Range": "bytes=0-0"}, timeout=_PROBE_TIMEOUT
            ) as response:
                if response.status_code == 206:
                    content_range = response.headers.get("content-range", "")
                    tail = content_range.rsplit("/", 1)[-1].strip() if "/" in content_range else ""
                    return (int(tail) if tail.isdigit() else None), True
                if response.status_code == 200:
                    length = response.headers.get("content-length")
                    return (int(length) if length and length.isdigit() else None), False
                return None, False
        except httpx.HTTPError:
            return None, False

    # ------------------------------------------------------------------ 分段
    def _segmented(
        self,
        client: httpx.Client,
        job: DownloadJob,
        urls: list[str],
        part: Path,
        total: int,
        prior: int,
        progress_key: str | None = None,
    ) -> None:
        chunk = max(1, self.settings.download_chunk_bytes)
        count = (total + chunk - 1) // chunk
        meta_path = self._meta_path(part)
        done = self._load_meta(meta_path, part, total, chunk, count)

        base = sum(self._chunk_span(i, total, chunk) for i in done)
        with self._progress_lock:
            if progress_key is not None:
                job.file_progress[progress_key] = base
                job.bytes_done = sum(job.file_progress.values())
            else:
                job.bytes_done = prior + base

        part.parent.mkdir(parents=True, exist_ok=True)
        if not meta_path.exists():
            # 先落一个空账本：哪怕刚开下就被取消，重启后也能看出这是分片任务
            meta_path.write_text(
                json.dumps({"total": total, "chunk": chunk, "done": []}), encoding="utf-8"
            )
        if not part.exists():
            part.touch()
        with open(part, "r+b") as handle:
            try:
                handle.truncate(total)
            except OSError:
                pass
            fd = handle.fileno()
            queue: deque[int] = deque(i for i in range(count) if i not in done)
            lock = threading.Lock()

            def worker() -> None:
                while True:
                    with lock:
                        if job.cancel_event.is_set() or not queue:
                            return
                        index = queue.popleft()
                    start = index * chunk
                    size = self._chunk_span(index, total, chunk)
                    self._fetch_chunk(client, job, urls, fd, start, size, progress_key=progress_key)
                    with lock:
                        _mark_done(meta_path, part, total, chunk, index)

            workers = min(self.settings.download_connections, max(1, len(queue)))
            threads = [threading.Thread(target=worker, name=f"seg-{i}") for i in range(workers)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            if job.cancel_event.is_set():
                raise _Cancelled
            try:
                os.fsync(fd)
            except OSError:
                pass

        meta_path.unlink(missing_ok=True)

    def _fetch_chunk(
        self,
        client: httpx.Client,
        job: DownloadJob,
        urls: list[str],
        fd: int,
        start: int,
        size: int,
        max_attempts: int = 4,
        progress_key: str | None = None,
    ) -> None:
        end = start + size - 1
        headers = {"Range": f"bytes={start}-{end}"}
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            for url in urls:
                if job.cancel_event.is_set():
                    raise _Cancelled
                try:
                    with client.stream("GET", url, headers=headers) as response:
                        if response.status_code in _RETRYABLE_STATUS:
                            last_error = _RetryableStatus(response.status_code)
                            continue
                        if response.status_code != 206:
                            last_error = DownloadError(f"分片请求返回 HTTP {response.status_code}")
                            continue
                        written = 0
                        for block in response.iter_bytes(256 * 1024):
                            if job.cancel_event.is_set():
                                raise _Cancelled
                            if not block:
                                continue
                            if written + len(block) > size:
                                block = block[: size - written]
                            os.pwrite(fd, block, start + written)
                            written += len(block)
                            self._tick(job, len(block), progress_key)
                            if written >= size:
                                break
                        if written < size:
                            last_error = DownloadError("分片提前结束")
                            continue
                        return
                except httpx.HTTPError as exc:
                    last_error = exc
                    continue
            if attempt < max_attempts - 1:
                time.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])

        if isinstance(last_error, _RetryableStatus):
            raise DownloadError(f"服务器限流或暂时不可用（HTTP {last_error.status}），请稍后再试")
        raise DownloadError(f"分片下载失败：{last_error}")

    def _should_parallel(self, client: httpx.Client, url: str, total: int) -> bool:
        """默认每条连接速度不同：先测单连接和 4 并发的实际带宽，再决定是否分段。"""
        mode = self.settings.segment_mode
        if mode == "always":
            return True
        if mode == "never":
            return False

        host = urlparse(url).netloc.lower()
        cached = self._parallel_pref.get(host)
        if cached is not None:
            return cached

        single = self._score_source(client, url)
        if single <= 0:
            return False

        span = _PAR_SPAN
        starts = [i * span for i in range(4)]

        def score(start: int) -> float:
            return self._score_range(client, url, start, span)

        with ThreadPoolExecutor(max_workers=4) as pool:
            parallel = sum(pool.map(score, starts))

        prefer = parallel > single * 1.15
        self._parallel_pref[host] = prefer
        return prefer

    def _score_range(self, client: httpx.Client, url: str, start: int, length: int) -> float:
        started = time.monotonic()
        received = 0
        end = start + length - 1
        try:
            with client.stream(
                "GET",
                url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=_PROBE_TIMEOUT,
            ) as response:
                if response.status_code not in {200, 206}:
                    return 0.0
                for chunk in response.iter_bytes(64 * 1024):
                    received += len(chunk)
                    if received >= length or time.monotonic() - started > _PAR_SECONDS:
                        break
        except httpx.HTTPError:
            return 0.0
        return received / max(time.monotonic() - started, 0.001)

    def _legacy_prefix(self, part: Path, total: int, chunk: int, count: int) -> set[int]:
        """没有分片元数据的老 .part：只有连续写满的才敢续，稀疏文件一律重下。"""
        if not part.exists():
            return set()
        size = part.stat().st_size
        if size <= 0 or size > total:
            part.unlink(missing_ok=True)
            return set()

        allocated = _allocated_bytes(part)
        if allocated < size * 0.95:
            # 稀疏文件：不知道哪些块真的写了，重下更安全
            part.unlink(missing_ok=True)
            return set()

        completed = count if size == total else size // chunk
        return set(range(completed))

    def _chunk_span(self, index: int, total: int, chunk: int) -> int:
        return min(chunk, total - index * chunk)

    def _meta_path(self, part: Path) -> Path:
        return Path(str(part) + ".json")

    def _load_meta(self, meta_path: Path, part: Path, total: int, chunk: int, count: int) -> set[int]:
        if not meta_path.is_file():
            return self._legacy_prefix(part, total, chunk, count)
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            if raw.get("total") != total or raw.get("chunk") != chunk:
                part.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                return set()
            return {int(i) for i in raw.get("done", []) if 0 <= int(i) < count}
        except (OSError, ValueError):
            part.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return set()

    # ------------------------------------------------------------------ 单流
    def _stream(
        self,
        client: httpx.Client,
        job: DownloadJob,
        url: str,
        part: Path,
        prior: int,
        total_hint: int | None,
        progress_key: str | None = None,
    ) -> None:
        offset = part.stat().st_size if part.exists() else 0
        with self._progress_lock:
            if progress_key is not None:
                job.file_progress[progress_key] = offset
                job.bytes_done = sum(job.file_progress.values())
            else:
                job.bytes_done = prior + offset

        for attempt in range(len(_RETRY_DELAYS) + 1):
            try:
                self._stream_once(
                    client, job, url, part, offset, total_hint,
                    progress_key=progress_key,
                    prior=prior,
                )
                return
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

    def _stream_once(
        self,
        client: httpx.Client,
        job: DownloadJob,
        url: str,
        part: Path,
        offset: int,
        total_hint: int | None,
        progress_key: str | None = None,
        prior: int = 0,
    ) -> None:
        headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
        if offset:
            headers["Range"] = f"bytes={offset}-"

        with client.stream("GET", url, headers=headers) as response:
            if response.status_code == 416 and offset:
                raise DownloadError("续传范围无效，请删除 .part 文件后重试")
            if offset and response.status_code == 200:
                with self._progress_lock:
                    job.bytes_done -= offset  # 服务器不认 Range，从头重下
                offset = 0
            if response.status_code in _RETRYABLE_STATUS:
                raise _RetryableStatus(response.status_code)
            if response.status_code not in (200, 206):
                raise DownloadError(f"服务器返回 HTTP {response.status_code}")

            length = response.headers.get("content-length")
            remaining = int(length) if length and length.isdigit() else None
            total = (remaining + offset) if remaining is not None else total_hint
            if total is not None and job.kind == "file":
                job.total_bytes = total
            if remaining is not None:
                self._ensure_space(job, remaining)

            mode = "ab" if offset else "wb"
            part.parent.mkdir(parents=True, exist_ok=True)
            with open(part, mode) as handle:
                for block in response.iter_bytes(CHUNK_SIZE):
                    if job.cancel_event.is_set():
                        raise _Cancelled
                    if not block:
                        continue
                    handle.write(block)
                    self._tick(job, len(block), progress_key)

    # ------------------------------------------------------------------ 进度
    def _tick(self, job: DownloadJob, nbytes: int, key: str | None = None) -> None:
        now = time.monotonic()
        with self._progress_lock:
            if key is not None:
                job.file_progress[key] = job.file_progress.get(key, 0) + nbytes
                job.bytes_done = sum(job.file_progress.values())
            else:
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


def _mark_done(meta_path: Path, part: Path, total: int, chunk: int, index: int) -> None:
    """把一个分片标记完成，原子写回 meta。"""
    done: list[int] = []
    if meta_path.is_file():
        try:
            done = [int(i) for i in json.loads(meta_path.read_text(encoding="utf-8")).get("done", [])]
        except (OSError, ValueError):
            done = []
    if index not in done:
        done.append(index)
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp.write_text(json.dumps({"total": total, "chunk": chunk, "done": sorted(done)}), encoding="utf-8")
    os.replace(tmp, meta_path)


def _client() -> httpx.Client:
    """直连客户端：trust_env=False 保证任何代理环境变量都不会被采用。"""
    return httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(30.0, read=120.0),
        headers={"User-Agent": USER_AGENT},
        trust_env=False,
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=32, keepalive_expiry=30.0),
    )


def _allocated_bytes(path: Path) -> int:
    try:
        stat = path.stat()
        blocks = getattr(stat, "st_blocks", None)
        if blocks is None:
            return stat.st_size
        return blocks * 512
    except OSError:
        return 0


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
