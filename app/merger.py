"""把 diffusers 目录结构合并导出为单个 safetensors。

DrawThings 的 .ckpt 其实是 SQLite 容器（表 `tensors`），格式与 safetensors 完全不同，
这里不做伪造；需要单文件时统一导出标准的 .safetensors。
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .downloader import DownloadError, FileExists, JobConflict, sanitize_filename
from .library import resolve_target
from .safetensors_io import list_tensors, merge_into

# 组件目录 → 单文件 checkpoint 里的键前缀
ARCH_PREFIXES: dict[str, dict[str, str]] = {
    "sd15": {
        "unet": "model.diffusion_model.",
        "vae": "first_stage_model.",
        "text_encoder": "cond_stage_model.transformer.",
    },
    "sdxl": {
        "unet": "model.diffusion_model.",
        "vae": "first_stage_model.",
        "text_encoder": "conditioner.embedders.0.transformer.",
        "text_encoder_2": "conditioner.embedders.1.model.",
    },
    "flux": {
        "transformer": "model.diffusion_model.",
        "vae": "first_stage_model.",
        "text_encoder": "cond_stage_model.transformer.",
        "text_encoder_2": "cond_stage_model.2.transformer.",
        "text_encoder_3": "cond_stage_model.3.transformer.",
        "safety_checker": "safety_checker.",
    },
    "generic": {},
}
ARCH_PREFIXES["sd3"] = ARCH_PREFIXES["flux"]

ARCH_LABELS = {
    "auto": "自动识别",
    "sd15": "Stable Diffusion 1.x",
    "sdxl": "SDXL / SD 2.x",
    "flux": "Flux / SD3 系列",
    "sd3": "SD3（与 Flux 同前缀）",
    "generic": "通用（组件目录名前缀）",
}

# 合并顺序：主模型在前，其它跟随；遇到未知组件按字母序补在后面
COMPONENT_ORDER = ["unet", "transformer", "vae", "text_encoder", "text_encoder_2", "text_encoder_3"]


@dataclass
class MergeComponent:
    name: str
    prefix: str
    files: list[Path]
    tensor_count: int
    bytes: int


@dataclass
class MergePlan:
    source_dir: Path
    arch: str
    components: list[MergeComponent]
    total_bytes: int
    total_tensors: int


@dataclass
class MergeJob:
    id: str
    filename: str
    source_dir: str
    state: str = "queued"
    arch: str = "auto"
    bytes_done: int = 0
    total_bytes: int | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> dict:
        percent = None
        if self.total_bytes:
            percent = round(min(self.bytes_done / self.total_bytes, 1.0) * 100, 1)
        return {
            "id": self.id,
            "kind": "merge",
            "filename": self.filename,
            "url": "",
            "source": self.source_dir,
            "state": self.state,
            "bytes_done": self.bytes_done,
            "total_bytes": self.total_bytes,
            "percent": percent,
            "speed_bps": 0.0,
            "eta_seconds": None,
            "error": self.error,
            "files_done": 0,
            "files_total": 1,
            "current_file": "",
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def detect_arch(source_dir: Path) -> str:
    names = {p.name.lower() for p in source_dir.iterdir() if p.is_dir()}
    if "unet" in names:
        return "sdxl" if "text_encoder_2" in names else "sd15"
    if "transformer" in names:
        return "flux"
    return "generic"


def _prefix_for(arch: str, component: str) -> str:
    return ARCH_PREFIXES.get(arch, {}).get(component, f"{component}.")


def _component_shards(component_dir: Path) -> list[Path]:
    index = component_dir / "model.safetensors.index.json"
    if index.is_file():
        try:
            import json

            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
            names = sorted(set(weight_map.values()))
            shards = [component_dir / name for name in names]
            if shards and all(p.is_file() for p in shards):
                return shards
        except (OSError, ValueError):
            pass
    return sorted(
        p for p in component_dir.glob("*.safetensors") if p.is_file() and not p.name.endswith(".part")
    )


def build_plan(source_dir: Path, arch: str = "auto") -> MergePlan:
    if not source_dir.is_dir():
        raise DownloadError(f"目录不存在：{source_dir.name}")

    resolved_arch = detect_arch(source_dir) if arch == "auto" else arch
    if resolved_arch not in ARCH_PREFIXES:
        raise DownloadError(f"不支持的架构：{arch}")

    present = {p.name.lower(): p for p in source_dir.iterdir() if p.is_dir()}
    components: list[MergeComponent] = []
    ordered = [name for name in COMPONENT_ORDER if name in present]
    ordered += sorted(name for name in present if name not in COMPONENT_ORDER)

    for name in ordered:
        directory = present[name]
        shards = _component_shards(directory)
        if not shards:
            continue
        tensor_count = 0
        byte_count = 0
        for shard in shards:
            try:
                tensor_count += len(list_tensors(shard))
            except Exception as exc:
                raise DownloadError(f"读取 {shard.name} 失败：{exc}") from exc
            byte_count += shard.stat().st_size
        components.append(
            MergeComponent(
                name=name,
                prefix=_prefix_for(resolved_arch, name),
                files=shards,
                tensor_count=tensor_count,
                bytes=byte_count,
            )
        )

    if not components:
        root_weights = [p for p in source_dir.glob("*.safetensors") if p.is_file()]
        if root_weights:
            raise DownloadError("这个目录里已经是单文件权重，不需要合并")
        raise DownloadError("没有找到 diffusers 组件（unet/transformer、vae、text_encoder…）")

    return MergePlan(
        source_dir=source_dir,
        arch=resolved_arch,
        components=components,
        total_bytes=sum(c.bytes for c in components),
        total_tensors=sum(c.tensor_count for c in components),
    )


class MergeManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._jobs: dict[str, MergeJob] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="modelanchor-merge")

    # ---------------------------------------------------------------- 生命周期
    def start(self, source_name: str, output_name: str | None, arch: str = "auto") -> MergeJob:
        source_dir = resolve_target(self.settings.model_dir, source_name)
        if not source_dir.is_dir():
            raise DownloadError(f"目录不存在：{source_name}")

        plan = build_plan(source_dir, arch)
        out_name = output_name or f"{source_name}.safetensors"
        if not out_name.lower().endswith(".safetensors"):
            raise DownloadError("导出文件名必须以 .safetensors 结尾")
        out_name = sanitize_filename(out_name, strict=True)
        out_path = resolve_target(self.settings.model_dir, out_name)

        free = 0
        try:
            usage = os.statvfs(self.settings.model_dir)
            free = usage.f_bavail * usage.f_frsize
        except OSError:
            pass
        if free and free < plan.total_bytes:
            raise DownloadError(
                f"磁盘空间不足：合并产物约 {plan.total_bytes / 2**30:.1f} GiB，剩余 {free / 2**30:.1f} GiB"
            )

        with self._lock:
            for existing in self._jobs.values():
                if existing.source_dir == source_dir.name and existing.state in {"queued", "running"}:
                    raise JobConflict("这个目录已经在合并中")
            if out_path.exists():
                raise FileExists(f"“{out_name}”已经存在")

        job = MergeJob(
            id=uuid.uuid4().hex[:12],
            filename=out_name,
            source_dir=source_dir.name,
            total_bytes=plan.total_bytes,
            arch=plan.arch,
        )
        with self._lock:
            self._jobs[job.id] = job
        self._pool.submit(self._run, job, plan, out_path)
        return job

    def get(self, job_id: str) -> MergeJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[MergeJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at)

    def cancel(self, job_id: str) -> MergeJob:
        job = self.get(job_id)
        if job is None:
            raise DownloadError("任务不存在")
        job.cancel_event.set()
        return job

    def forget(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise DownloadError("任务不存在")
            if job.state in {"queued", "running"}:
                raise DownloadError("任务未结束，先取消它")
            del self._jobs[job_id]

    def shutdown(self) -> None:
        for job in self.list():
            job.cancel_event.set()
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ 内部
    def _run(self, job: MergeJob, plan: MergePlan, out_path: Path) -> None:
        sources: list[tuple[str, Path]] = []
        for component in plan.components:
            for shard in component.files:
                sources.append((component.prefix, shard))

        last_update = 0.0

        def progress(done: int, total: int) -> None:
            nonlocal last_update
            if job.cancel_event.is_set():
                raise _MergeCancelled
            now = time.monotonic()
            if now - last_update > 0.3 or done >= total:
                job.bytes_done = done
                job.total_bytes = total
                last_update = now

        try:
            job.state = "running"
            job.started_at = time.time()
            merge_into(sources, out_path, on_progress=progress)
            job.state = "done"
            job.bytes_done = job.total_bytes or job.bytes_done
        except _MergeCancelled:
            job.state = "canceled"
            out_path.with_name(out_path.name + ".tmp").unlink(missing_ok=True)
        except DownloadError as exc:
            job.state = "failed"
            job.error = str(exc)
        except Exception as exc:  # noqa: BLE001
            job.state = "failed"
            job.error = f"合并失败：{exc}"
            out_path.with_name(out_path.name + ".tmp").unlink(missing_ok=True)
        finally:
            job.finished_at = time.time()


class _MergeCancelled(Exception):
    pass
