"""JSON API。前端只依赖这些接口，换前端不用动后端。"""

from __future__ import annotations

import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from . import catalog as catalog_mod
from . import library
from .config import APP_NAME, VERSION
from .downloader import DownloadError, DownloadManager, FileExists, JobConflict
from .hubs import HubError, is_downloadable
from .merger import MergeManager, build_plan
from .schemas import (
    CatalogEntry,
    CatalogResponse,
    DownloadJobOut,
    DownloadRequest,
    DownloadsResponse,
    HealthResponse,
    LibraryResponse,
    MergeComponentOut,
    MergePlanOut,
    MergeRequest,
    ModelFile,
    OkResponse,
    RepoDownloadRequest,
    RepoFileOut,
    RepoInfoOut,
    SearchHitOut,
    SearchResponse,
    TasksResponse,
)

router = APIRouter(prefix="/api", tags=["modelanchor"])


def _downloads(request: Request) -> DownloadManager:
    return request.app.state.manager


def _merges(request: Request) -> MergeManager:
    return request.app.state.merges


# --------------------------------------------------------------------- 基础
@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    return HealthResponse(
        app=APP_NAME,
        version=VERSION,
        status="ok",
        model_dir=str(settings.model_dir),
        port=settings.port,
        uptime_seconds=round(time.time() - request.app.state.started_at, 1),
    )


@router.get("/library", response_model=LibraryResponse)
def get_library(request: Request) -> LibraryResponse:
    settings = request.app.state.settings
    model_dir: Path = settings.model_dir
    files = library.scan(model_dir)
    free, total = library.disk_usage(model_dir)
    return LibraryResponse(
        model_dir=str(model_dir),
        model_dir_source=settings.model_dir_source,
        hf_base=settings.hf_base,
        exists=model_dir.is_dir(),
        writable=_is_writable(model_dir),
        warning=settings.warning,
        disk_free_bytes=free,
        disk_total_bytes=total,
        models=[
            ModelFile(
                filename=m.filename,
                size_bytes=m.size_bytes,
                modified_at=m.modified_at,
                kind=m.kind,
                partial=m.partial,
                is_dir=m.is_dir,
            )
            for m in files
        ],
    )


@router.get("/catalog", response_model=CatalogResponse)
def get_catalog(request: Request) -> CatalogResponse:
    settings = request.app.state.settings
    installed = {
        m.filename: m.size_bytes for m in library.scan(settings.model_dir) if not m.partial
    }
    entries = catalog_mod.load(settings.catalog_path, settings.hf_base)
    return CatalogResponse(
        base=settings.hf_base,
        entries=[
            CatalogEntry(
                id=e.id,
                name=e.name,
                filename=e.filename,
                kind=e.kind,
                repo=e.repo,
                url=e.url,
                size_bytes=e.size_bytes,
                note=e.note,
                installed=e.filename in installed,
                installed_bytes=installed.get(e.filename),
            )
            for e in entries
        ],
    )


# --------------------------------------------------------------------- 检索
@router.get("/search", response_model=SearchResponse)
def search(
    request: Request,
    q: str,
    source: str = "all",
    layout: str = "all",
    limit: int = 10,
) -> SearchResponse:
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="请输入检索关键词")
    if layout not in {"all", "single", "diffusers", "lora", "unknown"}:
        raise HTTPException(
            status_code=400, detail="layout 只能是 all / single / diffusers / lora / unknown"
        )

    hubs = request.app.state.hubs
    if source == "all":
        selected = list(hubs.values())
    elif source in hubs:
        selected = [hubs[source]]
    else:
        raise HTTPException(status_code=400, detail=f"未知来源：{source}")

    # 按布局过滤是在站点返回之后做的：多取一些候选，避免前排被过滤光
    fetch_limit = limit if layout == "all" else min(limit * 3, 50)

    results: list[SearchHitOut] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        futures = {pool.submit(h.search, query, fetch_limit): h for h in selected}
        done, pending = wait(futures, timeout=40)
        for future in done:
            hub = futures[future]
            try:
                for hit in future.result():
                    results.append(SearchHitOut(**hit.__dict__))
            except Exception as exc:  # noqa: BLE001 —— 单个站点失败不影响另一个
                errors.append(f"{hub.label}：{exc}")
        for future in pending:
            errors.append(f"{futures[future].label}：请求超时")

    if layout != "all":
        results = [r for r in results if r.layout == layout]
    results.sort(key=lambda r: r.downloads, reverse=True)
    if layout != "all":
        results = results[: limit * 2]
    return SearchResponse(results=results, errors=errors)


@router.get("/repos", response_model=RepoInfoOut)
def repo_info(
    request: Request,
    source: str,
    repo_id: str,
    refresh: bool = False,
) -> RepoInfoOut:
    hub = _hub(request, source)
    try:
        info = hub.repo_info(repo_id, refresh=refresh)
    except HubError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"读取仓库失败：{exc}") from exc

    files = [
        RepoFileOut(
            path=f.path,
            size=f.size,
            downloadable=is_downloadable(f.path),
            url=hub.file_url(info.repo_id, f.path),
        )
        for f in info.files
    ]
    return RepoInfoOut(
        source=info.source,
        repo_id=info.repo_id,
        layout=info.layout,
        page_url=info.page_url,
        total_bytes=sum(f.size or 0 for f in info.files),
        file_count=len(info.files),
        files=files,
    )


# --------------------------------------------------------------------- 下载
@router.post("/downloads", response_model=DownloadJobOut, status_code=202)
def create_download(payload: DownloadRequest, request: Request) -> DownloadJobOut:
    settings = request.app.state.settings

    if payload.catalog_id:
        entry = catalog_mod.find(
            catalog_mod.load(settings.catalog_path, settings.hf_base), payload.catalog_id
        )
        if entry is None:
            raise HTTPException(status_code=404, detail="目录里没有这个条目")
        url = entry.url
        filename = payload.filename or entry.filename
        sha256 = payload.sha256 or entry.sha256
    else:
        if not payload.url:
            raise HTTPException(status_code=400, detail="需要 url 或 catalog_id")
        if not settings.allow_arbitrary_url:
            raise HTTPException(status_code=403, detail="已禁用任意 URL 下载")
        url, filename, sha256 = payload.url, payload.filename, payload.sha256

    try:
        job = _downloads(request).start(url=url, filename=filename, sha256=sha256)
    except (FileExists, JobConflict) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DownloadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DownloadJobOut(**job.to_dict())


@router.post("/downloads/repo", response_model=DownloadJobOut, status_code=202)
def create_repo_download(payload: RepoDownloadRequest, request: Request) -> DownloadJobOut:
    hub = _hub(request, payload.source)
    try:
        info = hub.repo_info(payload.repo_id, refresh=payload.refresh)
    except HubError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"读取仓库失败：{exc}") from exc

    files = [f for f in info.files if is_downloadable(f.path)]
    if not files:
        raise HTTPException(status_code=400, detail="仓库里没有可下载的文件")

    items = [
        {"path": f.path, "url": hub.file_url(info.repo_id, f.path), "size": f.size} for f in files
    ]
    try:
        job = _downloads(request).start_repo(info.repo_id, items, source=payload.source)
    except (FileExists, JobConflict) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DownloadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DownloadJobOut(**job.to_dict())


@router.get("/downloads", response_model=DownloadsResponse)
def list_downloads(request: Request) -> DownloadsResponse:
    jobs = [DownloadJobOut(**job.to_dict()) for job in _downloads(request).list()]
    return DownloadsResponse(jobs=jobs)


@router.get("/tasks", response_model=TasksResponse)
def list_tasks(request: Request) -> TasksResponse:
    tasks = [DownloadJobOut(**job.to_dict()) for job in _downloads(request).list()]
    tasks += [DownloadJobOut(**job.to_dict()) for job in _merges(request).list()]
    tasks.sort(key=lambda t: t.created_at)
    return TasksResponse(tasks=tasks)


@router.post("/tasks/{task_id}/cancel", response_model=DownloadJobOut)
def cancel_task(task_id: str, request: Request) -> DownloadJobOut:
    job = _find_task(request, task_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    job.cancel_event.set()
    return DownloadJobOut(**job.to_dict())


@router.delete("/tasks/{task_id}", response_model=OkResponse)
def forget_task(task_id: str, request: Request) -> OkResponse:
    for manager in (_downloads(request), _merges(request)):
        job = manager.get(task_id)
        if job is not None:
            if job.state in {"queued", "running", "verifying"}:
                raise HTTPException(status_code=409, detail="任务未结束，先取消它")
            manager.forget(task_id)
            return OkResponse(detail="已移除任务记录")
    raise HTTPException(status_code=404, detail="任务不存在")


@router.post("/downloads/{job_id}/cancel", response_model=DownloadJobOut)
def cancel_download(job_id: str, request: Request) -> DownloadJobOut:
    try:
        job = _downloads(request).cancel(job_id)
    except DownloadError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DownloadJobOut(**job.to_dict())


@router.delete("/downloads/{job_id}", response_model=OkResponse)
def forget_download(job_id: str, request: Request) -> OkResponse:
    try:
        _downloads(request).forget(job_id)
    except DownloadError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return OkResponse(detail="已移除任务记录")


# --------------------------------------------------------------------- 合并
@router.get("/merges/plan", response_model=MergePlanOut)
def merge_plan(request: Request, source_dir: str, arch: str = "auto") -> MergePlanOut:
    settings = request.app.state.settings
    try:
        target = library.resolve_target(settings.model_dir, source_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        plan = build_plan(target, arch)
    except DownloadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return MergePlanOut(
        source_dir=source_dir,
        arch=plan.arch,
        components=[
            MergeComponentOut(
                name=c.name,
                prefix=c.prefix,
                files=[f.name for f in c.files],
                tensor_count=c.tensor_count,
                bytes=c.bytes,
            )
            for c in plan.components
        ],
        total_bytes=plan.total_bytes,
        total_tensors=plan.total_tensors,
    )


@router.post("/merges", response_model=DownloadJobOut, status_code=202)
def create_merge(payload: MergeRequest, request: Request) -> DownloadJobOut:
    try:
        job = _merges(request).start(payload.source_dir, payload.output_name, payload.arch)
    except (FileExists, JobConflict) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DownloadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DownloadJobOut(**job.to_dict())


# --------------------------------------------------------------------- 维护
@router.delete("/library/{filename}", response_model=OkResponse)
def delete_model(filename: str, request: Request) -> OkResponse:
    settings = request.app.state.settings
    try:
        target = library.resolve_target(settings.model_dir, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    try:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"删除失败：{exc}") from exc
    return OkResponse(detail=f"已删除 {filename}")


# --------------------------------------------------------------------- 工具
def _hub(request: Request, source: str):
    hubs = request.app.state.hubs
    if source not in hubs:
        raise HTTPException(status_code=400, detail=f"未知来源：{source}")
    return hubs[source]


def _find_task(request: Request, task_id: str):
    for manager in (_downloads(request), _merges(request)):
        job = manager.get(task_id)
        if job is not None:
            return job
    return None


def _is_writable(path: Path) -> bool:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return probe.is_dir() and os.access(probe, os.W_OK)
