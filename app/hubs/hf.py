"""HuggingFace（走镜像站）检索与文件列表。"""

from __future__ import annotations

import time
from urllib.parse import quote

import httpx

from .base import HubError, RepoFile, RepoInfo, SearchHit, TTLCache, classify

USER_AGENT = "ModelAnchor/0.1"


class HuggingFaceHub:
    name = "hf"
    label = "HuggingFace 镜像"

    def __init__(self, base: str, timeout: float = 20.0) -> None:
        self.base = base.rstrip("/")
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        self._cache = TTLCache(ttl=600)

    def _get(self, path: str, params: dict | None = None, attempts: int = 3):
        delay = 1.0
        for attempt in range(attempts):
            response = self._client.get(f"{self.base}{path}", params=params)
            if response.status_code in {429, 500, 502, 503, 504}:
                if attempt + 1 < attempts:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise HubError(f"HuggingFace 镜像暂时不可用（HTTP {response.status_code}），请稍后重试")
            if response.status_code == 404:
                raise HubError("仓库不存在，或已被作者删除")
            if response.status_code in {401, 403}:
                raise HubError("这个仓库需要登录或授权，无法直接下载")
            response.raise_for_status()
            return response
        raise HubError("请求失败")

    # ------------------------------------------------------------------ 搜索
    def search(self, query: str, limit: int = 12) -> list[SearchHit]:
        response = self._get(
            "/api/models",
            params={
                "search": query,
                "limit": max(1, min(limit, 50)),
                "full": "true",
                "sort": "downloads",
                "direction": "-1",
            },
        )
        items = response.json()
        hits: list[SearchHit] = []
        for item in items:
            repo_id = item.get("id") or item.get("modelId")
            if not repo_id:
                continue
            paths = [s.get("rfilename", "") for s in item.get("siblings", [])]
            layout, models = classify(paths, tags=item.get("tags") or [], repo_id=repo_id)
            hits.append(
                SearchHit(
                    source=self.name,
                    repo_id=repo_id,
                    downloads=int(item.get("downloads") or 0),
                    likes=int(item.get("likes") or 0),
                    pipeline_tag=item.get("pipeline_tag") or "",
                    updated_at=_parse_date(item.get("lastModified") or item.get("createdAt")),
                    layout=layout,
                    tags=_pipeline_tags(item),
                    page_url=self.page_url(repo_id),
                    total_bytes=sum(m.size or 0 for m in models),
                )
            )
        return hits

    # -------------------------------------------------------------- 仓库详情
    def repo_info(self, repo_id: str, refresh: bool = False) -> RepoInfo:
        key = f"hf:{repo_id}"
        if not refresh:
            cached = self._cache.get(key)
            if isinstance(cached, RepoInfo):
                return cached
        response = self._get(f"/api/models/{repo_id}", params={"blobs": "true"})
        payload = response.json()
        files: list[RepoFile] = []
        for sibling in payload.get("siblings", []):
            path = sibling.get("rfilename")
            if not path:
                continue
            size = sibling.get("size")
            if size is None and isinstance(sibling.get("lfs"), dict):
                size = sibling["lfs"].get("size")
            files.append(RepoFile(path, int(size) if size is not None else None))

        layout, _ = classify(
            [f.path for f in files], tags=payload.get("tags") or [], repo_id=repo_id
        )
        model_files = [f for f in files if f.path.lower().endswith((".safetensors", ".ckpt", ".gguf"))]
        info = RepoInfo(
            source=self.name,
            repo_id=repo_id,
            files=files,
            layout=layout,
            model_files=model_files,
            page_url=self.page_url(repo_id),
        )
        self._cache.set(key, info)
        return info

    # ------------------------------------------------------------------ 地址
    def file_url(self, repo_id: str, path: str) -> str:
        return f"{self.base}/{repo_id}/resolve/main/{quote(path, safe='/')}"

    def page_url(self, repo_id: str) -> str:
        return f"{self.base}/{repo_id}"


def _parse_date(value) -> float | None:
    if not value:
        return None
    from datetime import datetime

    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _pipeline_tags(item: dict) -> list[str]:
    tags = item.get("tags") or []
    keep = []
    for tag in tags:
        if tag.startswith(("base_model:", "license:", "region:", "arxiv:", "dataset:")):
            continue
        if tag in {"transformers", "diffusers", "gguf", "safetensors", "pytorch"}:
            continue
        keep.append(tag)
    return keep[:6]
