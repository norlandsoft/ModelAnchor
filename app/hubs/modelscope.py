"""ModelScope 检索与文件列表。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import httpx

from .base import HubError, RepoFile, RepoInfo, SearchHit, TTLCache, classify

USER_AGENT = "ModelAnchor/0.1"
MAX_DIRS = 60
MAX_DEPTH = 4


class ModelScopeHub:
    name = "modelscope"
    label = "ModelScope"

    def __init__(self, base: str = "https://modelscope.cn", timeout: float = 20.0) -> None:
        self.base = base.rstrip("/")
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"},
            trust_env=False,
        )
        self._cache = TTLCache(ttl=600)

    # ------------------------------------------------------------------ 搜索
    def search(self, query: str, limit: int = 12, fmt: str | None = None) -> list[SearchHit]:
        cache_key = f"search:{query}|{limit}"
        cached = self._cache.get(cache_key)
        if isinstance(cached, list):
            return cached

        limit = max(1, min(limit, 30))
        try:
            response = self._client.put(
                f"{self.base}/api/v1/dolphin/models",
                json={"PageSize": limit, "PageNumber": 1, "Name": query, "SortBy": "Default"},
            )
        except httpx.TimeoutException as exc:
            raise HubError("响应超时，稍后重试") from exc
        except httpx.HTTPError as exc:
            raise HubError(f"连接失败：{exc}") from exc
        response.raise_for_status()
        payload = response.json()
        if payload.get("Code") != 200:
            raise HubError(payload.get("Message") or "ModelScope 搜索失败")
        models = (payload.get("Data") or {}).get("Model", {}).get("Models") or []

        hits: list[SearchHit] = []
        for model in models:
            owner = model.get("Path") or ""
            name = model.get("Name") or ""
            if not owner or not name:
                continue
            repo_id = f"{owner}/{name}"
            tasks = model.get("Tasks") or []
            tags = [t.get("Name", "") for t in tasks if isinstance(t, dict)]
            tags += [lib for lib in (model.get("Libraries") or []) if lib not in {"pytorch", "safetensors"}]
            self._cache.set(f"tags:{repo_id}", tags)
            hits.append(
                SearchHit(
                    source=self.name,
                    repo_id=repo_id,
                    downloads=int(model.get("Downloads") or 0),
                    likes=int(model.get("Stars") or 0),
                    pipeline_tag=tasks[0].get("Name", "") if tasks and isinstance(tasks[0], dict) else "",
                    updated_at=float(model["LastUpdatedTime"]) if model.get("LastUpdatedTime") else None,
                    layout="unknown",
                    tags=tags[:6],
                    page_url=self.page_url(repo_id),
                )
            )

        # 搜索结果本身不带文件树，这里并发补一次目录探测，才能按"单文件 / diffusers"过滤
        if hits:
            with ThreadPoolExecutor(max_workers=min(6, len(hits))) as pool:
                list(pool.map(self._fill_layout, hits))
        self._cache.set(cache_key, hits)
        return hits

    def _fill_layout(self, hit: SearchHit) -> None:
        """只列根目录做布局判断：一次请求，比整棵树快一个数量级。"""
        owner, _, name = hit.repo_id.partition("/")
        if not name:
            return
        try:
            entries = self._list_dir(owner, name, "")
        except Exception:  # noqa: BLE001 —— 探测失败就保持 unknown，不影响搜索
            return

        # 目录补上斜杠，复用统一的 classify 规则
        paths = []
        for entry in entries:
            path = str(entry.get("Path") or "")
            if not path:
                continue
            paths.append(path + "/" if entry.get("Type") == "tree" else path)

        layout, _ = classify(paths, tags=hit.tags, repo_id=hit.repo_id)
        hit.layout = layout
        hit.files = [p for p in paths if not p.endswith("/")]
        if layout in {"single", "lora"}:
            weights = [
                e for e in entries
                if e.get("Type") != "tree"
                and str(e.get("Path", "")).lower().endswith((".safetensors", ".ckpt", ".gguf"))
            ]
            hit.total_bytes = sum(int(e.get("Size") or 0) for e in weights)

    # -------------------------------------------------------------- 仓库详情
    def repo_info(self, repo_id: str, refresh: bool = False) -> RepoInfo:
        key = f"ms:{repo_id}"
        if not refresh:
            cached = self._cache.get(key)
            if isinstance(cached, RepoInfo):
                return cached

        files = self._walk(repo_id)
        layout, _ = classify([f.path for f in files], tags=self._tags(repo_id), repo_id=repo_id)
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

    def _tags(self, repo_id: str) -> list[str]:
        """仓库标签（Libraries / Tags）。搜索时已经拿到就复用，否则补一次元数据请求。"""
        cached = self._cache.get(f"tags:{repo_id}")
        if cached is not None:
            return list(cached)

        owner, _, name = repo_id.partition("/")
        tags: list[str] = []
        try:
            response = self._client.get(f"{self.base}/api/v1/models/{owner}/{name}")
            data = (response.json() or {}).get("Data") or {}
            tags = [str(lib) for lib in (data.get("Libraries") or [])]
            raw_tags = data.get("Tags") or []
            if isinstance(raw_tags, list):
                tags += [t if isinstance(t, str) else str(t.get("Name", "")) for t in raw_tags]
        except Exception:  # noqa: BLE001 —— 标签拿不到不影响文件列表
            tags = []
        tags = [t for t in tags if t]
        self._cache.set(f"tags:{repo_id}", tags)
        return tags

    def _walk(self, repo_id: str) -> list[RepoFile]:
        owner, _, name = repo_id.partition("/")
        if not name:
            raise HubError(f"仓库名不合法：{repo_id}")

        files: list[RepoFile] = []
        queue: list[tuple[str, int]] = [("", 0)]
        seen_dirs = 0
        while queue:
            root, depth = queue.pop(0)
            for entry in self._list_dir(owner, name, root):
                path = entry.get("Path") or ""
                kind = entry.get("Type")
                if not path:
                    continue
                if kind == "tree":
                    if depth < MAX_DEPTH and seen_dirs < MAX_DIRS:
                        seen_dirs += 1
                        queue.append((path, depth + 1))
                else:
                    size = entry.get("Size")
                    files.append(RepoFile(path, int(size) if size else None))
        if not files:
            raise HubError("没有读到文件列表（仓库可能未公开或需要登录）")
        return files

    def _list_dir(self, owner: str, name: str, root: str) -> list[dict]:
        try:
            response = self._client.get(
                f"{self.base}/api/v1/models/{owner}/{name}/repo/files",
                params={"Revision": "master", "Root": root},
            )
        except httpx.TimeoutException as exc:
            raise HubError("响应超时，稍后重试") from exc
        except httpx.HTTPError as exc:
            raise HubError(f"连接失败：{exc}") from exc
        response.raise_for_status()
        payload = response.json()
        if payload.get("Code") != 200:
            raise HubError(payload.get("Message") or "读取目录失败")
        return (payload.get("Data") or {}).get("Files") or []

    # ------------------------------------------------------------------ 地址
    def file_url(self, repo_id: str, path: str) -> str:
        return f"{self.base}/models/{repo_id}/resolve/master/{quote(path, safe='/')}"

    def page_url(self, repo_id: str) -> str:
        return f"{self.base}/models/{repo_id}"
