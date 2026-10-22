"""两个模型站点的公共抽象：搜索结果、仓库文件、布局判定。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Protocol


class HubError(RuntimeError):
    """站点返回了可预期的错误（限流、仓库不存在、需要登录等）。"""


WEIGHT_EXTENSIONS = (".safetensors", ".ckpt", ".gguf", ".bin", ".pt", ".pth")

# 下载时跳过的东西：文档、git 元数据、测试与示例
_DOC_NAMES = {"readme.md", "license", "license.md", "notice.md", "notice", ".gitattributes"}

# LoRA 一类的适配器权重：标签、仓库名、文件名任意一处命中都算
LORA_TAGS = {"lora", "lora-adapter", "ic-lora", "lycoris", "locon", "dora", "peft", "adapter"}
LORA_NAME_HINTS = ("lora", "lycoris", "locon", "dora")
_SKIP_PREFIXES = (".git/", ".github/", "tests/", "examples/", "assets/", "_examples/")
_SKIP_SUFFIXES = (".md", ".txt", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")



@dataclass(frozen=True)
class RepoFile:
    path: str
    size: int | None = None


@dataclass
class RepoInfo:
    source: str
    repo_id: str
    files: list[RepoFile]
    layout: str  # single | diffusers | unknown
    model_files: list[RepoFile]
    page_url: str


@dataclass
class SearchHit:
    source: str
    repo_id: str
    downloads: int = 0
    likes: int = 0
    pipeline_tag: str = ""
    updated_at: float | None = None
    layout: str = "unknown"
    tags: list[str] = field(default_factory=list)
    page_url: str = ""
    total_bytes: int = 0


class Hub(Protocol):
    name: str
    label: str

    def search(self, query: str, limit: int) -> list[SearchHit]: ...

    def repo_info(self, repo_id: str, refresh: bool = False) -> RepoInfo: ...

    def file_url(self, repo_id: str, path: str) -> str: ...

    def page_url(self, repo_id: str) -> str: ...


def is_weight(path: str) -> bool:
    return path.lower().endswith(WEIGHT_EXTENSIONS)


def is_downloadable(path: str) -> bool:
    lowered = path.lower()
    if not lowered or lowered.endswith("/"):
        return False
    if lowered.startswith(_SKIP_PREFIXES):
        return False
    if lowered.endswith(_SKIP_SUFFIXES):
        return False
    name = lowered.rsplit("/", 1)[-1]
    return name not in _DOC_NAMES


def looks_like_lora(paths: list[str], tags=(), repo_id: str = "") -> bool:
    """判断是不是 LoRA / LyCORIS 这类适配器仓库。

    标签或仓库名里带 lora 最可靠；退而求其次，只有当**所有**权重文件名都带
    lora 特征时才算（避免把同时含主模型和 lora 的 Comfy 整合包误判）。
    """
    for tag in tags:
        lowered = str(tag).lower()
        if lowered in LORA_TAGS or lowered.startswith(("lora", "base_model:adapter")):
            return True
    if "lora" in repo_id.lower():
        return True

    weights = [p for p in paths if is_weight(p)]
    if not weights:
        return False
    return all(any(hint in p.rsplit("/", 1)[-1].lower() for hint in LORA_NAME_HINTS) for p in weights)


def classify(
    paths: list[str], tags=(), repo_id: str = ""
) -> tuple[str, list[RepoFile]]:
    """判断仓库类型：lora / diffusers / single / unknown。"""
    lowered = [p.lower() for p in paths]
    basenames = {p.rsplit("/", 1)[-1] for p in lowered}

    weights = [RepoFile(p, None) for p in paths if is_weight(p)]

    if weights and looks_like_lora(paths, tags, repo_id):
        return "lora", weights

    if "model_index.json" in basenames:
        return "diffusers", weights
    # 只有 unet/ 或 transformer/ 才是 diffusers 结构；ComfyUI 风格的仓库
    # 把权重放在 diffusion_models/ 里，哪怕同时有 vae/ 目录也不算。
    if any(p.startswith(("unet/", "transformer/")) for p in lowered):
        return "diffusers", weights
    if weights:
        return "single", weights
    return "unknown", []


class TTLCache:
    """很朴素的带过期时间的进程内缓存，够本地工具用。"""

    def __init__(self, ttl: float = 600.0) -> None:
        self.ttl = ttl
        self._data: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            hit = self._data.get(key)
            if not hit:
                return None
            stamp, value = hit
            if time.time() - stamp > self.ttl:
                self._data.pop(key, None)
                return None
            return value

    def set(self, key: str, value) -> None:
        with self._lock:
            self._data[key] = (time.time(), value)
