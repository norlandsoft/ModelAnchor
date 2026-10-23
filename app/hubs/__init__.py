"""模型站点集合。"""

from __future__ import annotations

from ..config import Settings
from .base import (
    Hub,
    HubError,
    RepoFile,
    RepoInfo,
    SearchHit,
    classify,
    is_downloadable,
    is_weight,
)
from .hf import HuggingFaceHub
from .modelscope import ModelScopeHub

__all__ = [
    "Hub",
    "HubError",
    "RepoFile",
    "RepoInfo",
    "SearchHit",
    "classify",
    "get_hubs",
    "is_downloadable",
    "is_weight",
]


def get_hubs(settings: Settings) -> dict[str, Hub]:
    return {
        "hf": HuggingFaceHub(settings.hf_base, timeout=settings.hub_timeout),
        "modelscope": ModelScopeHub(timeout=settings.hub_timeout),
    }
