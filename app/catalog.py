"""模型目录（catalog）。

目录是一个可编辑的 JSON 文件：每条形如
    {"id": ..., "name": ..., "repo": "drawthingsai/Qwen-Image-2.1",
     "filename": "qwen_image_2.1_q8p.ckpt", "kind": "diffusion", "note": ...}

下载地址由 `{hf_base}/{repo}/resolve/main/{filename}` 现拼，所以换镜像/换源
只需要改 `MODELANCHOR_HF_BASE`，不必改数据。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CatalogModel:
    id: str
    name: str
    repo: str
    filename: str
    kind: str
    note: str
    size_bytes: int | None
    sha256: str | None
    url: str


def _load_raw(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    models = data.get("models", [])
    return models if isinstance(models, list) else []


def load(path: Path, hf_base: str) -> list[CatalogModel]:
    base = hf_base.rstrip("/")
    out: list[CatalogModel] = []
    for raw in _load_raw(path):
        repo = str(raw.get("repo", "")).strip().strip("/")
        filename = str(raw.get("filename", "")).strip()
        if not repo or not filename:
            continue
        out.append(
            CatalogModel(
                id=str(raw.get("id") or filename),
                name=str(raw.get("name") or filename),
                repo=repo,
                filename=filename,
                kind=str(raw.get("kind") or "unknown"),
                note=str(raw.get("note") or ""),
                size_bytes=raw.get("size_bytes"),
                sha256=raw.get("sha256"),
                url=f"{base}/{repo}/resolve/main/{filename}",
            )
        )
    return out


def find(catalog: list[CatalogModel], catalog_id: str) -> CatalogModel | None:
    for entry in catalog:
        if entry.id == catalog_id:
            return entry
    return None
