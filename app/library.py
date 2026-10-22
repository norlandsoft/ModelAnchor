"""本地模型库：扫描 DrawThings 模型目录。"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import MODEL_EXTENSIONS


@dataclass(frozen=True)
class LocalModel:
    filename: str
    size_bytes: int
    modified_at: float
    partial: bool
    is_dir: bool = False

    @property
    def kind(self) -> str:
        if self.is_dir:
            return "folder"
        name = self.filename.lower()
        if self.partial:
            name = name.removesuffix(".part")
        if name.endswith(".ckpt"):
            return "ckpt"
        if name.endswith(".gguf"):
            return "gguf"
        if name.endswith(".safetensors"):
            return "safetensors"
        return "other"


def _is_model_file(path: Path) -> bool:
    name = path.name
    if name.startswith("."):
        return False
    lowered = name.lower()
    if lowered.endswith(".part"):
        lowered = lowered.removesuffix(".part")
    return lowered.endswith(MODEL_EXTENSIONS)


def scan(model_dir: Path) -> list[LocalModel]:
    """扫描目录（不递归，DrawThings 的模型都在同一层）。目录不存在时返回空表。"""
    if not model_dir.is_dir():
        return []

    found: list[LocalModel] = []
    try:
        entries = list(model_dir.iterdir())
    except OSError:
        # 目录存在但读不了（macOS TCC、权限位），当作空库，别把整个 API 打成 500
        return []
    for path in entries:
        if path.name.startswith("."):
            continue
        if path.is_dir():
            found.append(
                LocalModel(
                    filename=path.name,
                    size_bytes=_dir_size(path),
                    modified_at=_mtime(path),
                    partial=False,
                    is_dir=True,
                )
            )
            continue
        if not path.is_file() or not _is_model_file(path):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        found.append(
            LocalModel(
                filename=path.name,
                size_bytes=stat.st_size,
                modified_at=stat.st_mtime,
                partial=path.name.lower().endswith(".part"),
            )
        )
    found.sort(key=lambda m: m.filename.lower())
    return found


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda _e: None):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def disk_usage(model_dir: Path) -> tuple[int, int]:
    """返回 (剩余, 总量)。目录不存在时退到最近的已存在祖先。"""
    probe = model_dir
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError:
        return 0, 0
    return usage.free, usage.total


def resolve_target(model_dir: Path, filename: str) -> Path:
    """把用户给的文件名锁死在模型目录内，防止 ../ 逃逸。"""
    target = (model_dir / filename).resolve()
    root = model_dir.resolve()
    if target != root and root not in target.parents:
        raise ValueError("非法文件名：不能指向模型目录之外")
    return target
