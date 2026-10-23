"""运行配置。

所有可调项都走环境变量，方便以后接 launchd / docker，而不用改代码。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

APP_NAME = "ModelAnchor"
VERSION = "0.3.1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 可以进"模型库"的扩展名。DrawThings 用 .ckpt，LLM / 文本编码器用 .gguf。
MODEL_EXTENSIONS = (".ckpt", ".gguf", ".safetensors")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _probe(model_dir: Path) -> str:
    """探测目录能否真正读写（macOS 的 TCC 会让 is_dir() 成功但 listdir 被拒）。

    返回空字符串表示可用，否则是给人看的原因。
    """
    if not model_dir.is_dir():
        return "目录不存在"
    try:
        with os.scandir(model_dir) as entries:
            next(entries, None)
    except PermissionError:
        return "无读取权限（macOS：系统设置 → 隐私与安全性 → 完全磁盘访问权限，勾选终端）"
    except OSError as exc:
        return f"无法读取：{exc}"

    probe = model_dir / ".modelanchor-probe"
    try:
        probe.write_bytes(b"")
    except OSError as exc:
        return f"无写入权限：{exc}"
    finally:
        probe.unlink(missing_ok=True)
    return ""


def _resolve_model_dir() -> tuple[Path, str, str]:
    """返回 (目录, 来源, 警告)。默认放在项目内 models/，需要别的位置用环境变量覆盖。"""
    override = os.environ.get("MODELANCHOR_MODEL_DIR")
    if override:
        path = Path(override).expanduser().resolve()
        warning = _probe(path) if path.exists() else ""
        return path, "env:MODELANCHOR_MODEL_DIR", warning
    return PROJECT_ROOT / "models", "default", ""


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    model_dir_source: str
    host: str = "127.0.0.1"
    port: int = 7990
    max_concurrent_downloads: int = 2
    # 下载加速：分段并行连接数、分片大小、启用分段的下限
    download_connections: int = 4
    download_chunk_bytes: int = 16 * 1024 * 1024
    segment_min_bytes: int = 32 * 1024 * 1024
    # auto: 先测速再决定；always: 支持 Range 就并行；never: 只走单连接
    segment_mode: str = "auto"
    # HuggingFace 在国内经常直连不通，默认走镜像；可用 MODELANCHOR_HF_BASE 覆盖。
    hf_base: str = "https://hf-mirror.com"
    hub_timeout: float = 30.0
    catalog_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "catalog.json")
    web_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "web")
    allow_arbitrary_url: bool = True
    # 为空表示正常；非空时前端会把它当黄色告示显示
    warning: str = ""

    @property
    def part_suffix(self) -> str:
        return ".part"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    model_dir, source, warning = _resolve_model_dir()
    return Settings(
        model_dir=model_dir,
        model_dir_source=source,
        warning=warning,
        host=os.environ.get("MODELANCHOR_HOST", "127.0.0.1"),
        port=int(os.environ.get("MODELANCHOR_PORT", "7990")),
        max_concurrent_downloads=int(os.environ.get("MODELANCHOR_CONCURRENCY", "2")),
        download_connections=max(1, int(os.environ.get("MODELANCHOR_CONNECTIONS", "4"))),
        download_chunk_bytes=max(1, int(os.environ.get("MODELANCHOR_CHUNK_MB", "16"))) * 1024 * 1024,
        segment_min_bytes=max(0, int(os.environ.get("MODELANCHOR_SEGMENT_MIN_MB", "32"))) * 1024 * 1024,
        segment_mode=os.environ.get("MODELANCHOR_SEGMENT_MODE", "auto").strip().lower(),
        hf_base=os.environ.get("MODELANCHOR_HF_BASE", "https://hf-mirror.com").rstrip("/"),
        hub_timeout=float(os.environ.get("MODELANCHOR_HUB_TIMEOUT", "30")),
        allow_arbitrary_url=_env_bool("MODELANCHOR_ALLOW_ANY_URL", True),
    )
