"""跨站点找同一份文件的备用下载源。

HF 镜像的字节最终来自境外 CDN，速度经常只有 2–3 MB/s；同一个仓库/文件在
ModelScope 上也存在时，能从国内节点拿到 8–14 MB/s。这里负责找出这种镜像，
交给下载器测速后择优。
"""

from __future__ import annotations

from urllib.parse import unquote, urlparse

from .hubs import Hub
from .hubs.base import TTLCache

# 另一站点上没有这个仓库时记住结论，避免每个文件都去查一次
_REPO_MISS = TTLCache(ttl=600)


def alternate_urls(
    hubs: dict[str, Hub],
    source: str,
    repo_id: str,
    path: str,
    size: int | None = None,
) -> list[str]:
    """返回另一站点上同一文件的直链；找不到就返回空列表。"""
    other_name = "modelscope" if source == "hf" else "hf"
    other = hubs.get(other_name)
    if other is None:
        return []

    cache_key = f"{other_name}:{repo_id}"
    if _REPO_MISS.get(cache_key):
        return []

    try:
        info = other.repo_info(repo_id)
    except Exception:  # noqa: BLE001 —— 备用源只是加速，找不到就算了
        _REPO_MISS.set(cache_key, True)
        return []

    for candidate in info.files:
        if candidate.path != path:
            continue
        if size is not None and candidate.size is not None and candidate.size != size:
            continue  # 大小不同说明不是同一份文件，别拿它续传
        return [other.file_url(repo_id, path)]
    return []


def parse_resolve_url(url: str, hf_base: str) -> tuple[str, str, str] | None:
    """从「站点直链」里拆出 (source, repo_id, 文件路径)。

    支持：
      {hf_base}/{owner}/{repo}/resolve/{branch}/{path}
      https://modelscope.cn/models/{owner}/{repo}/resolve/{branch}/{path}
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None
    host = parsed.netloc.lower()
    segments = [unquote(part) for part in parsed.path.split("/") if part]

    if host == urlparse(hf_base).netloc.lower():
        if "resolve" in segments:
            index = segments.index("resolve")
            if index >= 2 and len(segments) > index + 2:
                repo_id = f"{segments[index - 2]}/{segments[index - 1]}"
                return "hf", repo_id, "/".join(segments[index + 2 :])
        return None

    if (
        host.endswith("modelscope.cn")
        and segments
        and segments[0] == "models"
        and "resolve" in segments
    ):
        index = segments.index("resolve")
        if index >= 2 and len(segments) > index + 2:
            repo_id = f"{segments[index - 2]}/{segments[index - 1]}"
            return "modelscope", repo_id, "/".join(segments[index + 2 :])
    return None
