"""safetensors 的原始读写。

合并只需要"改键名 + 搬字节"，所以这里不解析张量内容、也不依赖 torch / numpy：
读 header（8 字节长度 + JSON）拿到每个张量的 dtype/shape/偏移，按新键名重排后
把数据区原样拷贝。bf16、fp8 之类 numpy 不认识的 dtype 也能安全通过。
"""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

MAX_HEADER_BYTES = 512 * 1024 * 1024


class SafetensorsError(RuntimeError):
    pass


@dataclass(frozen=True)
class TensorMeta:
    name: str
    dtype: str
    shape: list[int]
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


def read_header(path: Path) -> tuple[dict, int]:
    """返回 (header, 数据区起始偏移)。"""
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) < 8:
            raise SafetensorsError(f"{path.name} 不是合法的 safetensors（文件过短）")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len <= 1 or header_len > MAX_HEADER_BYTES:
            raise SafetensorsError(f"{path.name} 不是合法的 safetensors（header 长度异常）")
        raw = handle.read(header_len)
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsError(f"{path.name} 的 header 解析失败：{exc}") from exc
    if not isinstance(header, dict):
        raise SafetensorsError(f"{path.name} 的 header 结构异常")
    return header, 8 + header_len


def list_tensors(path: Path) -> list[TensorMeta]:
    header, _ = read_header(path)
    entries: list[TensorMeta] = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        offsets = meta.get("data_offsets") if isinstance(meta, dict) else None
        if not offsets or len(offsets) != 2:
            continue
        entries.append(
            TensorMeta(
                name=name,
                dtype=str(meta.get("dtype")),
                shape=list(meta.get("shape") or []),
                start=int(offsets[0]),
                end=int(offsets[1]),
            )
        )
    entries.sort(key=lambda item: item.start)
    return entries


def merge_into(
    sources: list[tuple[str, Path]],
    out_path: Path,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """把若干 (键前缀, safetensors 文件) 合并成一个单文件，返回总字节数。"""
    header: dict[str, dict] = {}
    plan: list[tuple[Path, int, int, int]] = []  # (源文件, 数据区起点, 相对偏移, 字节数)
    data_start: dict[Path, int] = {}
    offset = 0

    for prefix, source in sources:
        entries = list_tensors(source)
        if not entries:
            continue
        _, start = read_header(source)
        data_start[source] = start
        for tensor in entries:
            key = f"{prefix}{tensor.name}"
            if key in header:
                # 同一个键出现两次（例如两个 text_encoder 用了同一前缀），保留先到的
                continue
            header[key] = {
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [offset, offset + tensor.nbytes],
            }
            plan.append((source, start, tensor.start, tensor.nbytes))
            offset += tensor.nbytes

    total = offset
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * padding

    written = 0
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        for source, start, rel, nbytes in plan:
            with open(source, "rb") as handle:
                handle.seek(start + rel)
                left = nbytes
                while left > 0:
                    block = handle.read(min(8 * 1024 * 1024, left))
                    if not block:
                        break
                    out.write(block)
                    left -= len(block)
            written += nbytes
            if on_progress:
                on_progress(written, total)

    os.replace(tmp, out_path)
    return total
