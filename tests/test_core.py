"""核心纯逻辑的单元测试：不联网、不写真实模型目录。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import catalog as catalog_mod
from app import library
from app.config import PROJECT_ROOT
from app.downloader import DownloadError, DownloadJob, sanitize_filename
from app.library import resolve_target


# ------------------------------------------------------------------ 文件名
def test_sanitize_from_url_strips_query_and_unquotes():
    name = sanitize_filename("https://example.com/a/b/flux%20model_q8p.ckpt?download=true")
    assert name == "flux model_q8p.ckpt"


def test_sanitize_rejects_empty():
    with pytest.raises(DownloadError):
        sanitize_filename("https://example.com/")


def test_sanitize_strict_rejects_path_separators():
    with pytest.raises(DownloadError):
        sanitize_filename("../evil.ckpt", strict=True)
    with pytest.raises(DownloadError):
        sanitize_filename("sub/evil.ckpt", strict=True)


def test_sanitize_keeps_plain_name():
    assert sanitize_filename("model.ckpt", strict=True) == "model.ckpt"


# -------------------------------------------------------------------- 路径
def test_resolve_target_stays_inside(tmp_path: Path):
    target = resolve_target(tmp_path, "ok.ckpt")
    assert target == (tmp_path / "ok.ckpt").resolve()


def test_resolve_target_rejects_escape(tmp_path: Path):
    with pytest.raises(ValueError):
        resolve_target(tmp_path, "../../etc/passwd")


# ------------------------------------------------------------------ 模型库
def test_scan_finds_models_and_partials(tmp_path: Path):
    (tmp_path / "a.ckpt").write_bytes(b"x" * 10)
    (tmp_path / "b.gguf").write_bytes(b"y" * 20)
    (tmp_path / "c.ckpt.part").write_bytes(b"z" * 5)
    (tmp_path / ".DS_Store").write_bytes(b"junk")
    (tmp_path / "notes.txt").write_text("不是模型", encoding="utf-8")

    models = {m.filename: m for m in library.scan(tmp_path)}
    assert set(models) == {"a.ckpt", "b.gguf", "c.ckpt.part"}
    assert models["c.ckpt.part"].partial is True
    assert models["c.ckpt.part"].kind == "ckpt"
    assert models["b.gguf"].kind == "gguf"


def test_scan_missing_dir_is_empty(tmp_path: Path):
    assert library.scan(tmp_path / "nope") == []


# ------------------------------------------------------------------ 目录表
def test_catalog_builds_urls_from_base(tmp_path: Path):
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "models": [
                    {"id": "x", "name": "X", "repo": "drawthingsai/R", "filename": "r_q8p.ckpt"}
                ]
            }
        ),
        encoding="utf-8",
    )
    entries = catalog_mod.load(path, "https://hf-mirror.com/")
    assert len(entries) == 1
    assert entries[0].url == "https://hf-mirror.com/drawthingsai/R/resolve/main/r_q8p.ckpt"
    assert catalog_mod.find(entries, "x") is entries[0]
    assert catalog_mod.find(entries, "missing") is None


def test_catalog_missing_file_is_empty(tmp_path: Path):
    assert catalog_mod.load(tmp_path / "absent.json", "https://hf-mirror.com") == []


def test_shipped_catalog_is_valid():
    entries = catalog_mod.load(PROJECT_ROOT / "data" / "catalog.json", "https://hf-mirror.com")
    assert len(entries) >= 5
    assert len({e.id for e in entries}) == len(entries)
    assert all(e.filename.endswith((".ckpt", ".gguf")) for e in entries)


# -------------------------------------------------------------------- 任务
def test_job_dict_has_percent_and_eta(tmp_path: Path):
    job = DownloadJob(
        id="abc",
        filename="m.ckpt",
        url="https://example.com/m.ckpt",
        dest=tmp_path / "m.ckpt",
    )
    job.total_bytes = 100
    job.bytes_done = 25
    job.speed_bps = 10
    data = job.to_dict()
    assert data["percent"] == 25.0
    assert data["eta_seconds"] == 7.5
    assert data["state"] == "queued"
