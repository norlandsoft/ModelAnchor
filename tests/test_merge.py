"""布局判定、safetensors 合并的测试（全部本地构造，不联网）。"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from app.downloader import DownloadError, slug_for_repo
from app.hubs.base import classify, is_downloadable
from app.merger import ARCH_LABELS, build_plan, detect_arch
from app.safetensors_io import list_tensors, merge_into, read_header


def make_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header: dict[str, dict] = {}
    data = b""
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(payload)]}
        data += payload
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)


# ------------------------------------------------------------------ 布局判定
def test_classify_diffusers_by_model_index():
    layout, models = classify(["model_index.json", "unet/diffusion_pytorch_model.safetensors", "vae/x.safetensors"])
    assert layout == "diffusers"
    assert len(models) == 2


def test_classify_diffusers_by_unet_dir():
    assert classify(["unet/model.safetensors", "vae/diffusion_pytorch_model.safetensors"])[0] == "diffusers"
    assert classify(["transformer/model.safetensors"])[0] == "diffusers"


def test_classify_comfy_style_repo_is_single():
    # ComfyUI 风格：权重在 diffusion_models/，旁边还有 vae/ 目录，但不是 diffusers 结构
    layout, models = classify(
        ["diffusion_models/flux_bf16.safetensors", "vae/ae.safetensors", "text_encoders/t5.safetensors"]
    )
    assert layout == "single"
    assert {m.path for m in models} == {
        "diffusion_models/flux_bf16.safetensors",
        "vae/ae.safetensors",
        "text_encoders/t5.safetensors",
    }


def test_classify_unknown_when_no_weights():
    assert classify(["README.md", "config.json"])[0] == "unknown"


def test_is_downloadable_skips_docs_and_git():
    assert not is_downloadable(".gitattributes")
    assert not is_downloadable("README.md")
    assert not is_downloadable("assets/cover.png")
    assert not is_downloadable("tests/test_model.py")
    assert is_downloadable("model_index.json")
    assert is_downloadable("unet/diffusion_pytorch_model.safetensors")


# ------------------------------------------------------------------ 合并
def _fake_diffusers(root: Path) -> None:
    make_safetensors(
        root / "transformer" / "diffusion_pytorch_model.safetensors",
        {"block.weight": ("F16", [2, 2], b"\x01\x02\x03\x04")},
    )
    make_safetensors(
        root / "vae" / "diffusion_pytorch_model.safetensors",
        {"decoder.bias": ("F32", [2], b"\xaa" * 8)},
    )
    make_safetensors(
        root / "text_encoder" / "model.safetensors",
        {"encoder.weight": ("BF16", [1, 2], b"\xcc\xcc\xcc\xcc")},
    )


def test_detect_arch(tmp_path: Path):
    _fake_diffusers(tmp_path)
    assert detect_arch(tmp_path) == "flux"

    sdxl = tmp_path / "sdxl"
    (sdxl / "unet").mkdir(parents=True)
    (sdxl / "text_encoder_2").mkdir()
    assert detect_arch(sdxl) == "sdxl"

    sd15 = tmp_path / "sd15"
    (sd15 / "unet").mkdir(parents=True)
    assert detect_arch(sd15) == "sd15"


def test_build_plan_prefixes(tmp_path: Path):
    _fake_diffusers(tmp_path)
    plan = build_plan(tmp_path, "auto")
    assert plan.arch == "flux"
    assert plan.total_tensors == 3
    by_name = {c.name: c for c in plan.components}
    assert by_name["transformer"].prefix == "model.diffusion_model."
    assert by_name["vae"].prefix == "first_stage_model."
    assert by_name["text_encoder"].prefix == "cond_stage_model.transformer."


def test_build_plan_rejects_single_file_dir(tmp_path: Path):
    make_safetensors(tmp_path / "already.safetensors", {"x": ("F32", [1], b"\x00\x00\x00\x00")})
    with pytest.raises(DownloadError):
        build_plan(tmp_path, "auto")


def test_build_plan_errors_on_empty_dir(tmp_path: Path):
    with pytest.raises(DownloadError):
        build_plan(tmp_path, "auto")


def test_merge_produces_valid_single_file(tmp_path: Path):
    src = tmp_path / "diffusers-model"
    _fake_diffusers(src)
    plan = build_plan(src, "auto")

    out = tmp_path / "merged.safetensors"
    sources = [(c.prefix, shard) for c in plan.components for shard in c.files]
    total = merge_into(sources, out)
    assert total > 0

    header, data_start = read_header(out)
    tensors = list_tensors(out)
    assert len(tensors) == 3
    assert {t.name for t in tensors} == {
        "model.diffusion_model.block.weight",
        "first_stage_model.decoder.bias",
        "cond_stage_model.transformer.encoder.weight",
    }
    assert all(data_start + t.end <= out.stat().st_size for t in tensors)

    # 原始字节必须原样搬过来（BF16 也不该被转换）
    by_name = {t.name: t for t in tensors}
    entry = by_name["model.diffusion_model.block.weight"]
    with open(out, "rb") as fh:
        fh.seek(data_start + entry.start)
        assert fh.read(entry.nbytes) == b"\x01\x02\x03\x04"
    assert entry.dtype == "F16"
    assert entry.shape == [2, 2]
    assert "__metadata__" not in header


def test_merge_handles_sharded_component(tmp_path: Path):
    component = tmp_path / "s" / "unet"
    make_safetensors(component / "part-1.safetensors", {"a": ("F32", [1], b"\x01\x01\x01\x01")})
    make_safetensors(component / "part-2.safetensors", {"b": ("F32", [1], b"\x02\x02\x02\x02")})
    (component / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}}),
        encoding="utf-8",
    )
    plan = build_plan(tmp_path / "s", "sd15")
    assert plan.arch == "sd15"
    assert [p.name for p in plan.components[0].files] == ["part-1.safetensors", "part-2.safetensors"]

    out = tmp_path / "sharded.safetensors"
    merge_into([(c.prefix, p) for c in plan.components for p in c.files], out)
    assert {t.name for t in list_tensors(out)} == {
        "model.diffusion_model.a",
        "model.diffusion_model.b",
    }


# ------------------------------------------------------------------ 杂项
def test_slug_for_repo_matches_frontend_rule():
    assert slug_for_repo("Qwen/Qwen-Image-2.1") == "Qwen_Qwen-Image-2.1"
    assert slug_for_repo("hf-internal-testing/tiny-flux-pipe") == "hf-internal-testing_tiny-flux-pipe"


def test_arch_labels_cover_prefixes():
    from app.merger import ARCH_PREFIXES

    assert set(ARCH_PREFIXES) == set(ARCH_LABELS) - {"auto"}
