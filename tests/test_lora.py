"""LoRA 识别规则的测试。"""

from __future__ import annotations

from app.hubs.base import classify, looks_like_lora


def test_lora_by_tag():
    layout, weights = classify(["pixel-art-xl.safetensors"], tags=["lora"], repo_id="nerijs/pixel-art-xl")
    assert layout == "lora"
    assert len(weights) == 1


def test_lora_by_base_model_adapter_tag():
    assert looks_like_lora(["model.safetensors"], tags=["base_model:adapter:black-forest-labs/FLUX.1-dev"])
    layout, _ = classify(
        ["model.safetensors"],
        tags=["lora", "base_model:adapter:black-forest-labs/FLUX.1-dev"],
        repo_id="someone/style",
    )
    assert layout == "lora"


def test_lora_by_repo_name_without_tags():
    # ModelScope 上很多仓库没有 tags，只能靠仓库名
    layout, _ = classify(["Qwen-Image-2.1-3D-lora_c2-st1000.safetensors"], repo_id="qaz963plm/Qwen-Image-2.1-3D-lora")
    assert layout == "lora"


def test_lora_by_filename_when_all_weights_are_lora():
    layout, _ = classify(
        ["sdxl/style_lora_v2.safetensors", "sdxl/style_lora_v1.safetensors"],
        tags=[],
        repo_id="someone/styles",
    )
    assert layout == "lora"


def test_mixed_comfy_repo_is_not_lora():
    # 整合包里同时有主模型和 lora，不能当成纯 LoRA
    layout, _ = classify(
        ["diffusion_models/base.safetensors", "loras/turbo_lora.safetensors"],
        tags=[],
        repo_id="Comfy-Org/Qwen-Image-2.1",
    )
    assert layout == "single"


def test_base_model_without_lora_signals_is_single():
    layout, _ = classify(["flux1-dev.safetensors"], tags=[], repo_id="black-forest-labs/FLUX.1-dev")
    assert layout == "single"


def test_diffusers_still_wins_over_loose_lora_filename():
    layout, _ = classify(
        ["model_index.json", "unet/diffusion_pytorch_model.safetensors", "README.md"],
        tags=[],
        repo_id="black-forest-labs/FLUX.1-dev",
    )
    assert layout == "diffusers"
