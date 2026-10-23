"""模糊检索的 token / 变体 / 打分测试。"""

from __future__ import annotations

from app.search import build_variants, detect_format, has_quant_keyword, score_hit, tokenize


def test_tokenize_splits_separators_and_digits():
    assert tokenize("moss-8steps.safetensors") == ["moss", "8", "steps"]
    assert tokenize("MOSS_Image_SFT_8-steps") == ["moss", "image", "sft", "8", "steps"]


def test_tokenize_normalizes_quant_phrases():
    assert "q4km" in tokenize("flux1-dev-Q4_K_M.gguf")
    assert tokenize("model-Q4_K_M.gguf") == ["model", "q4km"]


def test_build_variants_covers_separator_styles():
    variants = build_variants("moss-8steps.safetensors")
    assert variants[0] == "moss-8steps"
    assert "moss-8-steps" in variants
    # 兜底词用厂商名，而不是 "steps" 这种泛词
    assert "moss" in variants


def test_detect_format_and_quant():
    assert detect_format("moss-8steps.safetensors") == "safetensors"
    assert detect_format("flux Q4_K_M.gguf") == "gguf"
    assert detect_format("moss int8") is None
    assert has_quant_keyword("flux q4_k_m") is True
    assert has_quant_keyword("moss int8") is True
    assert has_quant_keyword("moss") is False


def test_score_prefers_matching_filename():
    files = ["diffusion_models/moss_image_sft_8steps_bf16.safetensors", "README.md"]
    score, matched = score_hit("moss-8steps", "someone/moss-image-sft", files)
    assert score > 0.8
    assert matched == ["diffusion_models/moss_image_sft_8steps_bf16.safetensors"]


def test_score_ignores_non_weight_files():
    score, matched = score_hit("moss int8", "a/b", ["images/int8_prompt.jpg"])
    assert matched == []
    assert score < 0.5


def test_format_mismatch_is_penalized():
    gguf, _ = score_hit("flux q4_k_m.gguf", "unsloth/flux-GGUF", ["flux1-dev-Q4_K_M.gguf"])
    safetensors, _ = score_hit("flux q4_k_m.gguf", "unsloth/flux-GGUF", ["flux1-dev-BF16.safetensors"])
    assert gguf > safetensors
