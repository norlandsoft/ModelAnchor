"""检索的模糊匹配：把文件名/关键词拆成 token，做归一化与打分。

站点 API 只按仓库名做子串匹配，而实际文件名常有分隔符差异：
`moss-8steps.safetensors` / `MOSS-Image-SFT-8-steps-dmd` / `moss_image_sft_8steps`。
这里先把查询拆成 token、生成几种写法去搜，再按「仓库名 + 文件名的 token 覆盖率」排序。
"""

from __future__ import annotations

import re

WEIGHT_EXTENSIONS = (".safetensors", ".ckpt", ".gguf")

# 只当筛选条件、不参与语义匹配的词
FORMAT_WORDS = {"safetensors", "ckpt", "gguf"}

# 量化/精度关键词：命中会加权
QUANT_WORDS = {
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "q8",
    "int4",
    "int8",
    "fp4",
    "fp8",
    "fp16",
    "bf16",
    "w4a8",
    "w6a8",
    "w8a8",
    "nvfp4",
    "i4x",
    "i8x",
    "q6p",
    "q8p",
    "gptq",
    "awq",
    "convrot",
    "gguf",
}


def strip_extension(text: str) -> str:
    lowered = text.strip()
    for ext in WEIGHT_EXTENSIONS:
        if lowered.lower().endswith(ext):
            return lowered[: -len(ext)]
    return lowered


def tokenize(text: str) -> list[str]:
    """`moss-8steps.safetensors` → ['moss', '8', 'steps']。"""
    lowered = strip_extension(text).lower()
    # q4_k_m / q4-k-m / q4km 归一成一个 token，避免拆成 q/4/k/m 四个碎片
    lowered = re.sub(r"q([2-8])[\s_-]?k[\s_-]?([ms])", r"q\1k\2", lowered)
    lowered = re.sub(r"i([48])[\s_-]?x\b", r"i\1x", lowered)
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    tokens: list[str] = []
    for chunk in lowered.split():
        for piece in re.findall(r"q[2-8]k[ms]|[a-z]+|\d+", chunk):
            if not piece:
                continue
            if len(piece) == 1 and piece.isalpha():
                continue  # q/k/m 这种单字母没信息量
            tokens.append(piece)
    return tokens


def build_variants(query: str, limit: int = 4) -> list[str]:
    """给站点搜索生成几种写法，覆盖分隔符差异。"""
    base = strip_extension(query).strip()
    tokens = tokenize(base)
    variants: list[str] = [base]
    if tokens:
        hyphen = "-".join(tokens)
        joined = "".join(tokens)
        # 兜底用第一个像样的英文词（通常是厂商名），别拿 "steps" 这种泛词去搜
        anchor = next((t for t in tokens if t.isalpha() and len(t) >= 3), tokens[0])
        for candidate in (hyphen, joined, anchor):
            if candidate and candidate.lower() not in {v.lower() for v in variants}:
                variants.append(candidate)
    return variants[:limit]


def detect_format(query: str) -> str | None:
    lowered = query.lower()
    for fmt in FORMAT_WORDS:
        if re.search(rf"(^|[^a-z]){fmt}([^a-z]|$)", lowered):
            return fmt
    return None


def has_quant_keyword(query: str) -> bool:
    lowered = strip_extension(query).lower()
    return bool(
        re.search(
            r"(q[2-8](?:_k(?:_[ms])?)?|int[48]|fp[48]|bf16|fp16|w[468]a8|nvfp4|i[48]x|q[68]p|gptq|awq|convrot)",
            lowered,
        )
    )


def _coverage(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    tokens = set(tokenize(text))
    if not tokens:
        return 0.0
    return len(query_tokens & tokens) / len(query_tokens)


def score_hit(query: str, repo_id: str, files: list[str]) -> tuple[float, list[str]]:
    """返回 (分数, 命中文件列表)。分数由仓库名与文件名的 token 覆盖率组成。"""
    tokens = set(tokenize(query))
    if not tokens:
        return 0.0, []

    repo_name = repo_id.replace("/", " ")
    name_score = _coverage(tokens, repo_name)

    matched: list[str] = []
    best_file = 0.0
    weight_files = [p for p in files if p.lower().endswith(WEIGHT_EXTENSIONS)]
    for path in weight_files:
        coverage = _coverage(tokens, path)
        best_file = max(best_file, coverage)
        if coverage >= 0.6:
            matched.append(path)

    # 文件名精确到 token 全覆盖时最可信；仓库名次之
    score = 0.55 * name_score + 0.45 * best_file
    if best_file >= 0.999:
        score += 0.2
    elif best_file >= 0.6:
        score += 0.05

    fmt = detect_format(query)
    if fmt and weight_files:
        has_format = any(path.lower().endswith(f".{fmt}") for path in weight_files)
        score += 0.1 if has_format else -0.3

    if has_quant_keyword(query):
        joined = " ".join(files).lower() + " " + repo_id.lower()
        score += 0.08 if re.search(
            r"(q[2-8]|int[48]|fp[48]|w[468]a8|nvfp4|i[48]x|gptq|awq|convrot)", joined
        ) else -0.05

    # 命中的文件按路径排序后带去重，最多 5 个
    deduped: list[str] = []
    for path in sorted(matched):
        if path not in deduped:
            deduped.append(path)
    return round(score, 4), deduped[:5]
