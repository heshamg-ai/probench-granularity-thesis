from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

CountFn = Callable[[str, Any], int]
TruncateFn = Callable[[str, Any, int], str]


@dataclass
class TokenizerSpec:
    name: str
    backend: Any
    count_fn: CountFn
    truncate_fn: TruncateFn


# -----------------------------
# OpenAI / tiktoken
# -----------------------------
def _tiktoken_tokenize(text: str, tokenizer) -> list[int]:
    return tokenizer.encode(text, disallowed_special=())


def _tiktoken_count(text: str, tokenizer) -> int:
    if not text:
        return 0
    return len(_tiktoken_tokenize(text, tokenizer))


def _tiktoken_truncate(text: str, tokenizer, max_tokens: int) -> str:
    if not text:
        return ""
    tokens = _tiktoken_tokenize(text, tokenizer)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens])


# -----------------------------
# Hugging Face
# -----------------------------
def _hf_tokenize(text: str, tokenizer) -> list[int]:
    return tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]


def _hf_count(text: str, tokenizer) -> int:
    if not text:
        return 0
    return len(_hf_tokenize(text, tokenizer))


def _hf_truncate(text: str, tokenizer, max_tokens: int) -> str:
    if not text:
        return ""
    tokens = _hf_tokenize(text, tokenizer)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)


# -----------------------------
# Shared helpers for count-only backends
# -----------------------------
def _cut_text(text: str) -> str:
    if not text:
        return ""

    text = text.rstrip()
    boundaries = ["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", ", ", " "]
    tail = text[max(0, len(text) - 300):]
    offset = len(text) - len(tail)

    for boundary in boundaries:
        pos = tail.rfind(boundary)
        if pos >= 0:
            return text[: offset + pos + len(boundary)].rstrip()

    return text


def _truncate_by_counting(
    text: str,
    backend: Any,
    max_tokens: int,
    count_fn: CountFn,
) -> str:
    if not text:
        return ""

    if max_tokens <= 0:
        return ""

    if count_fn(text, backend) <= max_tokens:
        return text

    lo, hi = 0, len(text)
    best = ""

    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid]
        n_tokens = count_fn(candidate, backend)

        if n_tokens <= max_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1

    return _cut_text(best)


# -----------------------------
# Gemini
# -----------------------------
def _gemini_count(text: str, backend) -> int:
    if not text:
        return 0

    client, model_name = backend
    result = client.models.count_tokens(
        model=model_name,
        contents=text,
    )
    print("COUNT_TOKENS RESULT:", result)
    print("type:", type(result))
    print("total_tokens:", getattr(result, "total_tokens", None))
    print("dict:", getattr(result, "__dict__", None))

    total = getattr(result, "total_tokens", None)
    if total is None:
        raise ValueError(f"Gemini count_tokens returned no total_tokens: {result!r}")

    return int(total)
    #return result.total_tokens


def _gemini_truncate(text: str, backend, max_tokens: int) -> str:
    return _truncate_by_counting(text, backend, max_tokens, _gemini_count)


# -----------------------------
# Claude
# -----------------------------
def _claude_count(text: str, backend) -> int:
    if not text:
        return 0

    client, model_name = backend
    result = client.messages.count_tokens(
        model=model_name,
        messages=[{"role": "user", "content": text}],
    )
    return result.input_tokens


def _claude_truncate(text: str, backend, max_tokens: int) -> str:
    return _truncate_by_counting(text, backend, max_tokens, _claude_count)


# -----------------------------
# Factory
# -----------------------------
@lru_cache(maxsize=32)
def get_tokenizer(tokenizer_name: str) -> TokenizerSpec:
    raw_name = tokenizer_name.strip()
    lower_name = raw_name.lower()

    if lower_name in {"cl100k", "cl100k_base", "llama3", "llama3.1"}:
        # Llama 3.x uses a tiktoken-based tokenizer very similar to cl100k_base.
        # Using cl100k_base as a close approximation avoids the gated HF repo requirement.
        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")
        return TokenizerSpec(
            name=raw_name,
            backend=tokenizer,
            count_fn=_tiktoken_count,
            truncate_fn=_tiktoken_truncate,
        )

    HF_ALIASES = {
        "llama": "togethercomputer/LLaMA-2-7B-32K",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "qwen": "Qwen/Qwen2.5-7B-Instruct",
        "qwen3.5": "Qwen/Qwen3.5-9B",
        "deepseek": "Qwen/Qwen2.5-7B-Instruct",
        "deepseekv4pro": "deepseek-ai/DeepSeek-V4-Pro",
        "phi4": "microsoft/Phi-4-mini-instruct",
        "glm5.1": "zai-org/GLM-5.1",
        "kimik2.5": "moonshotai/Kimi-K2.5",
        "kimik2.6": "moonshotai/Kimi-K2.6",
        "minimaxm2.5": "MiniMaxAI/MiniMax-M2.5",
        "minimaxm2.7": "MiniMaxAI/MiniMax-M2.7",
    }
    TRUST_REMOTE_CODE = {"glm5.1", "kimik2.5", "kimik2.6", "minimaxm2.5", "minimaxm2.7"}
    # Models whose config.json is incompatible with current transformers — load tokenizer files only
    FAST_TOKENIZER_ONLY = {"deepseekv4pro"}

    if lower_name in HF_ALIASES:
        model_id = HF_ALIASES[lower_name]
        if lower_name in FAST_TOKENIZER_ONLY:
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
                trust_remote_code=(lower_name in TRUST_REMOTE_CODE),
            )
        return TokenizerSpec(
            name=raw_name,
            backend=tokenizer,
            count_fn=_hf_count,
            truncate_fn=_hf_truncate,
        )

    if lower_name.startswith("hf"):
        from transformers import AutoTokenizer

        model_id = raw_name[3:].strip()
        if not model_id:
            raise ValueError("Expected Hugging Face model id after 'hf:'")

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        return TokenizerSpec(
            name=raw_name,
            backend=tokenizer,
            count_fn=_hf_count,
            truncate_fn=_hf_truncate,
        )

    if lower_name.startswith("gemini"):
        from google import genai

        model_name = raw_name[7:].strip() or os.getenv("GEMINI_MODEL")
        if not model_name:
            raise ValueError(
                "Expected Gemini model name after 'gemini:' or GEMINI_MODEL in env"
            )

        api_key = os.getenv("GEMINI_API_KEY")
        base_url = os.getenv("GEMINI_BASE_URL")

        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["http_options"] = {"base_url": base_url}

        client = genai.Client(**client_kwargs)

        return TokenizerSpec(
            name=raw_name,
            backend=(client, model_name),
            count_fn=_gemini_count,
            truncate_fn=_gemini_truncate,
        )

    if lower_name.startswith("claude"):
        import anthropic

        model_name = raw_name[7:].strip() or os.getenv("ANTHROPIC_MODEL")
        if not model_name:
            raise ValueError(
                "Expected Claude model name after 'claude:' or ANTHROPIC_MODEL in env"
            )

        api_key = os.getenv("ANTHROPIC_API_KEY")
        base_url = os.getenv("ANTHROPIC_BASE_URL")

        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        client = anthropic.Anthropic(**client_kwargs)

        return TokenizerSpec(
            name=raw_name,
            backend=(client, model_name),
            count_fn=_claude_count,
            truncate_fn=_claude_truncate,
        )

    raise ValueError(
        f"Unknown tokenizer '{tokenizer_name}'. "
        f"Use one of: cl100k, hf:<model_id>, gemini:<model_name>, claude:<model_name>"
    )


# -----------------------------
# Public API
# -----------------------------
def count_tokens(text: str, spec: TokenizerSpec) -> int:
    return spec.count_fn(text or "", spec.backend)


def truncate_to_token_budget(text: str, spec: TokenizerSpec, max_tokens: int) -> str:
    return spec.truncate_fn(text or "", spec.backend, max_tokens)