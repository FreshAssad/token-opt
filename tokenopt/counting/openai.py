"""OpenAI / GPT token counting via tiktoken — exact and offline.

Networking note: tiktoken downloads its BPE vocabulary the *first* time a given
encoding is used, then caches it on disk (``TIKTOKEN_CACHE_DIR``). That is a
one-time setup, not per-invocation telemetry — after the cache is warm, counting
is 100% offline.

Graceful fallback (design principle #7): if the vocabulary is unavailable *and*
can't be fetched (fully air-gapped box, no warm cache), we fall back to a
deterministic offline heuristic and clearly label the result as an approximation
rather than crashing.
"""
from __future__ import annotations

import re
from functools import lru_cache

from .base import CountResult

# o200k_base  -> GPT-4o / o1 / o3 / GPT-4.1 (current);
# cl100k_base -> GPT-4 / GPT-3.5 / embeddings (legacy).
_O200K_PREFIXES = ("gpt-4o", "gpt-4.1", "o1", "o3", "o4", "chatgpt", "gpt-5")
_CL100K_PREFIXES = ("gpt-4", "gpt-3.5", "text-embedding")

# None = unknown yet, True/False = cached availability of tiktoken vocab.
_TIKTOKEN_OK: bool | None = None

# Rough GPT-style pre-tokenization for the heuristic fallback.
_HEURISTIC_SPLIT = re.compile(r"\w+|[^\w\s]|\s+")


def _encoding_name(model: str) -> str:
    m = (model or "").strip().lower()
    if m in ("gpt", "openai", "o200k", "o200k_base"):
        return "o200k_base"
    if m in ("cl100k", "cl100k_base"):
        return "cl100k_base"
    if m.startswith(_O200K_PREFIXES):
        return "o200k_base"
    if m.startswith(_CL100K_PREFIXES):
        return "cl100k_base"
    try:
        import tiktoken

        return tiktoken.encoding_name_for_model(model)
    except Exception:
        return "o200k_base"


@lru_cache(maxsize=8)
def _get_encoding(name: str):
    import tiktoken

    return tiktoken.get_encoding(name)


def _heuristic_tokens(text: str) -> int:
    """Deterministic offline approximation of a BPE token count.

    Splits into word/punctuation/whitespace runs, then estimates ~4 chars per
    sub-token within each word-ish chunk (min 1), plus newlines. Correlates
    reasonably with real BPE counts; only used when tiktoken vocab is missing.
    """
    total = 0
    for piece in _HEURISTIC_SPLIT.findall(text or ""):
        if piece.isspace():
            total += piece.count("\n")
            continue
        total += max(1, (len(piece) + 3) // 4)
    return total


def _tiktoken_len(text: str, encoding_name: str) -> int | None:
    """Exact length via tiktoken, or None if the vocab can't be loaded."""
    global _TIKTOKEN_OK
    if _TIKTOKEN_OK is False:
        return None
    try:
        enc = _get_encoding(encoding_name)
        # disallowed_special=() => treat "<|endoftext|>" etc. as ordinary text.
        n = len(enc.encode(text or "", disallowed_special=()))
        _TIKTOKEN_OK = True
        return n
    except Exception:
        _TIKTOKEN_OK = False
        return None


def _encode_len(text: str, encoding_name: str) -> int:
    """Token length, preferring tiktoken and falling back to the heuristic."""
    n = _tiktoken_len(text, encoding_name)
    return n if n is not None else _heuristic_tokens(text)


def count(text: str, model: str = "gpt-4o") -> CountResult:
    """OpenAI token count — exact via tiktoken, or labelled approx if offline."""
    name = _encoding_name(model)
    n = _tiktoken_len(text, name)
    if n is not None:
        return CountResult(tokens=n, exact=True, method=f"tiktoken/{name}", model=model)
    return CountResult(
        tokens=_heuristic_tokens(text),
        exact=False,
        method="offline-heuristic(approx)",
        model=model,
    )


def count_proxy(text: str, model: str = "llama") -> CountResult:
    """Approximate count for local/open-weight models (tiktoken cl100k proxy)."""
    n = _tiktoken_len(text, "cl100k_base")
    if n is not None:
        return CountResult(
            tokens=n, exact=False, method="tiktoken/cl100k_base(approx)", model=model
        )
    return CountResult(
        tokens=_heuristic_tokens(text),
        exact=False,
        method="offline-heuristic(approx)",
        model=model,
    )
