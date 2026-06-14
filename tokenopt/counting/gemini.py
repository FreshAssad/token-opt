"""Gemini (Google) token counting.

Priority order, offline first:

1. ``use_api=True``  -> exact, via google-genai ``count_tokens`` (needs key).
2. Local Vertex tokenizer (optional ``token-opt[gemini]`` extra) -> near-exact,
   fully offline.
3. Fallback -> approximate, using o200k_base as a proxy. Labelled ``(approx)``.
"""
from __future__ import annotations

import os

from .base import CountResult

DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

_MODEL_ALIASES = {
    "gemini": DEFAULT_GEMINI_MODEL,
    "google": DEFAULT_GEMINI_MODEL,
    "gemini-flash": "gemini-1.5-flash",
    "gemini-pro": "gemini-1.5-pro",
}


def _resolve_model(model: str) -> str:
    m = (model or "").strip().lower()
    return _MODEL_ALIASES.get(m, model if m.startswith("gemini") else DEFAULT_GEMINI_MODEL)


def _proxy_tokens(text: str) -> int:
    from .openai import _encode_len

    return _encode_len(text, "o200k_base")


def _vertex_local_count(text: str, model: str):
    """Return token count via the optional offline Vertex tokenizer, or None."""
    try:
        from vertexai.preview import tokenization
    except Exception:
        return None
    try:
        tok = tokenization.get_tokenizer_for_model(_resolve_model(model))
        return int(tok.count_tokens(text or "").total_tokens)
    except Exception:
        return None


def count(text: str, model: str = "gemini", *, use_api: bool = False) -> CountResult:
    if use_api:
        return _api_count(text, model)

    local = _vertex_local_count(text, model)
    if local is not None:
        return CountResult(
            tokens=local,
            exact=False,  # near-exact, but not the provider's ground truth
            method="vertex-local(near-exact)",
            model=model,
        )

    n = _proxy_tokens(text)
    return CountResult(
        tokens=n, exact=False, method="proxy:o200k_base(approx)", model=model
    )


def _api_count(text: str, model: str) -> CountResult:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set — cannot get an exact "
            "Gemini count via --api. Unset --api for an offline estimate."
        )
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "The 'google-genai' package is required for Gemini --api. Install "
            "with 'pip install google-genai'."
        ) from exc

    client = genai.Client(api_key=key)
    resp = client.models.count_tokens(model=_resolve_model(model), contents=text or "")
    return CountResult(
        tokens=int(resp.total_tokens),
        exact=True,
        method=f"genai.count_tokens({_resolve_model(model)})",
        model=model,
    )
