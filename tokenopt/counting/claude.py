"""Claude (Anthropic) token counting.

There is no public *offline* tokenizer for Claude 3/4+. So:

* Offline (default): an **estimate**, produced with the o200k_base BPE as a
  proxy. Always labelled ``(estimate)`` — we never present it as exact.
* ``use_api=True``: **exact**, via Anthropic's free ``messages.count_tokens``
  endpoint (requires ``ANTHROPIC_API_KEY``). This is the only path that hits
  the network.

Newer Opus-class models tokenize a little denser; ``opus_correction`` applies a
~1.3x nudge to the offline estimate. It is off by default and clearly labelled.
"""
from __future__ import annotations

import os

from .base import CountResult

# Stable, well-known model id used for API counting when the caller only said
# "claude". Override by passing a full model id. Editable on purpose.
DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet-20241022"

# Multiplier for --opus-correction (newer Opus-class models tokenize denser).
OPUS_CORRECTION = 1.3

# Friendly alias -> concrete API model id for the count_tokens call.
_API_ALIASES = {
    "claude": DEFAULT_CLAUDE_MODEL,
    "anthropic": DEFAULT_CLAUDE_MODEL,
    "claude-sonnet": DEFAULT_CLAUDE_MODEL,
    "claude-3-5-sonnet": DEFAULT_CLAUDE_MODEL,
    "claude-opus": "claude-3-opus-20240229",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-haiku": "claude-3-haiku-20240307",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-3-5-haiku": "claude-3-5-haiku-20241022",
}


def _proxy_tokens(text: str) -> int:
    from .openai import _encode_len

    return _encode_len(text, "o200k_base")


def _resolve_api_model(model: str) -> str:
    m = (model or "").strip().lower()
    if m in _API_ALIASES:
        return _API_ALIASES[m]
    # Looks like a concrete dated id already (e.g. claude-3-5-sonnet-20241022)?
    if m.startswith("claude-") and any(ch.isdigit() for ch in m.split("-")[-1]):
        return model
    return DEFAULT_CLAUDE_MODEL


def count(
    text: str,
    model: str = "claude",
    *,
    use_api: bool = False,
    opus_correction: bool = False,
) -> CountResult:
    if use_api:
        return _api_count(text, model)

    n = _proxy_tokens(text)
    method = "proxy:o200k_base(estimate)"
    if opus_correction:
        n = round(n * OPUS_CORRECTION)
        method = f"proxy:o200k_base x{OPUS_CORRECTION}(estimate)"
    return CountResult(tokens=n, exact=False, method=method, model=model)


def _api_count(text: str, model: str) -> CountResult:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set — cannot get an exact Claude count via "
            "--api. Unset --api for an offline estimate, or export your key."
        )
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "The 'anthropic' package is required for --api. Install with "
            "'pip install anthropic'."
        ) from exc

    api_model = _resolve_api_model(model)
    client = anthropic.Anthropic()
    resp = client.messages.count_tokens(
        model=api_model,
        messages=[{"role": "user", "content": text or ""}],
    )
    return CountResult(
        tokens=int(resp.input_tokens),
        exact=True,
        method=f"anthropic.count_tokens({api_model})",
        model=model,
    )
