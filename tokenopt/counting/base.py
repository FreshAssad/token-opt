"""Common token-counting interface and provider dispatch.

Every provider returns a :class:`CountResult` so callers never have to care
whether a number is exact (tiktoken / a provider API) or an offline estimate.
Honesty is the whole point: an estimate is *always* labelled as one.
"""
from __future__ import annotations

from dataclasses import dataclass

# Friendly model name -> provider family. Anything not listed falls through to
# the prefix heuristics in ``resolve_family``.
KNOWN_MODELS = {
    # OpenAI / GPT
    "gpt": "openai", "openai": "openai",
    "gpt-4o": "openai", "gpt-4o-mini": "openai", "gpt-4.1": "openai",
    "gpt-4": "openai", "gpt-4-turbo": "openai", "gpt-3.5-turbo": "openai",
    "o1": "openai", "o3": "openai", "o3-mini": "openai", "o4-mini": "openai",
    # Anthropic / Claude
    "claude": "claude", "anthropic": "claude",
    "claude-sonnet": "claude", "claude-opus": "claude", "claude-haiku": "claude",
    "claude-3-5-sonnet": "claude", "claude-3-opus": "claude",
    "claude-3-5-haiku": "claude", "claude-3-haiku": "claude",
    # Google / Gemini
    "gemini": "gemini", "gemini-pro": "gemini", "gemini-flash": "gemini",
    "gemini-1.5-pro": "gemini", "gemini-1.5-flash": "gemini",
    "gemini-2.0-flash": "gemini",
    # Local / open weights
    "llama": "llama", "local": "llama", "mistral": "llama",
}


@dataclass
class CountResult:
    """Outcome of a token count.

    Attributes:
        tokens:  the token count.
        exact:   True only when the number is ground truth (tiktoken for
                 OpenAI, or a provider API). Estimates/approximations are False.
        method:  short human string describing how the number was produced.
        model:   the model the count is for (as requested by the user).
    """

    tokens: int
    exact: bool
    method: str
    model: str

    @property
    def qualifier(self) -> str:
        """A short parenthetical for display, e.g. ``"exact"`` / ``"estimate"``."""
        if self.exact:
            return "exact"
        return "approx" if "approx" in self.method else "estimate"

    def as_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "exact": self.exact,
            "qualifier": self.qualifier,
            "method": self.method,
            "model": self.model,
        }


def resolve_family(model: str) -> str:
    """Map a model name to a provider family: openai/claude/gemini/llama."""
    m = (model or "").strip().lower()
    if m in KNOWN_MODELS:
        return KNOWN_MODELS[m]
    if m.startswith(("gpt", "o1", "o3", "o4", "chatgpt", "text-", "davinci")):
        return "openai"
    if "claude" in m or m.startswith("anthropic"):
        return "claude"
    if "gemini" in m or m.startswith("google"):
        return "gemini"
    if "llama" in m or "mistral" in m or "mixtral" in m or m == "local":
        return "llama"
    # Unknown: default to the only exact offline tokenizer we have.
    return "openai"


def count(
    text: str,
    model: str = "gpt-4o",
    *,
    use_api: bool = False,
    opus_correction: bool = False,
) -> CountResult:
    """Count tokens for ``text`` under ``model``.

    Network is only ever touched when ``use_api`` is True (and even then only
    for Claude/Gemini). The OpenAI path is exact and fully offline.
    """
    family = resolve_family(model)

    if family == "openai":
        from . import openai as provider
        return provider.count(text, model)
    if family == "claude":
        from . import claude as provider
        return provider.count(
            text, model, use_api=use_api, opus_correction=opus_correction
        )
    if family == "gemini":
        from . import gemini as provider
        return provider.count(text, model, use_api=use_api)
    if family == "llama":
        from . import openai as provider
        return provider.count_proxy(text, model)

    # Should be unreachable thanks to the default in resolve_family.
    from . import openai as provider
    return provider.count(text, model)
