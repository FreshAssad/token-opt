"""Token counting: exact path (mocked tiktoken), heuristic fallback, Claude
estimate vs --api, and provider resolution. Runs fully offline."""
from __future__ import annotations

import sys
import types

import pytest

from tokenopt.counting import CountResult, count, resolve_family
from tokenopt.counting import openai as oai


class _FakeEnc:
    name = "o200k_base"

    def encode(self, text, disallowed_special=()):  # noqa: ARG002
        return text.split()  # 1 "token" per whitespace-separated word


def test_resolve_family():
    assert resolve_family("gpt-4o") == "openai"
    assert resolve_family("o1") == "openai"
    assert resolve_family("claude-3-5-sonnet") == "claude"
    assert resolve_family("gemini-1.5-pro") == "gemini"
    assert resolve_family("llama-3.1-70b") == "llama"
    assert resolve_family("totally-unknown") == "openai"  # safe default


def test_openai_exact_path(monkeypatch):
    monkeypatch.setattr(oai, "_get_encoding", lambda name: _FakeEnc())
    monkeypatch.setattr(oai, "_TIKTOKEN_OK", None)
    r = oai.count("alpha beta gamma", "gpt-4o")
    assert isinstance(r, CountResult)
    assert r.exact is True
    assert r.tokens == 3
    assert r.method.startswith("tiktoken/")
    assert r.qualifier == "exact"


def test_openai_heuristic_fallback(monkeypatch):
    def boom(_name):
        raise RuntimeError("vocab unavailable")

    monkeypatch.setattr(oai, "_get_encoding", boom)
    monkeypatch.setattr(oai, "_TIKTOKEN_OK", None)
    r = oai.count("hello world, this is a test", "gpt-4o")
    assert r.exact is False
    assert "heuristic" in r.method
    assert r.tokens > 0


def test_heuristic_is_deterministic():
    a = oai._heuristic_tokens("the quick brown fox")
    b = oai._heuristic_tokens("the quick brown fox")
    assert a == b and a > 0


def test_claude_offline_is_labelled_estimate():
    r = count("some claude-bound text", "claude")
    assert r.exact is False
    assert r.qualifier == "estimate"


def test_opus_correction_increases_estimate():
    text = "word " * 80
    base = count(text, "claude").tokens
    corrected = count(text, "claude", opus_correction=True).tokens
    assert corrected >= base


def test_claude_api_exact_mocked(monkeypatch):
    from tokenopt.counting import claude as cl

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    class _Resp:
        input_tokens = 123

    class _Messages:
        def count_tokens(self, **kwargs):  # noqa: ARG002
            return _Resp()

    class _Client:
        def __init__(self, *a, **k):
            self.messages = _Messages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_Client))
    r = cl.count("hi", "claude", use_api=True)
    assert r.exact is True
    assert r.tokens == 123
    assert "count_tokens" in r.method


def test_claude_api_requires_key(monkeypatch):
    from tokenopt.counting import claude as cl

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        cl.count("hi", "claude", use_api=True)
