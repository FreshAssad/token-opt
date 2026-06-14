"""Detection, pipe routing, and the offline / no-network guarantee."""
from __future__ import annotations

import socket

import pytest

from tokenopt.detect import detect, sniff
from tokenopt.pipe import lossless_text_cleanup, run


def test_detect_by_extension():
    assert detect("a.json", b"{}") == "data"
    assert detect("a.pdf", b"%PDF-1.4") == "doc"
    assert detect("a.html", b"<html>") == "doc"
    assert detect("a.py", b"print(1)") == "code"
    assert detect("a.txt", b"hello") == "prose"
    assert detect("a.eml", b"From: x") == "email"


def test_sniff_content():
    assert sniff(b"%PDF-1.7 ...") == "doc"
    assert sniff(b"PK\x03\x04zip") == "doc"
    assert sniff(b'{"a": 1}') == "data"
    assert sniff(b"<!doctype html><html>") == "doc"
    assert sniff(b"just some prose") == "prose"


def test_pipe_routes_json_to_data():
    data = b'{"rows":[{"a":1,"b":2},{"a":3,"b":4}]}'
    res = run(None, data, model="gpt-4o")
    assert res.category == "data"
    assert "rows[2]{a,b}:" in res.output
    assert res.report.before >= res.report.after


def test_pipe_prose_is_lossless_cleanup():
    data = b"line one   \n\n\n\nline two\n"
    res = run("notes.txt", data, model="gpt-4o")
    assert res.category == "prose"
    assert "\n\n\n" not in res.output
    assert "line one" in res.output and "line two" in res.output


def test_lossless_cleanup_preserves_words():
    src = "a  \n\n\n\n  b  "
    out = lossless_text_cleanup(src)
    assert out.split() == ["a", "b"]


def test_pipe_html_to_doc():
    pytest.importorskip("markitdown")
    html = b"<html><body><h1>Hi</h1><p>Para.</p></body></html>"
    res = run("page.html", html, model="gpt-4o")
    assert res.category == "doc"
    assert "# Hi" in res.output


def test_offline_no_network(monkeypatch):
    """After availability is determined once, the offline path must not open a
    socket (steady-state offline guarantee for the non-API code path)."""
    from tokenopt.counting import count
    from tokenopt.compress.data import compress_data

    count("warm up tokenizer state", "gpt-4o")  # decide tiktoken availability

    def _no_sockets(*_a, **_k):
        raise AssertionError("network access attempted without --api")

    monkeypatch.setattr(socket, "socket", _no_sockets)

    assert count("offline counting works fine", "gpt-4o").tokens > 0
    assert compress_data('{"x":[{"a":1},{"a":2}]}', model="gpt-4o").output
