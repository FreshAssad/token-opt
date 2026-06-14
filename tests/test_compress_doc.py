"""Document post-processing (pure functions) + a MarkItDown integration test."""
from __future__ import annotations

import pytest

from tokenopt.compress.doc import (
    collapse_blank_lines,
    flatten_tables,
    postprocess,
    strip_bibliography,
    strip_page_numbers,
    strip_repeated_lines,
)


def test_strip_page_numbers():
    src = "Intro\n3\nBody\nPage 4 of 10\n- 5 -\n12/34\nEnd"
    out = strip_page_numbers(src)
    assert "Intro" in out and "Body" in out and "End" in out
    assert "3" not in out.splitlines()
    assert "Page 4 of 10" not in out
    assert "- 5 -" not in out
    assert "12/34" not in out


def test_strip_repeated_headers_footers():
    lines = []
    for i in range(4):
        lines += ["ACME CONFIDENTIAL", f"unique paragraph {i} with distinct words"]
    out = strip_repeated_lines("\n".join(lines))
    assert "ACME CONFIDENTIAL" not in out  # recurring footer removed
    assert "unique paragraph 0" in out  # non-repeating content kept


def test_repeated_keeps_tables_and_headings():
    src = "\n".join(["# Heading"] * 3 + ["| a | b |"] * 3)
    out = strip_repeated_lines(src)
    assert "# Heading" in out
    assert "| a | b |" in out


def test_strip_bibliography():
    src = "# Paper\n\nbody\n\n## References\n\n[1] foo\n[2] bar"
    out = strip_bibliography(src)
    assert "body" in out
    assert "References" not in out
    assert "[1] foo" not in out


def test_flatten_tables():
    src = "| Q | Rev |\n| --- | --- |\n| Q1 | 100 |"
    out = flatten_tables(src)
    assert "Q - Rev" in out
    assert "Q1 - 100" in out
    assert "---" not in out


def test_collapse_blank_lines():
    out = collapse_blank_lines("a\n\n\n\nb\n   \n\n c ")
    assert "\n\n\n" not in out
    assert out.endswith("\n")


def test_postprocess_pipeline():
    src = "Title\n1\nReal text\nReal text\nReal text\nReal text\n2\n"
    out = postprocess(src)
    assert "Title" in out
    assert "1" not in out.splitlines()


def test_markitdown_html_integration():
    md = pytest.importorskip("markitdown")  # skip if not installed
    from tokenopt.compress.doc import compress_doc

    html = (
        b"<html><head><title>T</title></head><body>"
        b"<nav>Home About</nav><h1>Title</h1><p>Hello <b>world</b>.</p>"
        b"<footer>Confidential</footer></body></html>"
    )
    res = compress_doc(data=html, suffix=".html")
    assert "# Title" in res.output
    assert "Hello" in res.output
    assert res.raw  # raw conversion captured
