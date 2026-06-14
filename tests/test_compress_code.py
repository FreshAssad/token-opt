"""AST-based code compression (tree-sitter). Skipped if grammars aren't present."""
from __future__ import annotations

import ast

import pytest

pytest.importorskip("tree_sitter")
pytest.importorskip("tree_sitter_python")

from tokenopt.compress.code import compress_code, language_for

PY = '''# module comment
import os


def add(a, b):
    """Add."""
    # inner
    return a + b  # trailing


class Calc:
    # field
    def mul(self, a, b):
        return a * b
'''


def test_language_detection():
    assert language_for("x.py") == "python"
    assert language_for("x.js") == "javascript"
    assert language_for("x.ts") == "typescript"
    assert language_for("x.unknown") is None


def test_minify_strips_comments_keeps_runnable():
    res = compress_code(PY, filename="x.py")
    assert res.mode == "minify"
    assert "# module comment" not in res.output
    assert "# inner" not in res.output
    assert "# trailing" not in res.output
    assert '"""Add."""' in res.output  # docstring (a string literal) is kept
    ast.parse(res.output)  # must still be valid Python


def test_skeleton_drops_bodies_keeps_signatures():
    res = compress_code(PY, filename="x.py", skeleton=True)
    assert res.mode == "skeleton"
    assert "def add(a, b):" in res.output
    assert "def mul(self, a, b):" in res.output
    assert "return a + b" not in res.output
    assert "..." in res.output
    ast.parse(res.output)  # skeleton is still valid Python


def test_minify_reduces_or_equal_size():
    res = compress_code(PY, filename="x.py")
    assert len(res.output) < len(PY)


def test_unknown_language_falls_back_to_whitespace():
    res = compress_code("SELECT 1;   \n\n\n", filename="q.sql")
    assert res.mode == "whitespace"
    assert any("whitespace" in w for w in res.warnings)


def test_javascript_skeleton():
    pytest.importorskip("tree_sitter_javascript")
    js = "// h\nfunction f(a){\n  return a+1; // t\n}\n"
    res = compress_code(js, filename="x.js", skeleton=True)
    assert "function f(a)" in res.output
    assert "/* ... */" in res.output
    assert "return a+1" not in res.output
