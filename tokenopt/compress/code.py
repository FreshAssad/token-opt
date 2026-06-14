"""Source code -> AST-based compression via tree-sitter.

Two modes, both language-aware (so we never mistake `//` inside a string for a
comment):

* default ("minify"): strip comments + tidy whitespace. **Code still runs** —
  docstrings/string literals are left intact.
* ``--skeleton``: keep signatures, replace function/method bodies with a
  placeholder. LOSSY (bodies dropped); great for giving an LLM an overview.

Grammars come from PyPI (``token-opt[code]``), so this works offline. Languages
without an installed grammar fall back to a safe whitespace-only cleanup.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from functools import lru_cache

# language -> (pip module, factory attribute)
_GRAMMARS = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
}

_EXT_LANG = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".tsx": "tsx",
}

# Skeleton config by language family.
_SKELETON = {
    "python": {"funcs": {"function_definition"}, "body": "block", "js": False},
    "javascript": {
        "funcs": {"function_declaration", "function_expression", "arrow_function",
                  "method_definition", "generator_function_declaration"},
        "body": "statement_block", "js": True,
    },
}


def _skeleton_cfg(language: str):
    if language == "python":
        return _SKELETON["python"]
    if language in ("javascript", "typescript", "tsx"):
        return _SKELETON["javascript"]
    return None


@dataclass
class CodeResult:
    output: str
    language: str | None
    mode: str  # "minify" | "skeleton" | "whitespace"
    warnings: list[str] = field(default_factory=list)


@lru_cache(maxsize=8)
def _get_parser(language: str):
    spec = _GRAMMARS.get(language)
    if not spec:
        return None
    mod_name, factory = spec
    try:
        from tree_sitter import Language, Parser

        mod = importlib.import_module(mod_name)
        lang = Language(getattr(mod, factory)())
        return Parser(lang)
    except Exception:
        return None


def language_for(filename: str | None) -> str | None:
    if not filename:
        return None
    return _EXT_LANG.get(os.path.splitext(filename)[1].lower())


# --------------------------------------------------------------------------- #
# Byte-range helpers
# --------------------------------------------------------------------------- #
def _collect_comments(node, ranges: list[tuple[int, int]]) -> None:
    if node.type == "comment" or node.type.endswith("_comment"):
        ranges.append((node.start_byte, node.end_byte))
    for child in node.children:
        _collect_comments(child, ranges)


def _remove_ranges(src: bytes, ranges: list[tuple[int, int]]) -> bytes:
    out = bytearray()
    last = 0
    for start, end in sorted(set(ranges)):
        if start < last:
            continue
        out += src[last:start]
        last = end
    out += src[last:]
    return bytes(out)


def _expand_standalone(src: bytes, start: int, end: int) -> tuple[int, int]:
    """If a comment is alone on its line, expand the range to drop the whole
    line (incl. its newline) so no blank line is left behind."""
    line_start = src.rfind(b"\n", 0, start) + 1
    line_end = src.find(b"\n", end)
    if line_end == -1:
        line_end = len(src)
    if src[line_start:start].strip() == b"" and src[end:line_end].strip() == b"":
        return (line_start, line_end + 1 if line_end < len(src) else line_end)
    return (start, end)


def _strip_comments(parser, src: bytes) -> bytes:
    tree = parser.parse(src)
    ranges: list[tuple[int, int]] = []
    _collect_comments(tree.root_node, ranges)
    ranges = [_expand_standalone(src, s, e) for s, e in ranges]
    return _remove_ranges(src, ranges)


def _apply_edits(src: bytes, edits: list[tuple[int, int, str]]) -> bytes:
    for start, end, rep in sorted(edits, key=lambda e: e[0], reverse=True):
        src = src[:start] + rep.encode("utf-8") + src[end:]
    return src


def _python_placeholder(body, src: bytes) -> str:
    indent = " " * body.start_point[1]
    # Preserve a leading docstring for context.
    first = body.children[0] if body.children else None
    doc = ""
    if (first is not None and first.type == "expression_statement"
            and first.children and first.children[0].type == "string"):
        doc = src[first.children[0].start_byte:first.children[0].end_byte].decode(
            "utf-8", errors="replace")
    if doc:
        return f"{doc}\n{indent}..."
    return "..."


def _collect_skeleton_edits(node, cfg, src: bytes, edits: list) -> None:
    if node.type in cfg["funcs"]:
        body = next((c for c in node.children if c.type == cfg["body"]), None)
        if body is not None:
            rep = "{ /* ... */ }" if cfg["js"] else _python_placeholder(body, src)
            edits.append((body.start_byte, body.end_byte, rep))
            return  # don't descend into a body we're collapsing
    for child in node.children:
        _collect_skeleton_edits(child, cfg, src, edits)


def _ws_cleanup(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: list[str] = []
    blank = False
    for ln in lines:
        if ln == "":
            if not blank:
                out.append("")
            blank = True
        else:
            out.append(ln)
            blank = False
    return "\n".join(out).strip() + "\n"


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def compress_code(
    source: str,
    *,
    filename: str | None = None,
    language: str | None = None,
    skeleton: bool = False,
) -> CodeResult:
    lang = language or language_for(filename)
    parser = _get_parser(lang) if lang else None
    warnings: list[str] = []

    if parser is None:
        if lang is None:
            warnings.append("unknown language; applied whitespace cleanup only.")
        else:
            warnings.append(
                f"no tree-sitter grammar for '{lang}' "
                '(install backends: pip install "token-opt[code]"); '
                "whitespace cleanup only."
            )
        return CodeResult(_ws_cleanup(source), lang, "whitespace", warnings)

    src = source.encode("utf-8")

    if skeleton:
        cfg = _skeleton_cfg(lang)
        if cfg is None:
            warnings.append(f"--skeleton not supported for '{lang}'; minifying instead.")
            skeleton = False

    if skeleton:
        # Strip comments first so the body placeholder (a comment in JS) survives.
        src = _strip_comments(parser, src)
        tree = parser.parse(src)
        edits: list[tuple[int, int, str]] = []
        _collect_skeleton_edits(tree.root_node, _skeleton_cfg(lang), src, edits)
        src = _apply_edits(src, edits)
        mode = "skeleton"
    else:
        src = _strip_comments(parser, src)
        mode = "minify"

    return CodeResult(_ws_cleanup(src.decode("utf-8", errors="replace")), lang, mode, warnings)
