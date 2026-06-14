"""Documents (PDF/DOCX/PPTX/XLSX/HTML/...) -> clean Markdown via MarkItDown,
then deterministic post-processing to strip the chrome that bloats token counts:
repeated headers/footers, page numbers, and (optionally) a trailing
bibliography.

The conversion is lossless in spirit (it preserves the document's text and
structure); the stripping removes only mechanically-repeated boilerplate.
"""
from __future__ import annotations

import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field

# Lines that are *just* a page number, in various house styles.
_PAGE_NUM_PATTERNS = [
    re.compile(r"^\s*\d{1,4}\s*$"),
    re.compile(r"^\s*[-–—]\s*\d{1,4}\s*[-–—]\s*$"),
    re.compile(r"^\s*page\s+\d{1,4}(\s+of\s+\d{1,4})?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d{1,4}\s*/\s*\d{1,4}\s*$"),
    re.compile(r"^\s*p\.\s*\d{1,4}\s*$", re.IGNORECASE),
]

_BIBLIO_HEADING = re.compile(
    r"^#{1,6}\s*(references|bibliography|works cited|citations)\s*$",
    re.IGNORECASE,
)


class DocConversionError(RuntimeError):
    """Raised when MarkItDown can't convert the input (clean, user-facing)."""


@dataclass
class DocResult:
    output: str  # cleaned markdown
    raw: str  # markdown as MarkItDown produced it (pre-stripping)
    title: str | None = None
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Post-processing primitives (pure functions — unit-testable without MarkItDown)
# --------------------------------------------------------------------------- #
def strip_page_numbers(text: str) -> str:
    out = [ln for ln in text.splitlines() if not _is_page_number(ln)]
    return "\n".join(out)


def _is_page_number(line: str) -> bool:
    return any(p.match(line) for p in _PAGE_NUM_PATTERNS)


def strip_repeated_lines(text: str, *, min_count: int = 3, max_len: int = 80) -> str:
    """Remove short lines that recur many times — classic running headers/footers.

    Conservative on purpose: never touches headings, table rows, list items, or
    long lines, so real repeated content (e.g. table data) survives.
    """
    lines = text.splitlines()
    counts = Counter(ln.strip() for ln in lines if ln.strip())

    def removable(stripped: str) -> bool:
        if not stripped or len(stripped) > max_len:
            return False
        if counts[stripped] < min_count:
            return False
        if stripped.startswith("#"):  # heading
            return False
        if "|" in stripped:  # table row
            return False
        if stripped[0] in "-*+>" and (len(stripped) < 2 or stripped[1] == " "):
            return False  # list item / quote
        if re.match(r"^\d+[.)]\s", stripped):  # ordered list item
            return False
        return True

    return "\n".join(ln for ln in lines if not removable(ln.strip()))


def strip_bibliography(text: str) -> str:
    lines = text.splitlines()
    cut = None
    for i, ln in enumerate(lines):
        if _BIBLIO_HEADING.match(ln.strip()):
            cut = i
    if cut is None:
        return text
    return "\n".join(lines[:cut]).rstrip()


def flatten_tables(text: str) -> str:
    """Convert markdown pipe tables to plain ' - '-joined lines (drops borders)."""
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("|") and s.endswith("|"):
            # Skip the separator row (|---|---|).
            if re.match(r"^\|[\s:\-|]+\|$", s):
                continue
            cells = [c.strip() for c in s.strip("|").split("|")]
            out.append(" - ".join(c for c in cells if c))
        else:
            out.append(ln)
    return "\n".join(out)


def collapse_blank_lines(text: str) -> str:
    # Trim trailing whitespace per line, then squeeze 2+ blank lines into 1.
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


def postprocess(
    markdown: str,
    *,
    keep_tables: bool = True,
    drop_bibliography: bool = False,
) -> str:
    text = markdown
    text = strip_page_numbers(text)
    text = strip_repeated_lines(text)
    if drop_bibliography:
        text = strip_bibliography(text)
    if not keep_tables:
        text = flatten_tables(text)
    return collapse_blank_lines(text)


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def _guess_suffix(data: bytes) -> str:
    if data.startswith(b"%PDF"):
        return ".pdf"
    if data.startswith(b"PK\x03\x04"):
        return ".docx"  # best guess among office zip formats
    head = data[:256].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<body" in head:
        return ".html"
    return ".txt"


def _convert(source_path: str | None, data: bytes | None, suffix: str | None):
    try:
        from markitdown import MarkItDown
    except Exception as exc:  # pragma: no cover - import guard
        raise DocConversionError(
            'MarkItDown is unavailable. Install document support with: '
            'pip install "token-opt[doc]"'
        ) from exc

    md = MarkItDown()
    try:
        if source_path:
            res = md.convert(source_path)
        else:
            if data is None:
                raise ValueError("compress_doc needs either source_path or data")
            sfx = suffix or _guess_suffix(data)
            tmp = tempfile.NamedTemporaryFile(suffix=sfx, delete=False)
            try:
                tmp.write(data)
                tmp.close()
                res = md.convert(tmp.name)
            finally:
                os.unlink(tmp.name)
    except DocConversionError:
        raise
    except Exception as exc:
        # MarkItDown raises FileConversionException / MissingDependencyException /
        # UnsupportedFormatException — collapse to one clean line.
        msg = str(exc).strip().splitlines()
        detail = next((ln.strip() for ln in msg if ln.strip()), exc.__class__.__name__)
        hint = ""
        if "dependenc" in detail.lower() or "MissingDependency" in type(exc).__name__:
            hint = ' — install backends with: pip install "token-opt[doc]"'
        raise DocConversionError(f"could not convert document: {detail}{hint}") from exc

    raw = getattr(res, "text_content", None) or getattr(res, "markdown", "") or ""
    title = getattr(res, "title", None)
    return raw, title


def compress_doc(
    source_path: str | None = None,
    data: bytes | None = None,
    *,
    suffix: str | None = None,
    keep_tables: bool = True,
    strip_bibliography: bool = False,
) -> DocResult:
    raw, title = _convert(source_path, data, suffix)
    warnings: list[str] = []

    if not raw.strip():
        warnings.append(
            "no extractable text — the document may be scanned/image-only "
            "(OCR is out of scope for v1)."
        )

    cleaned = postprocess(
        raw, keep_tables=keep_tables, drop_bibliography=strip_bibliography
    )
    return DocResult(output=cleaned, raw=raw, title=title, warnings=warnings)
