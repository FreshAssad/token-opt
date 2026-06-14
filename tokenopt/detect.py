"""File-type detection by extension first, then content sniffing.

Returns a coarse category that the pipe orchestrator routes on:
    "doc"        - PDF/DOCX/PPTX/XLSX/HTML  (-> MarkItDown)
    "data"       - JSON                     (-> TOON encoder)
    "code"       - source files             [v2]
    "email"      - .eml / .mbox             [v2]
    "transcript" - .vtt / .srt              [v2]
    "prose"      - plain text / markdown
    "unknown"    - couldn't tell
"""
from __future__ import annotations

import json
import os

DOC_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".html", ".htm"}
DATA_EXTS = {".json", ".jsonl", ".ndjson"}
CODE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp", ".hpp",
    ".cc", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".cs",
    ".sh", ".bash", ".sql", ".css", ".scss",
}
EMAIL_EXTS = {".eml", ".mbox", ".msg"}
TRANSCRIPT_EXTS = {".vtt", ".srt"}
PROSE_EXTS = {".txt", ".md", ".markdown", ".rst", ".text", ".log"}


def detect_by_ext(path: str) -> str | None:
    ext = os.path.splitext(path)[1].lower()
    if ext in DOC_EXTS:
        return "doc"
    if ext in DATA_EXTS:
        return "data"
    if ext in CODE_EXTS:
        return "code"
    if ext in EMAIL_EXTS:
        return "email"
    if ext in TRANSCRIPT_EXTS:
        return "transcript"
    if ext in PROSE_EXTS:
        return "prose"
    return None


def sniff(data: bytes) -> str:
    """Best-effort content sniffing for extension-less / stdin input."""
    if not data:
        return "unknown"
    head = data[:4096]

    # Binary signatures.
    if head.startswith(b"%PDF"):
        return "doc"
    if head.startswith(b"PK\x03\x04"):
        # Zip container: docx/xlsx/pptx (and many others). Treat as doc.
        return "doc"

    # Text-ish from here on.
    try:
        text = head.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return "unknown"

    stripped = text.lstrip()
    low = stripped.lower()
    if low.startswith("<!doctype html") or low.startswith("<html") or "<body" in low[:200]:
        return "doc"

    if stripped[:1] in "{[":
        # Validate against the *whole* buffer when we have it.
        try:
            json.loads(data.decode("utf-8"))
            return "data"
        except Exception:
            # Could be JSONL or just looks like JSON; treat opening brace/bracket
            # as a data signal anyway.
            if stripped[:1] in "{[":
                return "data"

    return "prose"


def detect(path: str | None, data: bytes | None) -> str:
    """Detect category. Extension wins; otherwise sniff the bytes."""
    if path and path != "-":
        by_ext = detect_by_ext(path)
        if by_ext:
            return by_ext
    if data is not None:
        return sniff(data)
    return "unknown"
