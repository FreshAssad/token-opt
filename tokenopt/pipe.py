"""Auto-detect input type, route to the right compressor, report savings.

Compressed payload is returned for stdout; the :class:`Report` is for stderr.
Only ``doc`` and ``data`` do real compression in v1; everything else gets a
safe, lossless whitespace cleanup so `pipe` never silently drops information.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .counting import count
from .cost import cost_per_tokens
from .detect import detect
from .report import Report


@dataclass
class PipeResult:
    output: str
    category: str
    report: Report
    warnings: list[str] = field(default_factory=list)


def lossless_text_cleanup(text: str) -> str:
    """Trim trailing whitespace and squeeze runs of blank lines. Lossless."""
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
    return "\n".join(out).strip()


def _is_binary_doc(data: bytes) -> bool:
    return data.startswith(b"%PDF") or data.startswith(b"PK\x03\x04")


def run(
    path: str | None,
    data: bytes,
    *,
    model: str = "gpt-4o",
    keep_tables: bool = True,
    strip_bibliography: bool = False,
    data_format: str = "toon",
    use_api: bool = False,
    opus_correction: bool = False,
    force: str | None = None,
) -> PipeResult:
    # ``force`` lets the explicit `compress doc|data` commands skip detection.
    category = force or detect(path, data)
    warnings: list[str] = []

    def cnt(text: str) -> int:
        return count(text, model, use_api=use_api, opus_correction=opus_correction).tokens

    exact = count("x", model, use_api=use_api).exact

    if category == "data":
        from .compress.data import compress_data

        text = data.decode("utf-8", errors="replace")
        result = compress_data(text, fmt=data_format, model=model)
        before, after = cnt(text), result.tokens.get("chosen", cnt(result.output))
        warnings += result.notes
        output = result.output

    elif category == "doc":
        import os

        from .compress.doc import compress_doc

        named = bool(path) and path != "-"
        use_path = path if (named and os.path.exists(path)) else None
        suffix = (os.path.splitext(path)[1].lower() or None) if named else None
        result = compress_doc(
            source_path=use_path,
            data=None if use_path else data,
            suffix=suffix,
            keep_tables=keep_tables,
            strip_bibliography=strip_bibliography,
        )
        output = result.output
        warnings += result.warnings
        # Honest baseline: for text sources (HTML), compare against the original
        # text; for binary (PDF/DOCX) compare against the raw extraction.
        if _is_binary_doc(data):
            before = cnt(result.raw)
        else:
            before = cnt(data.decode("utf-8", errors="replace"))
        after = cnt(output)

    else:
        # code / email / transcript / prose / unknown -> lossless cleanup (v1).
        text = data.decode("utf-8", errors="replace")
        output = lossless_text_cleanup(text)
        before, after = cnt(text), cnt(output)
        if category in ("code", "email", "transcript"):
            warnings.append(
                f"'{category}' compression is a v2 feature; applied lossless "
                "whitespace cleanup only."
            )

    cost_saved = cost_per_tokens(before - after, model)
    report = Report(
        before=before,
        after=after,
        model=model,
        exact=exact,
        cost_saved=cost_saved,
        notes=warnings or None,
    )
    return PipeResult(output=output, category=category, report=report, warnings=warnings)
