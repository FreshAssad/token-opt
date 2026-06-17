"""Auto-detect input type, route to the right compressor, report savings.

Compressed payload is returned for stdout; the :class:`Report` is for stderr.
``data``/``doc``/``code``/``email``/``transcript`` get real compression; prose
and unknown input get a safe, lossless whitespace cleanup, so `pipe` never
silently drops information. The LOSSY ``generic`` summarizer is only used when
explicitly requested (never via auto-detect).
"""
from __future__ import annotations

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
    skeleton: bool = False,
    keep_quotes: bool = False,
    summary: bool = False,
    ratio: float = 0.3,
    rename: bool = False,
    language: str | None = None,
    doc_backend: str = "markitdown",
) -> PipeResult:
    import os

    # ``force`` lets the explicit `compress <mode>` commands skip detection.
    category = force or detect(path, data)
    warnings: list[str] = []
    named = bool(path) and path != "-"

    def cnt(text: str) -> int:
        return count(text, model, use_api=use_api, opus_correction=opus_correction).tokens

    exact = count("x", model, use_api=use_api).exact
    text_in = data.decode("utf-8", errors="replace")

    if category == "data":
        from .compress.data import compress_data

        result = compress_data(text_in, fmt=data_format, model=model)
        output = result.output
        before, after = cnt(text_in), result.tokens.get("chosen", cnt(output))
        warnings += result.notes

    elif category == "doc":
        from .compress.doc import compress_doc

        use_path = path if (named and os.path.exists(path)) else None
        suffix = (os.path.splitext(path)[1].lower() or None) if named else None
        result = compress_doc(
            source_path=use_path,
            data=None if use_path else data,
            suffix=suffix,
            keep_tables=keep_tables,
            strip_bibliography=strip_bibliography,
            backend=doc_backend,
        )
        output = result.output
        warnings += result.warnings
        # Honest baseline: text sources (HTML) vs original text; binary
        # (PDF/DOCX) vs the raw extraction.
        before = cnt(result.raw) if _is_binary_doc(data) else cnt(text_in)
        after = cnt(output)

    elif category == "code":
        from .compress.code import compress_code

        result = compress_code(
            text_in, filename=path if named else None, language=language,
            skeleton=skeleton, rename=rename,
        )
        output = result.output
        warnings += result.warnings
        before, after = cnt(text_in), cnt(output)

    elif category == "email":
        from .compress.email import compress_email

        result = compress_email(data, filename=path if named else None, keep_quotes=keep_quotes)
        output = result.output
        warnings += result.warnings
        before, after = cnt(text_in), cnt(output)

    elif category == "transcript":
        from .compress.transcript import compress_transcript

        result = compress_transcript(
            data, filename=path if named else None, summary=summary, ratio=ratio
        )
        output = result.output
        warnings += result.warnings
        before, after = cnt(text_in), cnt(output)

    elif category == "generic":
        # LOSSY extractive prose — only reached when explicitly forced.
        from .compress.generic import summarize

        result = summarize(text_in, ratio=ratio)
        output = result.output
        warnings += result.notes
        before, after = cnt(text_in), cnt(output)

    else:
        # prose / unknown -> lossless whitespace cleanup (never drops info).
        output = lossless_text_cleanup(text_in)
        before, after = cnt(text_in), cnt(output)

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
