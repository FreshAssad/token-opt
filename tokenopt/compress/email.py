"""Email threads -> deduplicated quoted history.

Deep threads re-quote the entire conversation in every reply. This parses the
message(s) (.eml single, .mbox multiple, or raw text), keeps each message's new
content, and drops the redundant re-quoted history and signatures.

Pure stdlib (email/mailbox) — no extra dependencies. ``keep_quotes`` preserves
quoted lines (still de-duplicating exact repeats).
"""
from __future__ import annotations

import html as _html
import re
from dataclasses import dataclass, field

_ATTRIBUTION = re.compile(r"""^\s*(on\s.+\bwrote:|.*<[^>]+@[^>]+>\s*wrote:)\s*$""", re.IGNORECASE)
_OUTLOOK_HDR = re.compile(r"^\s*(from|sent|to|cc|subject):\s", re.IGNORECASE)
_SIG_DELIM = re.compile(r"^--\s*$")
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


@dataclass
class EmailResult:
    output: str
    messages: int
    warnings: list[str] = field(default_factory=list)


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n", text)
    text = _TAG.sub("", text)
    return _html.unescape(text)


def _looks_like_eml(text: str) -> bool:
    head = text[:2000]
    return bool(re.match(r"^[A-Za-z][\w-]*:\s", head)) and "\n\n" in head


def _extract_body(msg) -> str:
    part = None
    try:
        part = msg.get_body(preferencelist=("plain", "html"))
    except Exception:
        part = None
    target = part or msg
    try:
        content = target.get_content()
    except Exception:
        payload = target.get_payload(decode=True)
        content = payload.decode("utf-8", errors="replace") if payload else str(target.get_payload())
    ctype = target.get_content_type() if hasattr(target, "get_content_type") else "text/plain"
    if ctype == "text/html":
        content = _strip_html(content)
    return content


def _parse_eml(raw: bytes):
    import email
    from email import policy

    msg = email.message_from_bytes(raw, policy=policy.default)
    headers = {k: msg[k] for k in ("From", "Date", "Subject", "To") if msg[k]}
    return headers, _extract_body(msg)


def _parse_mbox(data: bytes):
    import mailbox
    import os
    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=".mbox", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        mb = mailbox.mbox(tmp.name)
        out = [_parse_eml(mb.get_bytes(key)) for key in mb.keys()]
        mb.close()
        return out
    finally:
        os.unlink(tmp.name)


def _messages_from(data: bytes, filename: str | None):
    text = data.decode("utf-8", errors="replace")
    if (filename and filename.lower().endswith(".mbox")) or text.startswith("From "):
        try:
            return _parse_mbox(data)
        except Exception:
            pass
    if _looks_like_eml(text):
        try:
            return [_parse_eml(data)]
        except Exception:
            pass
    return [({}, text)]  # raw text


def _normalize(line: str) -> str:
    return _WS.sub(" ", line.lstrip("> ").strip()).lower()


def _collapse_blanks(lines: list[str]) -> str:
    out: list[str] = []
    blank = False
    for ln in lines:
        if ln.strip() == "":
            if not blank:
                out.append("")
            blank = True
        else:
            out.append(ln)
            blank = False
    return "\n".join(out).strip()


def _format_headers(headers: dict) -> str:
    parts = [f"{k}: {v}" for k, v in headers.items() if k in ("From", "Date", "Subject")]
    return " | ".join(parts)


def compress_email(
    data: bytes,
    *,
    filename: str | None = None,
    keep_quotes: bool = False,
) -> EmailResult:
    messages = _messages_from(data, filename)
    seen: set[str] = set()
    blocks: list[str] = []

    for headers, body in messages:
        kept: list[str] = []
        for raw_line in body.splitlines():
            line = raw_line.rstrip()
            if _SIG_DELIM.match(line):
                break  # signature delimiter — drop the rest of this message
            if _ATTRIBUTION.match(line) or _OUTLOOK_HDR.match(line):
                continue
            quoted = line.lstrip().startswith(">")
            if quoted and not keep_quotes:
                continue
            norm = _normalize(line)
            if norm == "":
                kept.append("")
                continue
            if norm in seen:
                continue
            seen.add(norm)
            kept.append(line)

        text = _collapse_blanks(kept)
        hdr = _format_headers(headers)
        if not text and not hdr:
            continue
        block = f"=== {hdr} ===\n{text}" if hdr else text
        blocks.append(block.rstrip())

    output = "\n\n".join(blocks).strip() + "\n"
    return EmailResult(output=output, messages=len(messages))
