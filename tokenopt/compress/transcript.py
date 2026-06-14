"""Meeting/podcast transcripts (.srt/.vtt) -> clean text.

Strips cue numbers, timestamps, and cue tags; merges consecutive lines by the
same speaker; removes filler words. ``--summary`` produces an extractive
(LOSSY) summary using the bundled TextRank. Pure stdlib + the local summarizer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_CUE_NUMBER = re.compile(r"^\d+$")
_TAG = re.compile(r"<[^>]+>")
_SPEAKER_VTT = re.compile(r"^<v\s+([^>]+)>(.*)$", re.IGNORECASE)
_SPEAKER_INLINE = re.compile(r"^([A-Z][A-Za-z0-9 ._'-]{0,30}):\s+(.*)$")

# Conservative filler set — intentionally excludes "like"/"so" to avoid
# changing meaning.
_FILLER_WORDS = re.compile(r"\b(?:u+m+|u+h+|e+r+|e+rm|a+h+|h+m+|mm+)\b", re.IGNORECASE)
_FILLER_PHRASES = re.compile(r"\b(?:you know|i mean|sort of|kind of)\b,?\s*", re.IGNORECASE)


@dataclass
class TranscriptResult:
    output: str
    cues: int
    warnings: list[str] = field(default_factory=list)


def _extract_speaker(text: str):
    m = _SPEAKER_VTT.match(text)
    if m:
        return m.group(1).strip(), m.group(2)
    m = _SPEAKER_INLINE.match(text)
    if m:
        return m.group(1).strip(), m.group(2)
    return None, text


def parse_cues(text: str) -> list[tuple[str | None, str]]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cues: list[tuple[str | None, str]] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.split("\n")
        if lines and lines[0].strip().upper().startswith("WEBVTT"):
            lines = lines[1:]
        if lines and lines[0].strip().upper().split()[0:1] in (["NOTE"], ["STYLE"], ["REGION"]):
            continue
        content = []
        for ln in lines:
            s = ln.strip()
            if not s or "-->" in s or _CUE_NUMBER.match(s):
                continue
            content.append(s)
        if not content:
            continue
        joined = " ".join(content)
        speaker, txt = _extract_speaker(joined)
        txt = _TAG.sub("", txt).strip()
        if txt:
            cues.append((speaker, txt))
    return cues


def _merge_by_speaker(cues):
    merged: list[list] = []
    for speaker, txt in cues:
        if merged and merged[-1][0] == speaker:
            merged[-1][1] += " " + txt
        else:
            merged.append([speaker, txt])
    return merged


def remove_filler(text: str) -> str:
    text = _FILLER_WORDS.sub("", text)
    text = _FILLER_PHRASES.sub("", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)  # space before punctuation
    text = re.sub(r"([,.!?;:])\1+", r"\1", text)  # doubled punctuation
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"^[\s,.;:]+", "", text)
    return text.strip()


def compress_transcript(
    data: bytes | str,
    *,
    filename: str | None = None,
    summary: bool = False,
    ratio: float = 0.3,
    strip_filler: bool = True,
) -> TranscriptResult:
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    cues = parse_cues(text)
    warnings: list[str] = []
    if not cues:
        warnings.append("no cues found; is this a .srt/.vtt transcript?")
        return TranscriptResult(text.strip() + "\n", 0, warnings)

    merged = _merge_by_speaker(cues)
    lines: list[str] = []
    for speaker, txt in merged:
        if strip_filler:
            txt = remove_filler(txt)
        if not txt:
            continue
        lines.append(f"{speaker}: {txt}" if speaker else txt)

    body = "\n".join(lines)
    if summary:
        from .generic import summarize

        res = summarize(body.replace("\n", " "), ratio=ratio)
        body = res.output
        warnings.extend(res.notes)

    return TranscriptResult(body.strip() + "\n", len(cues), warnings)
