"""Generic prose -> extractive summary.

LOSSY and opt-in. This drops sentences; never use it where every word matters.

Implementation is a self-contained TextRank (no spaCy model download required,
which keeps it MIT-licensed, light, and runnable on an air-gapped box). An
optional spaCy backend can be added later behind a flag.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# A compact English stopword set — enough to keep TextRank similarity sane
# without pulling an NLP model.
_STOPWORDS = set(
    """a an and are as at be by for from has he in is it its of on that the to
    was were will with this these those i you we they them his her our your their
    but or nor so if then than too very can could should would may might must
    not no do does did done have had having been being about into over under
    again further once here there all any both each few more most other some such
    only own same s t just don now""".split()
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_WORD = re.compile(r"[A-Za-z0-9']+")


class NotYetImplemented(NotImplementedError):
    """Kept for backwards-compat with earlier stub imports."""


@dataclass
class SummaryResult:
    output: str
    kept: int
    total: int
    ratio: float
    notes: list[str] = field(default_factory=list)


def split_sentences(text: str) -> list[str]:
    # Split on blank lines first (paragraph/bullets), then sentence punctuation.
    chunks: list[str] = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = " ".join(para.split())
        if not para:
            continue
        parts = _SENT_SPLIT.split(para)
        chunks.extend(p.strip() for p in parts if p.strip())
    return chunks


def _content_words(sentence: str) -> list[str]:
    return [
        w for w in (t.lower() for t in _WORD.findall(sentence))
        if w not in _STOPWORDS and len(w) > 1
    ]


def _similarity(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    overlap = len(sa & sb)
    if overlap == 0:
        return 0.0
    # Classic TextRank normalization by sentence lengths.
    denom = math.log(len(sa) + 1) + math.log(len(sb) + 1)
    return overlap / denom if denom else 0.0


def _textrank(sentences: list[str], *, damping: float = 0.85, iters: int = 40) -> list[float]:
    n = len(sentences)
    if n == 1:
        return [1.0]
    words = [_content_words(s) for s in sentences]
    # Weighted adjacency.
    weights = [[0.0] * n for _ in range(n)]
    out = [0.0] * n
    for i in range(n):
        for j in range(i + 1, n):
            w = _similarity(words[i], words[j])
            weights[i][j] = weights[j][i] = w
        out[i] = sum(weights[i]) or 1e-9

    scores = [1.0 / n] * n
    for _ in range(iters):
        new = [(1 - damping) / n] * n
        for i in range(n):
            s = 0.0
            for j in range(n):
                if weights[j][i]:
                    s += weights[j][i] / out[j] * scores[j]
            new[i] += damping * s
        scores = new
    return scores


def summarize(text: str, *, ratio: float = 0.3, max_sentences: int | None = None) -> SummaryResult:
    sentences = split_sentences(text)
    total = len(sentences)
    notes: list[str] = []
    if total <= 1:
        return SummaryResult(text.strip(), total, total, ratio,
                             ["too short to summarize; returned as-is"])

    ratio = min(max(ratio, 0.0), 1.0)
    keep = max(1, round(total * ratio))
    if max_sentences:
        keep = min(keep, max_sentences)

    scores = _textrank(sentences)
    ranked = sorted(range(total), key=lambda i: scores[i], reverse=True)[:keep]
    chosen = sorted(ranked)  # restore original order
    output = " ".join(sentences[i] for i in chosen)
    notes.append(f"extractive summary (LOSSY): kept {len(chosen)}/{total} sentences")
    return SummaryResult(output, len(chosen), total, ratio, notes)


# Backwards-compatible entry point used by the CLI.
def compress_generic(text: str, *, ratio: float = 0.3) -> SummaryResult:
    return summarize(text, ratio=ratio)
