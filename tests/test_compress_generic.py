"""Extractive prose summary (self-contained TextRank, LOSSY)."""
from __future__ import annotations

from tokenopt.compress.generic import split_sentences, summarize

TEXT = (
    "Climate change is accelerating worldwide. "
    "Global temperatures have risen sharply over the past century. "
    "Scientists attribute the warming to greenhouse gas emissions. "
    "Carbon dioxide is the primary greenhouse gas of concern. "
    "Renewable energy can reduce emissions significantly. "
    "Solar and wind power get cheaper every year. "
    "An unrelated cat sat quietly on a distant mat."
)


def test_split_sentences():
    assert len(split_sentences(TEXT)) == 7


def test_summary_respects_ratio():
    res = summarize(TEXT, ratio=0.3)
    assert res.total == 7
    assert res.kept == max(1, round(7 * 0.3))
    assert res.kept < res.total


def test_summary_is_deterministic():
    a = summarize(TEXT, ratio=0.4).output
    b = summarize(TEXT, ratio=0.4).output
    assert a == b


def test_summary_preserves_original_order():
    res = summarize(TEXT, ratio=0.6)
    sentences = split_sentences(TEXT)
    indices = [sentences.index(s) for s in split_sentences(res.output) if s in sentences]
    assert indices == sorted(indices)


def test_short_text_returned_asis():
    res = summarize("Only one sentence here.", ratio=0.3)
    assert res.kept == res.total
    assert "Only one sentence" in res.output
