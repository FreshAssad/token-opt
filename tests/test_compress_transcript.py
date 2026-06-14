"""Transcript cleanup for .srt/.vtt (stdlib + local TextRank)."""
from __future__ import annotations

from tokenopt.compress.transcript import compress_transcript, parse_cues, remove_filler

SRT = b"""1
00:00:01,000 --> 00:00:04,000
Alice: Hello everyone, um, welcome.

2
00:00:04,000 --> 00:00:07,000
Alice: Today we discuss the roadmap.

3
00:00:07,000 --> 00:00:10,000
Bob: Uh, sounds good.
"""

VTT = b"""WEBVTT

00:00.000 --> 00:02.000
<v Carol>Let's begin the demo.

00:02.000 --> 00:04.000
<v Carol>It shows the dashboard.
"""


def test_srt_strips_timestamps_and_numbers():
    out = compress_transcript(SRT, filename="t.srt").output
    assert "-->" not in out
    assert "00:00:01" not in out
    assert "Hello everyone" in out


def test_filler_removed():
    assert "um" not in remove_filler("Hello, um, there").lower().split()
    out = compress_transcript(SRT, filename="t.srt").output
    assert " um" not in out.lower()


def test_speaker_merge_srt():
    out = compress_transcript(SRT, filename="t.srt").output
    # Alice's two consecutive cues merge into one line.
    alice_lines = [ln for ln in out.splitlines() if ln.startswith("Alice:")]
    assert len(alice_lines) == 1
    assert "welcome" in alice_lines[0] and "roadmap" in alice_lines[0]


def test_vtt_speaker_extraction():
    out = compress_transcript(VTT, filename="t.vtt").output
    assert out.startswith("Carol:")
    assert "demo" in out and "dashboard" in out


def test_parse_cues_count():
    assert len(parse_cues(SRT.decode())) == 3


def test_summary_mode_is_shorter():
    long_srt = b"\n\n".join(
        f"{i}\n00:00:0{i},000 --> 00:00:0{i+1},000\nSpeaker: Distinct sentence number {i} about topic {i}.".encode()
        for i in range(1, 9)
    )
    full = compress_transcript(long_srt, filename="t.srt")
    summ = compress_transcript(long_srt, filename="t.srt", summary=True, ratio=0.3)
    assert len(summ.output) < len(full.output)
