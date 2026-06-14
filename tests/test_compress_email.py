"""Email thread dedup (stdlib only — always runs offline)."""
from __future__ import annotations

from tokenopt.compress.email import compress_email

EML = b"""From: alice@example.com
To: bob@example.com
Subject: Update
Date: Mon, 1 Jun 2026 10:00:00 +0000

Please review the draft by Friday.

On Sun, 31 May 2026, Bob wrote:
> Sure, will do.
>> Earlier question about timeline.

--
Alice
Senior PM
"""


def test_drops_quoted_history_and_signature():
    res = compress_email(EML, filename="t.eml")
    assert "Please review the draft by Friday." in res.output
    assert "Sure, will do." not in res.output  # quoted -> dropped
    assert "Earlier question" not in res.output
    assert "Senior PM" not in res.output  # signature -> dropped
    assert "Subject: Update" in res.output


def test_keep_quotes_preserves_quoted_lines():
    res = compress_email(EML, filename="t.eml", keep_quotes=True)
    assert "Sure, will do." in res.output
    assert "Earlier question about timeline." in res.output


def test_dedup_across_repeated_content():
    # Same line repeated (as quoted history would be) collapses to one.
    raw = b"From: x@y\nSubject: S\n\nUnique line.\nUnique line.\n> Unique line.\n"
    res = compress_email(raw, filename="t.eml")
    assert res.output.count("Unique line.") == 1


def test_raw_text_without_headers():
    res = compress_email(b"just a plain note\nwith two lines\n")
    assert "just a plain note" in res.output
    assert res.messages == 1
