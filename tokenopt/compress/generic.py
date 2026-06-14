"""[v2] Generic prose -> extractive summary (spaCy + pytextrank).

LOSSY and opt-in. Deferred (build spec §9). A *lossless* whitespace cleanup for
prose lives in ``tokenopt.pipe`` and is what `pipe` uses by default.
"""
from __future__ import annotations


class NotYetImplemented(NotImplementedError):
    pass


def compress_generic(*_args, **_kwargs):
    raise NotYetImplemented(
        "compress generic (lossy extractive prose) is planned for v2."
    )
