"""[v2] Source code -> AST-based minification (tree-sitter).

Deferred. Tracked in the build spec (§9). Importing/calling raises a clear
message so the CLI can degrade gracefully.
"""
from __future__ import annotations


class NotYetImplemented(NotImplementedError):
    pass


def compress_code(*_args, **_kwargs):
    raise NotYetImplemented(
        "compress code is planned for v2 (tree-sitter AST). For now, pipe code "
        "through `token-opt count` or `token-opt pipe` for a lossless pass."
    )
