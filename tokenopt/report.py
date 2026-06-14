"""Human-readable before/after reporting (goes to stderr, never stdout).

Keeping diagnostics on stderr is what makes token-opt pipe-friendly: the
compressed payload on stdout stays clean for the next command in the pipeline.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass


def pct_saved(before: int, after: int) -> float:
    if before <= 0:
        return 0.0
    return (1.0 - after / before) * 100.0


@dataclass
class Report:
    before: int
    after: int
    model: str
    exact: bool = True
    cost_saved: float | None = None
    notes: list[str] | None = None

    @property
    def saved_tokens(self) -> int:
        return self.before - self.after

    @property
    def pct(self) -> float:
        return pct_saved(self.before, self.after)

    def render(self) -> str:
        p = self.pct
        sign = "-" if p >= 0 else "+"
        line = (
            f"[token-opt] {self.model}: "
            f"{self.before:,} -> {self.after:,} tokens "
            f"({sign}{abs(p):.1f}%)"
        )
        if self.cost_saved is not None and self.cost_saved != 0:
            verb = "saved" if self.cost_saved >= 0 else "added"
            line += f"  est. ${abs(self.cost_saved):.4f} {verb}"
        if not self.exact:
            line += "  (estimate)"
        parts = [line]
        for note in self.notes or []:
            parts.append(f"           {note}")
        return "\n".join(parts)

    def as_dict(self) -> dict:
        return {
            "before_tokens": self.before,
            "after_tokens": self.after,
            "saved_tokens": self.saved_tokens,
            "pct_saved": round(self.pct, 2),
            "model": self.model,
            "exact": self.exact,
            "cost_saved": self.cost_saved,
            "notes": self.notes or [],
        }


def print_report(report: Report, *, quiet: bool = False) -> None:
    if quiet:
        return
    print(report.render(), file=sys.stderr)


def warn(msg: str, *, quiet: bool = False) -> None:
    if quiet:
        return
    print(f"[token-opt] warning: {msg}", file=sys.stderr)
