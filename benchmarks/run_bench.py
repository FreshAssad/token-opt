#!/usr/bin/env python3
"""Regenerate the measured-savings table from benchmarks/corpus/.

Run:  python benchmarks/run_bench.py
This prints a Markdown table and, if README.md has the BENCH markers, rewrites
the table in place. Numbers are produced the same way the CLI produces them, so
claims stay honest and reproducible.

Note: counts are exact when tiktoken's vocabulary is available; on a fully
air-gapped box they fall back to a labelled heuristic (marked *approx*).
"""
from __future__ import annotations

from pathlib import Path

from tokenopt.pipe import run

CORPUS = Path(__file__).resolve().parent / "corpus"
README = Path(__file__).resolve().parent.parent / "README.md"
START = "<!-- BENCH:START -->"
END = "<!-- BENCH:END -->"


def bench() -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    skipped: list[str] = []
    for f in sorted(CORPUS.glob("*")):
        if not f.is_file():
            continue
        try:
            result = run(str(f), f.read_bytes(), model="gpt-4o")
        except Exception as exc:  # e.g. a document backend isn't installed
            skipped.append(f"`{f.name}` ({str(exc)[:80]})")
            continue
        rep = result.report
        rows.append({
            "file": f.name,
            "category": result.category,
            "chosen": _chosen_note(result),
            "before": rep.before,
            "after": rep.after,
            "pct": rep.pct,
            "exact": rep.exact,
        })
    return rows, skipped


def _chosen_note(result) -> str:
    for note in result.warnings:
        if "JSON" in note or "TOON" in note:
            return "json-fallback" if "compact JSON" in note else "toon"
    return {"data": "toon", "doc": "markdown"}.get(result.category, result.category)


def render(rows: list[dict], skipped: list[str]) -> str:
    head = "| File | Type | Result | Before | After | Saved |"
    sep = "|---|---|---|---:|---:|---:|"
    lines = [head, sep]
    for r in rows:
        approx = "" if r["exact"] else " ¹"
        lines.append(
            f"| `{r['file']}` | {r['category']} | {r['chosen']} | "
            f"{r['before']:,} | {r['after']:,} | **{r['pct']:.1f}%**{approx} |"
        )
    out = "\n".join(lines)
    if any(not r["exact"] for r in rows):
        out += "\n\n¹ heuristic estimate (tiktoken vocab unavailable in this environment)."
    if skipped:
        out += "\n\n_Skipped (backend not installed): " + ", ".join(skipped) + "._"
    return out


def update_readme(table: str) -> bool:
    if not README.exists():
        return False
    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        return False
    pre = text.split(START)[0]
    post = text.split(END)[1]
    README.write_text(pre + START + "\n" + table + "\n" + END + post, encoding="utf-8")
    return True


def main() -> None:
    rows, skipped = bench()
    table = render(rows, skipped)
    print(table)
    if update_readme(table):
        print("\n[run_bench] README.md table updated.")
    else:
        print("\n[run_bench] (README markers not found — printed only.)")


if __name__ == "__main__":
    main()
