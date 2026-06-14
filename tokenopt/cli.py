"""token-opt command-line interface (typer).

Conventions:
  * Compressed payload -> stdout (clean, pipe-friendly).
  * Diagnostics (before/after, cost) -> stderr.
  * Every command reads a FILE, a DIR, or '-' for stdin.
  * --json everywhere for machine-readable output.
"""
from __future__ import annotations

import json as _json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .counting import count as count_tokens
from .cost import PriceError, estimate_cost, load_prices, resolve_price_key
from .detect import detect
from .report import print_report

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Locally & deterministically compress and count tokens before you send "
        "text/code/docs/data to ANY LLM. Offline-first; honest before/after numbers."
    ),
)
compress_app = typer.Typer(no_args_is_help=True, help="Compress content by type.")
app.add_typer(compress_app, name="compress")


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def _read_bytes(target: str) -> tuple[bytes, Optional[str]]:
    """Return (data, path). path is None for stdin."""
    if target == "-":
        return sys.stdin.buffer.read(), None
    p = Path(target)
    if not p.exists():
        typer.echo(f"[token-opt] error: no such file or directory: {target}", err=True)
        raise typer.Exit(code=1)
    return p.read_bytes(), str(p)


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


_DEFAULT_SKIP = {
    ".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache",
    ".pytest_cache", ".idea", ".vscode", "dist", "build", ".tox",
}


def _iter_files(root: Path):
    """Yield files under root, respecting .gitignore when inside a git repo."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others",
             "--exclude-standard"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0 and out.stdout.strip():
            for line in out.stdout.splitlines():
                p = root / line
                if p.is_file():
                    yield p
            return
    except Exception:
        pass
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _DEFAULT_SKIP]
        for fn in filenames:
            yield Path(dirpath) / fn


def _emit_stdout(text: str) -> None:
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")


# --------------------------------------------------------------------------- #
# Root callback (--version)
# --------------------------------------------------------------------------- #
def _version_cb(value: bool):
    if value:
        typer.echo(f"token-opt {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", callback=_version_cb, is_eager=True,
        help="Show version and exit.",
    ),
):
    pass


# --------------------------------------------------------------------------- #
# count
# --------------------------------------------------------------------------- #
@app.command()
def count(
    target: str = typer.Argument("-", metavar="FILE|DIR|-"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="claude|gpt-4o|gemini|llama|..."),
    api: bool = typer.Option(False, "--api", help="Use provider API for an exact count (needs key)."),
    opus_correction: bool = typer.Option(False, "--opus-correction", help="Apply ~1.3x to offline Claude estimates."),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output."),
):
    """Count tokens for a file, a directory, or stdin."""
    p = Path(target) if target != "-" else None

    if p and p.is_dir():
        files = sorted(_iter_files(p))
        rows = []
        total = 0
        for f in files:
            text = _decode(f.read_bytes())
            r = count_tokens(text, model, use_api=api, opus_correction=opus_correction)
            total += r.tokens
            rows.append((str(f), r))
        if json_out:
            payload = {
                "model": model,
                "files": [{"path": path, **r.as_dict()} for path, r in rows],
                "total_tokens": total,
            }
            _emit_stdout(_json.dumps(payload, ensure_ascii=False, indent=2))
            return
        for path, r in rows:
            tilde = "" if r.exact else "~"
            print(f"{tilde}{r.tokens:>9,}  {path}")
        exact_all = all(r.exact for _, r in rows)
        print(f"{'' if exact_all else '~'}{total:>9,}  TOTAL ({len(rows)} files, {model})")
        return

    data, path = _read_bytes(target)
    text = _decode(data)
    if detect(path, data) == "doc" and (data.startswith(b"%PDF") or data.startswith(b"PK\x03\x04")):
        print(
            "[token-opt] note: this looks like a binary document; "
            "`token-opt compress doc` will give a meaningful count.",
            file=sys.stderr,
        )
    r = count_tokens(text, model, use_api=api, opus_correction=opus_correction)
    if json_out:
        out = {"source": path or "stdin", **r.as_dict()}
        _emit_stdout(_json.dumps(out, ensure_ascii=False, indent=2))
        return
    tilde = "" if r.exact else "~"
    print(f"{tilde}{r.tokens:,} tokens  ({r.qualifier})  [model={r.model}, via {r.method}]")


# --------------------------------------------------------------------------- #
# cost
# --------------------------------------------------------------------------- #
@app.command()
def cost(
    target: str = typer.Argument("-", metavar="FILE|-"),
    model: str = typer.Option("gpt-4o", "--model", "-m"),
    monthly_queries: Optional[int] = typer.Option(None, "--monthly-queries", help="Project monthly spend."),
    compare: Optional[str] = typer.Option(None, "--compare", help="Comma list, e.g. claude,gpt-4o,gemini."),
    api: bool = typer.Option(False, "--api"),
    json_out: bool = typer.Option(False, "--json"),
):
    """Estimate input (prompt) cost. Output cost is unknown until the model replies."""
    data, path = _read_bytes(target)
    text = _decode(data)
    prices = load_prices()

    models = [m.strip() for m in compare.split(",")] if compare else [model]
    rows = []
    for m in models:
        r = count_tokens(text, m, use_api=api)
        try:
            c = estimate_cost(r.tokens, m, prices=prices)
            in_cost = c["input_cost"]
            in_rate = c["input_rate"]
            priced = True
        except PriceError:
            in_cost, in_rate, priced = 0.0, 0.0, False
        monthly = in_cost * monthly_queries if monthly_queries else None
        rows.append({
            "model": m, "resolved": resolve_price_key(m, prices),
            "tokens": r.tokens, "exact": r.exact, "input_rate": in_rate,
            "input_cost": in_cost, "monthly_cost": monthly, "priced": priced,
        })

    if json_out:
        _emit_stdout(_json.dumps(
            {"source": path or "stdin", "monthly_queries": monthly_queries, "rows": rows},
            ensure_ascii=False, indent=2,
        ))
        return

    has_monthly = monthly_queries is not None
    header = f"{'MODEL':<20}{'TOKENS':>10}  {'$/1M-IN':>8}  {'INPUT $':>10}"
    if has_monthly:
        header += f"  {'MONTHLY $':>11}"
    print(header)
    print("-" * len(header))
    for row in rows:
        tk = f"{'' if row['exact'] else '~'}{row['tokens']:,}"
        rate = f"{row['input_rate']:.2f}" if row["priced"] else "n/a"
        cost_s = f"${row['input_cost']:.4f}" if row["priced"] else "n/a"
        line = f"{row['model']:<20}{tk:>10}  {rate:>8}  {cost_s:>10}"
        if has_monthly:
            line += f"  {('$'+format(row['monthly_cost'], '.2f')) if row['priced'] else 'n/a':>11}"
        print(line)
    if any(not r["exact"] for r in rows):
        print("\n~ = token count is an estimate (e.g. offline Claude); use --api for exact.", file=sys.stderr)


# --------------------------------------------------------------------------- #
# compress (shared engine via pipe.run)
# --------------------------------------------------------------------------- #
def _safe_run(path, data, **kwargs):
    """Call pipe.run, turning expected failures into clean stderr + exit 1."""
    from .compress.doc import DocConversionError
    from .pipe import run

    try:
        return run(path, data, **kwargs)
    except _json.JSONDecodeError as e:
        typer.echo(f"[token-opt] error: invalid JSON: {e}", err=True)
        raise typer.Exit(code=1)
    except (DocConversionError, RuntimeError, ValueError, OSError) as e:
        typer.echo(f"[token-opt] error: {e}", err=True)
        raise typer.Exit(code=1)


def _emit_compress(result, *, json_out: bool, quiet: bool) -> None:
    if json_out:
        payload = {"category": result.category, "output": result.output, **result.report.as_dict()}
        _emit_stdout(_json.dumps(payload, ensure_ascii=False, indent=2))
        return
    _emit_stdout(result.output)
    print_report(result.report, quiet=quiet)


def _doc_impl(
    target: str = typer.Argument("-", metavar="FILE|-"),
    model: str = typer.Option("gpt-4o", "--model", "-m"),
    keep_tables: bool = typer.Option(True, "--keep-tables/--no-keep-tables", help="Preserve (default) or flatten pipe tables."),
    strip_bibliography: bool = typer.Option(False, "--strip-bibliography", help="Drop a trailing References/Bibliography section."),
    api: bool = typer.Option(False, "--api"),
    json_out: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress the stderr report."),
):
    """PDF/DOCX/HTML/PPTX/XLSX -> clean Markdown."""
    data, path = _read_bytes(target)
    result = _safe_run(
        path, data, model=model, keep_tables=keep_tables,
        strip_bibliography=strip_bibliography, use_api=api, force="doc",
    )
    _emit_compress(result, json_out=json_out, quiet=quiet)


def _data_impl(
    target: str = typer.Argument("-", metavar="FILE|-"),
    model: str = typer.Option("gpt-4o", "--model", "-m"),
    fmt: str = typer.Option("toon", "--format", help="toon|csv"),
    api: bool = typer.Option(False, "--api"),
    json_out: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
):
    """JSON -> TOON (falls back to compact JSON if TOON isn't smaller)."""
    data, path = _read_bytes(target)
    result = _safe_run(path, data, model=model, data_format=fmt, use_api=api, force="data")
    _emit_compress(result, json_out=json_out, quiet=quiet)


compress_app.command("doc")(_doc_impl)
compress_app.command("data")(_data_impl)
# Colon aliases read nicely: `token-opt compress:doc file.pdf`.
app.command("compress:doc", hidden=True)(_doc_impl)
app.command("compress:data", hidden=True)(_data_impl)


# v2 stubs — present in the command surface, clearly deferred.
def _make_v2_stub(name: str):
    def _stub(target: str = typer.Argument("-", metavar="FILE|-")):
        print(
            f"[token-opt] 'compress {name}' is a v2 feature and not implemented "
            f"in v1. Try `token-opt pipe {target}` for a lossless pass.",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)

    _stub.__name__ = f"compress_{name}"
    _stub.__doc__ = f"[v2] compress {name} — not yet implemented."
    return _stub


for _name in ("code", "email", "transcript", "generic"):
    _fn = _make_v2_stub(_name)
    compress_app.command(_name)(_fn)
    app.command(f"compress:{_name}", hidden=True)(_fn)


# --------------------------------------------------------------------------- #
# pipe
# --------------------------------------------------------------------------- #
@app.command()
def pipe(
    target: str = typer.Argument("-", metavar="FILE|-"),
    model: str = typer.Option("gpt-4o", "--model", "-m"),
    fmt: str = typer.Option("toon", "--format", help="data format when routing to JSON: toon|csv"),
    keep_tables: bool = typer.Option(True, "--keep-tables/--no-keep-tables"),
    strip_bibliography: bool = typer.Option(False, "--strip-bibliography"),
    api: bool = typer.Option(False, "--api"),
    json_out: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
):
    """Auto-detect the input type, route to the right compressor, and report."""
    data, path = _read_bytes(target)
    result = _safe_run(
        path, data, model=model, keep_tables=keep_tables,
        strip_bibliography=strip_bibliography, data_format=fmt, use_api=api,
    )
    _emit_compress(result, json_out=json_out, quiet=quiet)


if __name__ == "__main__":  # pragma: no cover
    app()
