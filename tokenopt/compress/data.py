"""JSON -> TOON (Token-Oriented Object Notation) with a JSON fallback guard.

TOON's big win is *uniform arrays of objects*: declare the field names once and
stream rows, dropping repeated keys, braces, and quotes. On nested or
non-uniform data that win evaporates — sometimes TOON is even larger. So this
module always measures both and emits whichever is smaller (lossless either
way). We never market a fixed multiplier.

The encoder is a small, deterministic transform (the mature TOON SDK is
TypeScript; this is a focused Python re-implementation of the parts that pay
off, with embedded compact-JSON for structures TOON can't tabularize).
"""
from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field

INDENT = "  "  # two spaces per level


# --------------------------------------------------------------------------- #
# Scalar / key formatting
# --------------------------------------------------------------------------- #
def _is_scalar(v) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def _looks_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _needs_quote(s: str) -> bool:
    if s == "":
        return True
    if s != s.strip():  # leading/trailing whitespace
        return True
    if s.lower() in ("true", "false", "null"):
        return True
    if _looks_numeric(s):
        return True
    return any(ch in s for ch in ',:{}[]"\n\t\r#')


def _fmt_number(v) -> str:
    if isinstance(v, bool):  # bool is a subclass of int — handle first
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return json.dumps(v)  # "NaN"/"Infinity" (rare; round-trips for LLMs)
        # Compact but lossless float repr.
        return repr(v)
    return str(v)


def _fmt_scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return _fmt_number(v)
    s = str(v)
    return json.dumps(s, ensure_ascii=False) if _needs_quote(s) else s


def _fmt_key(k) -> str:
    k = str(k)
    if k == "" or _needs_quote(k) or " " in k:
        return json.dumps(k, ensure_ascii=False)
    return k


# --------------------------------------------------------------------------- #
# Array shape analysis
# --------------------------------------------------------------------------- #
def _all_scalar(arr: list) -> bool:
    return all(_is_scalar(x) for x in arr)


def _uniform_flat_object_keys(arr: list):
    """If arr is a non-empty list of dicts that all share the same keys and
    whose values are all scalar, return the ordered key list; else None."""
    if not arr or not all(isinstance(x, dict) for x in arr):
        return None
    keys = list(arr[0].keys())
    keyset = set(keys)
    for obj in arr:
        if set(obj.keys()) != keyset:
            return None
        if not all(_is_scalar(v) for v in obj.values()):
            return None
    return keys


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
def _encode_kv(key, value, lines: list[str], level: int) -> None:
    pad = INDENT * level
    fk = _fmt_key(key)
    if _is_scalar(value):
        lines.append(f"{pad}{fk}: {_fmt_scalar(value)}")
    elif isinstance(value, dict):
        if not value:
            lines.append(f"{pad}{fk}: {{}}")
        else:
            lines.append(f"{pad}{fk}:")
            for k2, v2 in value.items():
                _encode_kv(k2, v2, lines, level + 1)
    elif isinstance(value, list):
        _encode_array(fk, value, lines, level)
    else:  # pragma: no cover - non-JSON type
        lines.append(f"{pad}{fk}: {json.dumps(value, ensure_ascii=False)}")


def _encode_array(prefix: str | None, arr: list, lines: list[str], level: int) -> None:
    pad = INDENT * level
    head = prefix if prefix is not None else ""
    n = len(arr)

    if n == 0:
        lines.append(f"{pad}{head}[0]:")
        return

    if _all_scalar(arr):
        inline = ",".join(_fmt_scalar(x) for x in arr)
        lines.append(f"{pad}{head}[{n}]: {inline}")
        return

    keys = _uniform_flat_object_keys(arr)
    if keys is not None:
        header = ",".join(_fmt_key(k) for k in keys)
        lines.append(f"{pad}{head}[{n}]{{{header}}}:")
        row_pad = INDENT * (level + 1)
        for obj in arr:
            lines.append(row_pad + ",".join(_fmt_scalar(obj[k]) for k in keys))
        return

    # Fallback: non-uniform / nested array. Emit one compact-JSON item per line.
    # Lossless; the guard below will usually prefer plain JSON for these shapes.
    lines.append(f"{pad}{head}[{n}]:")
    item_pad = INDENT * (level + 1)
    for item in arr:
        lines.append(item_pad + json.dumps(item, ensure_ascii=False, separators=(",", ":")))


def encode_toon(value) -> str:
    """Encode a parsed JSON value to TOON text (2-space indentation)."""
    lines: list[str] = []
    if isinstance(value, dict):
        if not value:
            return "{}"
        for k, v in value.items():
            _encode_kv(k, v, lines, 0)
    elif isinstance(value, list):
        _encode_array(None, value, lines, 0)
    else:
        lines.append(_fmt_scalar(value))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CSV (flat tabular only)
# --------------------------------------------------------------------------- #
class NotTabular(Exception):
    pass


def _rows_for_csv(value):
    """Return a list-of-flat-dicts suitable for CSV, or raise NotTabular."""
    candidate = value
    if isinstance(value, dict):
        # {"items": [ ... ]} with a single array value.
        arrays = [v for v in value.values() if isinstance(v, list)]
        if len(value) == 1 and arrays:
            candidate = arrays[0]
        else:
            raise NotTabular("top-level object is not a single array of rows")
    if not isinstance(candidate, list) or not candidate:
        raise NotTabular("not a non-empty array")
    if not all(isinstance(x, dict) for x in candidate):
        raise NotTabular("array elements are not all objects")
    if not all(all(_is_scalar(v) for v in row.values()) for row in candidate):
        raise NotTabular("rows contain nested values")
    return candidate


def _csv_cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):  # keep booleans LLM/JSON-consistent (lowercase)
        return "true" if v else "false"
    return v  # csv writer str()s ints/floats/strings


def encode_csv(value) -> str:
    rows = _rows_for_csv(value)
    # Ordered union of keys (first-seen order).
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _csv_cell(row.get(k)) for k in fields})
    return buf.getvalue().rstrip("\r\n")


# --------------------------------------------------------------------------- #
# Top-level: compress
# --------------------------------------------------------------------------- #
@dataclass
class DataResult:
    output: str
    chosen_format: str  # "toon" | "json" | "csv"
    tokens: dict = field(default_factory=dict)  # {original, toon, json, chosen}
    notes: list[str] = field(default_factory=list)


def _count(text: str, model: str) -> int:
    from ..counting import count

    return count(text, model).tokens


def compress_data(
    json_text: str,
    *,
    fmt: str = "toon",
    model: str = "gpt-4o",
) -> DataResult:
    """Compress JSON text.

    fmt="toon": emit TOON, but fall back to compact JSON if TOON isn't smaller.
    fmt="csv":  emit CSV for flat tabular data; otherwise warn and use the
                toon/json path.
    """
    obj = json.loads(json_text)
    compact_json = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    orig_tokens = _count(json_text, model)
    json_tokens = _count(compact_json, model)
    notes: list[str] = []

    if fmt == "csv":
        try:
            csv_text = encode_csv(obj)
            csv_tokens = _count(csv_text, model)
            return DataResult(
                output=csv_text,
                chosen_format="csv",
                tokens={"original": orig_tokens, "csv": csv_tokens, "chosen": csv_tokens},
                notes=notes,
            )
        except NotTabular as exc:
            notes.append(f"CSV not applicable ({exc}); using TOON/JSON instead.")
            # fall through to toon path

    toon_text = encode_toon(obj)
    toon_tokens = _count(toon_text, model)

    tokens = {"original": orig_tokens, "toon": toon_tokens, "json": json_tokens}
    if toon_tokens <= json_tokens:
        tokens["chosen"] = toon_tokens
        return DataResult(output=toon_text, chosen_format="toon", tokens=tokens, notes=notes)

    notes.append(
        f"TOON not beneficial for this shape ({toon_tokens} vs {json_tokens} "
        "tokens) — emitting compact JSON."
    )
    tokens["chosen"] = json_tokens
    return DataResult(output=compact_json, chosen_format="json", tokens=tokens, notes=notes)
