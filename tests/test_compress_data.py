"""TOON encoding, CSV, and the JSON-fallback guard."""
from __future__ import annotations

import json

from tokenopt.compress import data as D
from tokenopt.compress.data import compress_data, encode_csv, encode_toon


def test_uniform_array_is_tabular():
    obj = {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
    toon = encode_toon(obj)
    lines = toon.splitlines()
    assert lines[0] == "users[2]{id,name}:"
    assert lines[1] == "  1,Alice"
    assert lines[2] == "  2,Bob"


def test_scalar_array_is_inline():
    assert encode_toon({"nums": [1, 2, 3]}) == "nums[3]: 1,2,3"


def test_nested_object_uses_indentation():
    toon = encode_toon({"a": {"b": {"c": 1}}})
    assert toon == "a:\n  b:\n    c: 1"


def test_string_quoting_rules():
    # Commas, leading space, and literal-looking strings must be quoted.
    out = encode_toon({"v": ["a,b", " lead", "true", "x"]})
    assert '"a,b"' in out
    assert '" lead"' in out
    assert '"true"' in out
    assert out.endswith(",x")  # plain string stays bare


def test_compress_data_picks_toon_when_smaller():
    payload = json.dumps({"rows": [{"a": i, "b": i * 2, "c": "x"} for i in range(20)]})
    res = compress_data(payload, model="gpt-4o")
    assert res.chosen_format == "toon"
    assert res.tokens["chosen"] <= res.tokens["json"]


def test_guard_falls_back_to_json(monkeypatch):
    # Force a length-based counter so TOON's newlines/indent lose to compact JSON
    # on a deeply nested, non-tabular shape -> guard must emit JSON.
    monkeypatch.setattr(D, "_count", lambda text, model: len(text))
    nested = json.dumps({"a": {"b": {"c": {"d": {"e": [1, [2, [3, [4]]]]}}}}})
    res = compress_data(nested, model="gpt-4o")
    assert res.chosen_format == "json"
    assert any("not beneficial" in n for n in res.notes)
    # Output must be valid, lossless JSON.
    assert json.loads(res.output) == json.loads(nested)


def test_csv_export_lowercases_bools():
    obj = {"rows": [{"id": 1, "ok": True}, {"id": 2, "ok": False}]}
    csv_text = encode_csv(obj)
    assert csv_text.splitlines()[0] == "id,ok"
    assert "1,true" in csv_text
    assert "2,false" in csv_text


def test_csv_quotes_commas():
    obj = [{"name": "a,b"}]
    assert '"a,b"' in encode_csv(obj)
