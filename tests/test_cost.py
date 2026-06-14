"""Cost math and price-table resolution."""
from __future__ import annotations

import json

import pytest

from tokenopt.cost import PriceError, estimate_cost, load_prices, resolve_price_key


def test_prices_load_and_have_models():
    prices = load_prices()
    assert "models" in prices and prices["models"]
    assert "gpt-4o" in prices["models"]


def test_alias_resolution():
    prices = load_prices()
    assert resolve_price_key("gpt", prices) == "gpt-4o"
    assert resolve_price_key("claude", prices) == "claude-3-5-sonnet"
    assert resolve_price_key("nope-9000", prices) is None


def test_estimate_cost_input_math():
    prices = {"models": {"m": {"input": 2.0, "output": 8.0}}, "aliases": {}}
    c = estimate_cost(1_000_000, "m", prices=prices)
    assert c["input_cost"] == pytest.approx(2.0)
    assert c["total_cost"] == pytest.approx(2.0)  # no output tokens given


def test_estimate_cost_with_output():
    prices = {"models": {"m": {"input": 2.0, "output": 8.0}}, "aliases": {}}
    c = estimate_cost(1_000_000, "m", output_tokens=500_000, prices=prices)
    assert c["output_cost"] == pytest.approx(4.0)
    assert c["total_cost"] == pytest.approx(6.0)


def test_unknown_model_raises():
    with pytest.raises(PriceError):
        estimate_cost(100, "does-not-exist", prices={"models": {}, "aliases": {}})


def test_env_override(tmp_path, monkeypatch):
    custom = tmp_path / "p.json"
    custom.write_text(json.dumps({"models": {"x": {"input": 99.0, "output": 1.0}}, "aliases": {}}))
    monkeypatch.setenv("TOKENOPT_PRICES", str(custom))
    load_prices.cache_clear()  # lru_cache
    prices = load_prices()
    assert prices["models"]["x"]["input"] == 99.0
    load_prices.cache_clear()
