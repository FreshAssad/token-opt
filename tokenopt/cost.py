"""Cost estimation from token counts and a bundled price snapshot.

We can only know the *input* (prompt) size for content the user hands us — the
response length is unknown until the model answers. So cost figures here are
**input cost** unless you supply an expected output size. This is stated plainly
rather than guessed.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path


class PriceError(Exception):
    pass


@lru_cache(maxsize=4)
def load_prices(path: str | None = None) -> dict:
    """Load the price table.

    Resolution order:
      1. explicit ``path`` argument,
      2. ``TOKENOPT_PRICES`` environment variable,
      3. a ``prices.json`` in the current working directory,
      4. the snapshot bundled inside the package.
    """
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    env = os.environ.get("TOKENOPT_PRICES")
    if env:
        candidates.append(Path(env))
    candidates.append(Path.cwd() / "prices.json")

    for cand in candidates:
        if cand.is_file():
            return json.loads(cand.read_text(encoding="utf-8"))

    # Bundled default.
    from importlib.resources import files

    text = files("tokenopt").joinpath("prices.json").read_text(encoding="utf-8")
    return json.loads(text)


def resolve_price_key(model: str, prices: dict) -> str | None:
    """Map a (possibly aliased) model name to a key in prices['models']."""
    models = prices.get("models", {})
    aliases = prices.get("aliases", {})
    if model in models:
        return model
    if model in aliases and aliases[model] in models:
        return aliases[model]
    low = (model or "").lower()
    if low in models:
        return low
    if low in aliases and aliases[low] in models:
        return aliases[low]
    return None


def price_for(model: str, prices: dict | None = None) -> dict | None:
    prices = prices or load_prices()
    key = resolve_price_key(model, prices)
    if key is None:
        return None
    return prices["models"][key]


def estimate_cost(
    input_tokens: int,
    model: str,
    *,
    output_tokens: int = 0,
    prices: dict | None = None,
) -> dict:
    """Return a cost breakdown for ``model``.

    Raises PriceError if the model is unknown to the price table.
    """
    prices = prices or load_prices()
    p = price_for(model, prices)
    if p is None:
        raise PriceError(f"No price entry for model '{model}'.")
    in_rate = float(p.get("input", 0.0))
    out_rate = float(p.get("output", 0.0))
    in_cost = input_tokens / 1_000_000 * in_rate
    out_cost = output_tokens / 1_000_000 * out_rate
    return {
        "model": model,
        "resolved": resolve_price_key(model, prices),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_rate": in_rate,
        "output_rate": out_rate,
        "input_cost": in_cost,
        "output_cost": out_cost,
        "total_cost": in_cost + out_cost,
    }


def cost_per_tokens(tokens: int, model: str, prices: dict | None = None) -> float:
    """Convenience: just the input-side dollar cost for ``tokens`` tokens."""
    try:
        return estimate_cost(tokens, model, prices=prices)["input_cost"]
    except PriceError:
        return 0.0
