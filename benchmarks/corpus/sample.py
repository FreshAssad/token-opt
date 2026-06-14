#!/usr/bin/env python3
# Demo module for the token-opt benchmark corpus.
# It is intentionally over-commented so `compress code` has something to remove.

"""A tiny inventory helper.

This module-level docstring is kept by `minify` (it's a real string literal),
but every ``#`` comment below is stripped.
"""

import math  # standard library


# ---- data model ----------------------------------------------------------
class Item:
    # An item has a name and a unit price.
    def __init__(self, name, price):
        self.name = name  # display name
        self.price = price  # price per unit in USD

    def total(self, qty):
        # multiply price by quantity, rounding up to the cent
        return math.ceil(self.price * qty * 100) / 100  # cents


def subtotal(items, quantities):
    # Sum the totals for a parallel list of items and quantities.
    acc = 0.0
    for item, qty in zip(items, quantities):
        acc += item.total(qty)  # accumulate
    return acc  # final subtotal
