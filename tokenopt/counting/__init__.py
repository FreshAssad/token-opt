"""Token counting across providers.

Public surface:
    from tokenopt.counting import count, CountResult
"""
from .base import CountResult, count, resolve_family

__all__ = ["CountResult", "count", "resolve_family"]
