"""
CDC variable mapping, ontological registry, and alias normalization.

Submodules:
    - registry: Canonical variable and concept registry.
    - alias_mapper: Regex and surface-text matcher to canonical variables.
    - report: Mapping coverage and unmapped entity frequency reporting.
"""

from src.mapping.alias_mapper import AliasMapper
from src.mapping.registry import CanonicalRegistry

__all__ = [
    "AliasMapper",
    "CanonicalRegistry",
]
