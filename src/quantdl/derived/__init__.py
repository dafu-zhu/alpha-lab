"""
Derived Data Module

This module contains logic for computing derived metrics from raw data.

Submodules:
- metrics: Compute derived fundamental metrics (margins, returns, growth, etc.)
- ttm: Compute trailing twelve months (TTM) metrics
"""

from .metrics import compute_derived
from .ttm import compute_ttm_long

__all__ = ['compute_derived', 'compute_ttm_long']
