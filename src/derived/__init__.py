"""
Derived Data Module

This module contains logic for computing derived metrics from raw data.

Submodules:
- fundamental: Compute derived fundamental metrics (margins, returns, growth, etc.)
"""

from .fundamental import compute_derived

__all__ = ['compute_derived']
