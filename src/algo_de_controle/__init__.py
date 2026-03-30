"""Control algo package for potential-seeking experiments.

Provides a manager that orchestrates initial placement, gradient-based
search using the existing `gradient_seeker`, detection of local maxima,
and an ascending/cruise mode when a maximum is found.
"""

from .manager import ControlManager

__all__ = ["ControlManager"]
