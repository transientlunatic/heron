"""
Backwards-compatible imports for the old gpytorch module.

The canonical location is now heron.models.gp.exact.
"""

from .gp.exact import ExactGPSurrogate, ExactGPSurrogate as HeronNonSpinningApproximantMatern

__all__ = ["ExactGPSurrogate", "HeronNonSpinningApproximantMatern"]
