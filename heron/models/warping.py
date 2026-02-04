"""
Time warping functions for GP regression on gravitational waveforms.

Physical motivation: Gravitational waveforms evolve on different timescales
during inspiral vs merger/ringdown. The chirp time τ(f) ∝ (M*eta)^(-5/8) * f^(-8/3)
describes the time-to-merger as a function of frequency. By warping time coordinates
to better match this physical evolution, we can:

1. Reduce training data requirements (fewer points needed)
2. Improve GP interpolation near merger (more uniform in warped space)
3. Better capture the rapid evolution near merger

Types of warping:
- Simple: Linear compression of inspiral (t_warp = t / scale for t < 0)
- Chirp time: Physical warping based on PN inspiral timescale
- Hybrid: Learned warping on top of physical warping
"""

import torch
import numpy as np


class TimeWarping:
    """Base class for time coordinate warping."""

    def warp(self, t):
        """Transform physical time to warped time (for training/prediction)."""
        raise NotImplementedError

    def unwarp(self, t_warp):
        """Transform warped time back to physical time (for output)."""
        raise NotImplementedError


class SimpleWarping(TimeWarping):
    """
    Simple linear compression of negative times (current implementation).

    For t < 0: t_warp = t / scale (compress inspiral)
    For t >= 0: t_warp = t (no change to ringdown)

    Parameters:
    -----------
    scale : float
        Compression factor for negative times (default: 2)
    """

    def __init__(self, scale=2):
        self.scale = scale

    def warp(self, t):
        """Warp time coordinates."""
        t_warp = t.clone() if torch.is_tensor(t) else np.copy(t)
        mask = t_warp < 0
        t_warp[mask] = t_warp[mask] / self.scale
        return t_warp

    def unwarp(self, t_warp):
        """Unwarp time coordinates."""
        t = t_warp.clone() if torch.is_tensor(t_warp) else np.copy(t_warp)
        mask = t < 0
        t[mask] = t[mask] * self.scale
        return t


class ChirpTimeWarping(TimeWarping):
    """
    Physical warping based on post-Newtonian chirp time.

    The chirp time τ(f) ∝ (M*η)^(-5/8) * (π*M*f)^(-8/3) gives the time until
    merger as a function of GW frequency. This provides a natural warping that:

    - Maps early inspiral (slowly evolving) to larger warped time range
    - Maps late inspiral/merger (rapidly evolving) to smaller warped time range
    - Creates more uniform evolution in warped space

    For implementation, we use a power-law approximation:
        t_warp = sign(t) * |t|^alpha

    where alpha ≈ 3/8 corresponds to the Newtonian chirp time scaling.

    Parameters:
    -----------
    alpha : float
        Power-law exponent (default: 0.375 = 3/8 for Newtonian chirp time)
    t_ref : float
        Reference time for normalization (default: 0.1 seconds)
    """

    def __init__(self, alpha=0.375, t_ref=0.1):
        self.alpha = alpha
        self.t_ref = t_ref
        self.inv_alpha = 1.0 / alpha

    def warp(self, t):
        """
        Warp time using power-law: t_warp = sign(t) * |t|^alpha

        This compresses the late inspiral relative to early inspiral,
        creating more uniform spacing in the warped coordinate.
        """
        t_warp = torch.sign(t) * torch.abs(t / self.t_ref) ** self.alpha * self.t_ref
        return t_warp

    def unwarp(self, t_warp):
        """
        Unwarp time: t = sign(t_warp) * |t_warp|^(1/alpha)
        """
        t = torch.sign(t_warp) * torch.abs(t_warp / self.t_ref) ** self.inv_alpha * self.t_ref
        return t


class PiecewiseWarping(TimeWarping):
    """
    Piecewise warping with different scales for different time regions.

    This allows fine control over warping in different phases:
    - Early inspiral: mild compression
    - Late inspiral: stronger compression
    - Merger: no compression
    - Ringdown: mild expansion

    Parameters:
    -----------
    breakpoints : list of float
        Time boundaries for piecewise regions (e.g., [-0.1, -0.01, 0, 0.05])
    scales : list of float
        Compression factors for each region (length = len(breakpoints) - 1)
    """

    def __init__(self, breakpoints=None, scales=None):
        if breakpoints is None:
            # Default: early inspiral, late inspiral, merger/ringdown
            breakpoints = [-0.1, -0.01, 0.0, 0.05]
        if scales is None:
            # Default: compress inspiral, keep merger/ringdown
            scales = [1.5, 3.0, 1.0]  # [early, late, ringdown]

        assert len(scales) == len(breakpoints) - 1, \
            "Number of scales must be one less than number of breakpoints"

        self.breakpoints = breakpoints
        self.scales = scales

    def _find_region(self, t):
        """Find which piecewise region each time belongs to."""
        regions = torch.zeros_like(t, dtype=torch.long)
        for i, bp in enumerate(self.breakpoints[1:]):
            regions[t >= bp] = i + 1
        return regions

    def warp(self, t):
        """Apply piecewise warping."""
        t_warp = torch.zeros_like(t)
        regions = self._find_region(t)

        for i in range(len(self.scales)):
            mask = regions == i
            if mask.any():
                t_region = t[mask]
                t_ref = self.breakpoints[i]
                scale = self.scales[i]
                # Warp relative to region start
                t_warp[mask] = t_ref + (t_region - t_ref) / scale

        return t_warp

    def unwarp(self, t_warp):
        """Undo piecewise warping."""
        # This requires inverting the piecewise transformation
        # For now, implement simple version
        raise NotImplementedError("Piecewise unwarping not yet implemented")


class AdaptiveWarping(TimeWarping):
    """
    Learnable warping function that adapts during training.

    Uses a neural network or spline to learn the optimal warping function.
    Can be initialized with a physical warping and refined during training.

    This is for future work - allows learning optimal warping from data.
    """

    def __init__(self, base_warping=None):
        self.base_warping = base_warping or SimpleWarping()
        # TODO: Add learnable component (e.g., spline or small NN)

    def warp(self, t):
        # Start with base warping
        t_warp = self.base_warping.warp(t)
        # TODO: Add learned correction
        return t_warp

    def unwarp(self, t_warp):
        # TODO: Invert learned warping
        return self.base_warping.unwarp(t_warp)


def get_warping(warping_type='simple', **kwargs):
    """
    Factory function to create warping objects.

    Parameters:
    -----------
    warping_type : str
        Type of warping: 'simple', 'chirp', 'piecewise', 'adaptive'
    **kwargs : dict
        Parameters for the specific warping type

    Returns:
    --------
    TimeWarping object

    Examples:
    ---------
    >>> warp = get_warping('simple', scale=2)
    >>> warp = get_warping('chirp', alpha=0.375)
    >>> warp = get_warping('piecewise', breakpoints=[-0.1, 0, 0.05], scales=[2, 1])
    """
    warping_types = {
        'simple': SimpleWarping,
        'chirp': ChirpTimeWarping,
        'piecewise': PiecewiseWarping,
        'adaptive': AdaptiveWarping,
    }

    if warping_type not in warping_types:
        raise ValueError(f"Unknown warping type: {warping_type}. "
                        f"Choose from {list(warping_types.keys())}")

    return warping_types[warping_type](**kwargs)
