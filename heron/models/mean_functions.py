"""
Custom mean functions for GPyTorch models using waveform approximants.

This module provides mean function implementations that wrap gravitational wave
approximants to use as mean functions in Gaussian Process models.
"""

import torch
import gpytorch
import numpy as np
from typing import Dict, Optional, Tuple

try:
    import jax
    import jax.numpy as jnp
    from jax import config
    # Enable float64 for better precision
    config.update("jax_enable_x64", True)
    from ripple.waveforms import IMRPhenomD
    from ripple import ms_to_Mc_eta
    RIPPLE_AVAILABLE = True
except ImportError:
    RIPPLE_AVAILABLE = False
    print("Warning: ripple not available. IMRPhenomD mean function will not work.")


class IMRPhenomDMeanFunction(gpytorch.means.Mean):
    """
    Mean function that evaluates IMRPhenomD using ripple.

    This mean function generates time-domain waveforms using the IMRPhenomD
    approximant via the ripple library (JAX-based, GPU-accelerated).

    Parameters
    ----------
    total_mass : float
        Total mass of the binary system in solar masses
    distance : float
        Luminosity distance in Mpc
    delta_t : float
        Time step for waveform generation in seconds
    f_lower : float
        Lower frequency cutoff in Hz
    f_ref : float
        Reference frequency in Hz (default: 20 Hz)
    device : torch.device
        Device to run computations on
    y_mean : float, optional
        Mean of the training data for normalization. If provided along with
        y_std, the mean function output will be normalized to match the
        training data normalization.
    y_std : float, optional
        Standard deviation of the training data for normalization.
    polarization : str
        Polarization to evaluate ('plus' or 'cross'). Default is 'plus'.
    warp_scale : float, optional
        Time warping factor used on training data. If provided, the mean
        function will unwarp the time coordinates before generating waveforms.
        Default is None (no unwarping).
    """

    def __init__(
        self,
        total_mass: float,
        distance: float,
        delta_t: float = 1.0 / 4096,
        f_lower: float = 20.0,
        f_ref: float = 20.0,
        device: Optional[torch.device] = None,
        y_mean: Optional[float] = None,
        y_std: Optional[float] = None,
        polarization: str = 'plus',
        warp_scale: Optional[float] = None
    ):
        super().__init__()

        if not RIPPLE_AVAILABLE:
            raise ImportError(
                "ripple is required for IMRPhenomDMeanFunction. "
                "Install it with: pip install ripplegw"
            )

        self.total_mass = total_mass
        self.distance = distance
        self.delta_t = delta_t
        self.f_lower = f_lower
        self.f_ref = f_ref
        self.device = device or torch.device("cpu")

        # Normalization parameters for matching training data normalization
        self.y_mean = y_mean
        self.y_std = y_std

        # Polarization for waveform generation
        self.polarization = polarization

        # Time warping factor (to unwarp input coordinates)
        self.warp_scale = warp_scale

        # Cache for waveforms to avoid recomputation
        self._waveform_cache = {}

    def _generate_waveform_ripple(
        self,
        mass_ratio: float,
        chi1: float = 0.0,
        chi2: float = 0.0,
        duration: float = 4.0,
        polarization: str = 'plus'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a waveform using ripple's IMRPhenomD.

        Parameters
        ----------
        mass_ratio : float
            Mass ratio q = m2/m1 where m1 >= m2
        chi1 : float
            Dimensionless spin of primary (default: 0)
        chi2 : float
            Dimensionless spin of secondary (default: 0)
        duration : float
            Duration of waveform in seconds
        polarization : str
            Polarization to return ('plus' or 'cross')

        Returns
        -------
        times : np.ndarray
            Time array
        waveform : np.ndarray
            Waveform amplitude at each time
        """
        # Convert mass ratio to component masses
        m1 = self.total_mass / (1.0 + mass_ratio)
        m2 = mass_ratio * m1

        # Convert to chirp mass and symmetric mass ratio
        Mc_eta = ms_to_Mc_eta(jnp.array([m1, m2]))
        Mc = float(Mc_eta[0])
        eta = float(Mc_eta[1])

        # Set up frequency array
        # Use Nyquist frequency based on delta_t
        f_max = 1.0 / (2.0 * self.delta_t)
        n_samples = int(duration / self.delta_t)
        delta_f = 1.0 / duration

        # Create frequency array
        freqs = jnp.arange(self.f_lower, f_max, delta_f)

        # Parameters for IMRPhenomD: [Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination]
        # For mean function, we use face-on (inclination=0) and standard phases
        tc = 0.0  # Coalescence time
        phic = 0.0  # Coalescence phase
        inclination = 0.0  # Face-on for simplicity

        theta = jnp.array([Mc, eta, chi1, chi2, self.distance, tc, phic, inclination])

        # Generate frequency-domain waveform
        hp_fd, hc_fd = IMRPhenomD.gen_IMRPhenomD_hphc(freqs, theta, self.f_ref)

        # Convert to time domain using IFFT
        # Pad to make proper length using JAX's immutable array syntax
        n_freqs = len(freqs)

        # Create full frequency array including negative frequencies
        # Use jnp.at[].set() for JAX-compatible array assignment
        hp_full = jnp.zeros(n_samples, dtype=jnp.complex128)
        hp_full = hp_full.at[:n_freqs].set(hp_fd)

        hc_full = jnp.zeros(n_samples, dtype=jnp.complex128)
        hc_full = hc_full.at[:n_freqs].set(hc_fd)

        # IFFT to get time domain
        hp_td = jnp.fft.ifft(hp_full) * n_samples
        hc_td = jnp.fft.ifft(hc_full) * n_samples

        # Create time array (centered on coalescence)
        times = jnp.arange(n_samples) * self.delta_t - duration / 2.0

        # Convert to numpy
        times_np = np.array(times)

        if polarization == 'plus':
            waveform_np = np.real(np.array(hp_td))
        else:
            waveform_np = np.real(np.array(hc_td))

        return times_np, waveform_np

    def _interpolate_waveform(
        self,
        target_times: torch.Tensor,
        waveform_times: np.ndarray,
        waveform_data: np.ndarray
    ) -> torch.Tensor:
        """
        Interpolate waveform to target times.

        Parameters
        ----------
        target_times : torch.Tensor
            Times where we need waveform values
        waveform_times : np.ndarray
            Original time array from waveform generation
        waveform_data : np.ndarray
            Waveform amplitude values

        Returns
        -------
        torch.Tensor
            Interpolated waveform values
        """
        # Convert to numpy for interpolation
        target_times_np = target_times.cpu().numpy()

        # Use linear interpolation
        interpolated = np.interp(
            target_times_np,
            waveform_times,
            waveform_data,
            left=0.0,  # Zero outside range
            right=0.0
        )

        # Convert back to torch
        return torch.tensor(interpolated, dtype=torch.float32, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the mean function at input points.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [n_points, 2] where:
            - x[:, 0] is mass ratio
            - x[:, 1] is time (in geometric units, M=1)

        Returns
        -------
        torch.Tensor
            Mean function values at input points
        """
        # Extract unique mass ratios
        mass_ratios = x[:, 0].unique()
        times = x[:, 1]

        # Initialize output
        output = torch.zeros(len(x), device=self.device)

        # For each unique mass ratio, generate waveform and interpolate
        for q in mass_ratios:
            # Find all points with this mass ratio
            mask = x[:, 0] == q
            target_times = times[mask]

            # Unwarp times if warp_scale is set
            # During training, negative times are divided by warp_scale
            # We need to reverse this: multiply negative times by warp_scale
            if self.warp_scale is not None:
                target_times_unwarped = target_times.clone()
                target_times_unwarped[target_times < 0] *= self.warp_scale
            else:
                target_times_unwarped = target_times

            # Convert from geometric time (M=1) to physical time (seconds)
            # Factor to convert: M_sun in seconds = 4.925e-6 s
            M_sec = self.total_mass * 4.925e-6
            target_times_physical = target_times_unwarped * M_sec

            # Generate cache key using stored polarization
            cache_key = (float(q), self.polarization)

            if cache_key not in self._waveform_cache:
                # Generate waveform
                # Estimate duration based on time range with padding (use unwarped times)
                t_min = float(target_times_unwarped.min() * M_sec)
                t_max = float(target_times_unwarped.max() * M_sec)
                duration = max(4.0, 2.0 * (t_max - t_min))  # At least 4 seconds

                try:
                    wf_times, wf_data = self._generate_waveform_ripple(
                        mass_ratio=float(q),
                        duration=duration,
                        polarization=self.polarization
                    )
                    self._waveform_cache[cache_key] = (wf_times, wf_data)
                except Exception as e:
                    print(f"Warning: Failed to generate waveform for q={q}: {e}")
                    # Return zeros if generation fails
                    output[mask] = 0.0
                    continue

            wf_times, wf_data = self._waveform_cache[cache_key]

            # Interpolate to target times
            interpolated = self._interpolate_waveform(
                target_times_physical,
                wf_times,
                wf_data
            )

            output[mask] = interpolated

        # Apply normalization if parameters are set
        # This transforms the physical waveform values to normalized space
        if self.y_mean is not None and self.y_std is not None:
            output = (output - self.y_mean) / self.y_std

        return output

    def set_normalization(self, y_mean: float, y_std: float):
        """
        Set normalization parameters to match training data normalization.

        This must be called before training to ensure the mean function
        output is in the same normalized space as the training targets.

        Parameters
        ----------
        y_mean : float
            Mean of the training data
        y_std : float
            Standard deviation of the training data
        """
        self.y_mean = y_mean
        self.y_std = y_std
        # Clear cache since normalization changed
        self._waveform_cache.clear()

    def clear_cache(self):
        """Clear the waveform cache."""
        self._waveform_cache.clear()


class ZeroMeanWithApproximant(gpytorch.means.Mean):
    """
    Wrapper that can switch between zero mean and approximant mean.

    This is useful for training where you might want to train with
    zero mean initially and then switch to approximant mean.
    """

    def __init__(self, approximant_mean: Optional[IMRPhenomDMeanFunction] = None):
        super().__init__()
        self.approximant_mean = approximant_mean
        self.use_approximant = approximant_mean is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_approximant and self.approximant_mean is not None:
            return self.approximant_mean(x)
        else:
            # Zero mean
            return torch.zeros(x.shape[0], device=x.device)
