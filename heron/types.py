from itertools import cycle

# GWPy to help with timeseries
from gwpy.timeseries import TimeSeriesBase, TimeSeries
from gwpy.frequencyseries import FrequencySeries

from lal import cached_detector_by_prefix, TimeDelayFromEarthCenter, LIGOTimeGPS
from lalinference import DetFrameToEquatorial

import numpy as array_library
import numpy as np
import matplotlib.pyplot as plt


class TimeSeries(TimeSeries):
    """
    Overload the GWPy timeseries so that additional methods can be defined upon it.
    """


    def plot(self, *args, **kwargs):
        f = super().plot(*args, **kwargs)
        ax = f.get_axes()[0]
        if hasattr(self, "_variance") and self._variance is not None:
            ax.fill_between(self.times.value,
                            self.data + self._variance,
                            self.data - self._variance,
                            alpha=0.5)
        return f

    @property
    def variance(self):
        if isinstance(self._variance, type(None)):
            return self._variance
        elif isinstance(self.covariance, type(None)):
            return np.diag(self.covariance)

    def determine_overlap(self, timeseries_a, timeseries_b):
        """
        Determine the overlap between two timeseries efficiently using binary search.

        Optimized version: O(log N) instead of O(N) per check.
        Uses searchsorted for fast index finding instead of argmin.
        """
        # Extract time arrays - handle both regular arrays and astropy quantities
        times_a = timeseries_a.times.value if hasattr(timeseries_a.times, 'value') else timeseries_a.times
        times_b = timeseries_b.times.value if hasattr(timeseries_b.times, 'value') else timeseries_b.times

        # Get bounds
        a_start, a_end = times_a[0], times_a[-1]
        b_start, b_end = times_b[0], times_b[-1]

        # Get sample spacing (assumed uniform)
        dt_a = times_a[1] - times_a[0]
        dt_b = times_b[1] - times_b[0]
        tolerance = max(dt_a, dt_b) * 0.5  # Half a sample for numerical tolerance

        # Quick check: do the time ranges overlap at all?
        if a_end + tolerance < b_start or b_end + tolerance < a_start:
            return None  # No overlap

        # Determine overlap region
        overlap_start = max(a_start, b_start)
        overlap_end = min(a_end, b_end)

        if overlap_start > overlap_end + tolerance:
            return None  # No overlap

        # Use binary search (searchsorted) to find indices efficiently
        # This is O(log N) instead of O(N) from argmin(abs(...))
        start_a = array_library.searchsorted(times_a, overlap_start, side='left')
        finish_a = array_library.searchsorted(times_a, overlap_end, side='right') - 1

        start_b = array_library.searchsorted(times_b, overlap_start, side='left')
        finish_b = array_library.searchsorted(times_b, overlap_end, side='right') - 1

        # Clamp to valid indices
        start_a = max(0, min(start_a, len(times_a) - 1))
        finish_a = max(0, min(finish_a, len(times_a) - 1))
        start_b = max(0, min(start_b, len(times_b) - 1))
        finish_b = max(0, min(finish_b, len(times_b) - 1))

        # Verify we have a valid overlap
        if start_a > finish_a or start_b > finish_b:
            return None

        return (start_a, finish_a), (start_b, finish_b)

    def align(self, waveform_b):
        """
        Align this waveform with another one by altering the phase.
        """

        indices = self.determine_overlap(self, waveform_b)

        return self[indices[0][0]:indices[0][1]], waveform_b[indices[1][0]: indices[1][1]]


class PSD(FrequencySeries):
    def __init__(self, data, frequencies, *args, **kwargs):
        super(PSD).__init__(*args, **kwargs)


class WaveformBase(TimeSeries):
    def __init__(self, *args, **kwargs):
        super(WaveformBase).__init__()

class Waveform(WaveformBase):
    def __init__(self, variance=None, covariance=None, covariance_gpu=None,
                 output_scale=1.0, distance_factor=1.0, *args, **kwargs):
        """
        Initialize Waveform with optional lazy GPU covariance loading.

        Parameters
        ----------
        variance : array-like, optional
            Variance of the waveform (diagonal covariance)
        covariance : array-like, optional
            Full covariance matrix (CPU)
        covariance_gpu : torch.Tensor, optional
            Covariance matrix on GPU (transferred lazily on access)
        output_scale : float, optional
            Output scaling factor for lazy GPU transfer
        distance_factor : float, optional
            Distance scaling factor for lazy GPU transfer
        """
        self._covariance = covariance
        self._covariance_gpu = covariance_gpu
        self._output_scale = output_scale
        self._distance_factor = distance_factor
        self._variance = variance
        super(Waveform, self).__init__(*args, **kwargs)

    def __new__(self, variance=None, covariance=None, covariance_gpu=None,
                output_scale=1.0, distance_factor=1.0, *args, **kwargs):
        waveform = super(Waveform, self).__new__(TimeSeries, *args, **kwargs)
        waveform._covariance = covariance
        waveform._covariance_gpu = covariance_gpu
        waveform._output_scale = output_scale
        waveform._distance_factor = distance_factor
        waveform._variance = variance

        return waveform

    @property
    def covariance(self):
        """
        Get covariance matrix with lazy GPU transfer.

        If covariance was stored on GPU, transfers to CPU on first access.
        This avoids unnecessary GPU->CPU transfers when covariance is not needed.
        """
        if self._covariance is None and self._covariance_gpu is not None:
            # Lazy transfer from GPU to CPU
            try:
                # Handle torch tensors (lazy GPU->CPU transfer)
                cov_cpu = self._covariance_gpu.cpu()
                # Apply scaling factors
                self._covariance = (
                    cov_cpu.numpy()
                    / self._output_scale
                    / self._output_scale
                    / self._distance_factor**2
                )
            except Exception as e:
                # If transfer fails, return None rather than crashing
                import warnings
                warnings.warn(f"Failed to transfer covariance from GPU: {e}")
                return None
        return self._covariance

    @covariance.setter
    def covariance(self, value):
        """Set covariance matrix directly."""
        self._covariance = value


class WaveformDict:
    def __init__(self, parameters=None, **kwargs):
        self.waveforms = kwargs
        self._parameters = parameters

    def __getitem__(self, item):
        return self.waveforms[item]

    def __repr__(self):
        return f"<WaveformDict with components: {list(self.waveforms.keys())}>"

    def __array__(self):
        # Default to 'plus' component for array conversion
        list_of_arrays = [waveform for waveform in self.waveforms.values()]
        return array_library.vstack(list_of_arrays).T

    @property
    def times(self):
        # Default to 'plus' component for times
        if "plus" in self.waveforms:
            return self.waveforms["plus"].times
        else:
            first_key = list(self.waveforms.keys())[0]
            return self.waveforms[first_key].times

    @property
    def parameters(self):
        return self._parameters

    @property
    def hrss(self):
        if "plus" in self.waveforms and "cross" in self.waveforms:
            return array_library.sqrt(
                self.waveforms["plus"] ** 2 + self.waveforms["cross"] ** 2
            )
        else:
            raise NotImplementedError

    def project(
        self,
        detector,
        ra=None,
        dec=None,
        psi=None,
        time=None,
        iota=None,
        phi_0=None,
        **kwargs
    ):
        """
        Project this waveform onto a detector.

        Parameters
        ----------
        detector : `heron.detectors.Detector`
          The detector onto which the waveform should be projected.
        ra : float, optional
          The right ascension of the signal source
        dec : float, optional
          The declination of the signal source.
        psi : float, optional
          The polarisation angle of the signal.
        time : `astropy.time.Time`, optional
          The time at which to project the waveform. If not specified, the epoch of the waveform is used.
        iota : float, optional
          The inclination angle of the source.
        phi_0 : float, optional
          The initial phase of the source.
        """

        if not time:
            time = self.waveforms["plus"].epoch.value

        if ((ra is None) and (dec is None)) and (
            ("ra" in self._parameters) and ("dec" in self._parameters)
        ):
            ra = self._parameters["ra"]
            dec = self._parameters["dec"]

            dt = detector.geocentre_delay(ra=ra, dec=dec, times=time)

        elif (
            ("azimuth" in self._parameters.keys())
            and ("zenith" in self._parameters.keys())
            and ("reference_frame" in self._parameters.keys())
        ):
            # Use horizontal coordinates.
            det1 = cached_detector_by_prefix[self._parameters["reference_frame"][0]]
            det2 = cached_detector_by_prefix[self._parameters["reference_frame"][1]]
            ra, dec, dt = DetFrameToEquatorial(
                det1,
                det2,
                time,
                self._parameters["azimuth"],
                self._parameters["zenith"],
            )

        elif (ra is None) and (dec is None):
            raise ValueError("Right ascension and declination must both be specified.")

        else:
            dt = detector.geocentre_delay(ra=ra, dec=dec, times=time)

        if "plus" in self.waveforms and "cross" in self.waveforms:

            if not iota and "theta_jn" in self._parameters:
                iota = self._parameters["theta_jn"]
            elif isinstance(iota, type(None)):
                raise ValueError("Theta_jn must be specified!")

            if not phi_0 and "phase" in self._parameters:
                phi_0 = self._parameters["phase"]
            elif isinstance(phi_0, type(None)):
                raise ValueError("Initial phase must be specified!")

            if not psi and "psi" in self._parameters:
                psi = self._parameters["psi"]
            elif isinstance(psi, type(None)):
                raise ValueError("Polarisation must be specified!")

            response = detector.antenna_response(ra, dec, psi, time=time)

            plus_prefactor = (
                array_library.cos(phi_0)
                * (1 + array_library.cos(iota) ** 2)
                * response.plus
                + array_library.sin(phi_0) * array_library.cos(iota) * response.cross
            )
            cross_prefactor = (
                array_library.cos(phi_0) * array_library.cos(iota) * response.cross
                - array_library.sin(phi_0)
                * (1 + array_library.cos(iota) ** 2)
                * response.plus
            )

            projected_data = (
                self.waveforms["plus"].data * plus_prefactor
                + self.waveforms["cross"].data * cross_prefactor
            )

            if self.waveforms["plus"]._variance is not None:
                projected_variance = (
                    self.waveforms["plus"]._variance * plus_prefactor**2
                    + self.waveforms["cross"]._variance * cross_prefactor**2
                )
            else:
                projected_variance = None

            if self.waveforms["plus"].covariance is not None:
                projected_covariance = (
                    self.waveforms["plus"].covariance * plus_prefactor**2
                    + self.waveforms["cross"].covariance * cross_prefactor**2
                )
            else:
                projected_covariance = None

            projected_waveform = Waveform(
                data=projected_data,
                variance=projected_variance,
                covariance=projected_covariance,
                times=self.waveforms["plus"].times,
            )

            projected_waveform.shift(dt)

            return projected_waveform

        else:
            raise NotImplementedError


class WaveformManifold:
    """
    Store a manifold of different waveform points.
    """

    def __init__(self):
        self.locations = []
        self.data = []

    def add_waveform(self, waveforms: WaveformDict):
        self.locations.append(waveforms.parameters)
        self.data.append(waveforms)

    def array(self, component="plus", parameter="m1"):
        all_data = []
        for wn in range(len(self.locations)):
            data = array_library.array(
                list(
                    zip(
                        cycle([self.locations[wn][parameter]]),
                        self.data[wn][component].times.value,
                        self.data[wn][component].value,
                    )
                )
            )
            all_data.append(data)
        return array_library.vstack(all_data)

    def plot(self, component="plus", parameter="m1"):
        f, ax = plt.subplots(1, 1)
        for wn in range(len(self.locations)):
            data = array_library.array(
                list(
                    zip(
                        cycle([self.locations[wn][parameter]]),
                        self.data[wn][component].times.value,
                    )
                )
            )
            plt.scatter(data[:, 1], data[:, 0], c=self.data[wn][component], marker=".")
        return f
