"""
Testing models
--------------

Simple waveform models for unit tests. Not for production use.
"""

import numpy as np
import scipy.spatial.distance
from astropy import units as u
from ..types import Waveform, WaveformDict
from . import WaveformApproximant


class SineGaussianWaveform(WaveformApproximant):
    """A simple SineGaussian waveform for testing purposes."""

    def __init__(self):
        super().__init__()
        self._args = {
            "width": 0.02 * u.second,
            "frequency": 500 * u.Hertz,
            "segment length": 1 * u.second,
            "amplitude": 1.0,
        }

    def time_domain(self, parameters, times=None, sample_rate=1024 * u.Hertz):
        epoch = parameters.get("gpstime", parameters.get("epoch", 0))
        amplitude = parameters.get("amplitude", self._args["amplitude"])
        self._args.update(parameters)
        width = self._args['width']
        length = self._args['segment length']

        if times is not None:
            # External time array provided — use it directly (unitless seconds)
            times = np.asarray(times, dtype=float)
            has_units = False
        elif "time" in parameters:
            # Dict-style time grid (from evaluator / predict interface)
            t = parameters["time"]
            times = np.linspace(t["lower"], t["upper"], t["number"])
            has_units = False
        else:
            times = np.linspace(-length / 2, length / 2, int((length * sample_rate).value))
            has_units = True

        if has_units:
            envelope = amplitude * np.exp(
                (-(times - epoch) ** 2 / (2 * width ** 2)).value
            ) / np.sqrt(2 * np.pi * width ** 2)
            strain = np.sin((times * u.second * self._args['frequency']).value) * envelope
            times_out = times
        else:
            # Unitless path — strip units from width and frequency
            w = float(width / u.second) if hasattr(width, 'unit') else float(width)
            f = float(self._args['frequency'] / u.Hertz) if hasattr(self._args['frequency'], 'unit') else float(self._args['frequency'])
            envelope = amplitude * np.exp(
                -(times - epoch) ** 2 / (2 * w ** 2)
            ) / np.sqrt(2 * np.pi * w ** 2)
            strain = np.sin(times * f * 2 * np.pi) * envelope
            times_out = times

        covariance = np.exp(
            -0.5 * scipy.spatial.distance.cdist(
                np.expand_dims(times_out, 1),
                np.expand_dims(times_out, 1),
                'sqeuclidean',
            )
        ) * np.exp(100 * np.abs(float(width / u.second if hasattr(width, 'unit') else width) - 0.05))
        hp_data = Waveform(
            data=strain,
            times=np.asarray(times_out, dtype=float),
            covariance=covariance,
            t0=epoch,
        )
        hx_data = Waveform(
            data=strain,
            times=np.asarray(times_out, dtype=float),
            covariance=covariance,
            t0=epoch,
        )
        return WaveformDict(parameters=self._args, plus=hp_data, cross=hx_data)
