"""Tests for heron.types — Waveform and WaveformDict."""

import numpy as np
import pytest

from heron.types import Waveform, WaveformDict


class TestWaveform:

    def test_basic_creation(self):
        times = np.linspace(0, 1, 100)
        data = np.sin(2 * np.pi * 10 * times)
        wf = Waveform(data=data, times=times)

        assert len(wf) == 100
        assert wf.dt == pytest.approx(times[1] - times[0])
        assert wf.variance is None
        assert wf.covariance is None

    def test_with_covariance(self):
        n = 50
        times = np.linspace(0, 1, n)
        data = np.zeros(n)
        cov = np.eye(n) * 0.01

        wf = Waveform(data=data, times=times, covariance=cov)

        assert wf.covariance.shape == (n, n)
        np.testing.assert_allclose(wf.variance, np.full(n, 0.01))
        np.testing.assert_allclose(wf.std, np.full(n, 0.1))

    def test_duration(self):
        wf = Waveform(data=np.zeros(10), times=np.linspace(0, 2, 10))
        assert wf.duration == pytest.approx(2.0)

    def test_arrays_cast_to_float64(self):
        wf = Waveform(data=[1, 2, 3], times=[0.0, 0.1, 0.2])
        assert wf.data.dtype == np.float64
        assert wf.times.dtype == np.float64


class TestWaveformDict:

    def _make_dict(self):
        n = 100
        times = np.linspace(-0.5, 0.02, n)
        plus = Waveform(data=np.sin(times), times=times)
        cross = Waveform(data=np.cos(times), times=times)
        return WaveformDict(parameters={"mass_ratio": 0.8}, plus=plus, cross=cross)

    def test_getitem(self):
        wd = self._make_dict()
        assert isinstance(wd["plus"], Waveform)
        assert isinstance(wd["cross"], Waveform)

    def test_setitem(self):
        wd = self._make_dict()
        new_wf = Waveform(data=np.zeros(10), times=np.arange(10))
        wd["extra"] = new_wf
        assert "extra" in wd

    def test_contains(self):
        wd = self._make_dict()
        assert "plus" in wd
        assert "missing" not in wd

    def test_iter(self):
        wd = self._make_dict()
        keys = list(wd)
        assert "plus" in keys
        assert "cross" in keys

    def test_times(self):
        wd = self._make_dict()
        np.testing.assert_array_equal(wd.times, wd["plus"].times)

    def test_parameters(self):
        wd = self._make_dict()
        assert wd.parameters["mass_ratio"] == 0.8

    def test_hrss(self):
        wd = self._make_dict()
        hrss = wd.hrss
        expected = np.sqrt(wd["plus"].data**2 + wd["cross"].data**2)
        np.testing.assert_allclose(hrss, expected)

    def test_repr(self):
        wd = self._make_dict()
        assert "plus" in repr(wd)
        assert "cross" in repr(wd)
