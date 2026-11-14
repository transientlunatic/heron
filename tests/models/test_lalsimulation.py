"""Tests for heron.models.lalsimulation — requires lalsuite."""

import unittest
import numpy as np
import pytest

try:
    import lalsimulation
    from heron.models.lalsimulation import (
        LALSimulationApproximant,
        IMRPhenomD,
        IMRPhenomPv2,
        SEOBNRv3,
        SEOBNRv4,
    )
    HAS_LAL = True
except ImportError:
    HAS_LAL = False


@pytest.mark.skipif(not HAS_LAL, reason="lalsuite not installed")
class TestLALSimulationApproximant(unittest.TestCase):

    def setUp(self):
        self.approx = LALSimulationApproximant()

    def test_initialization(self):
        self.assertIn("m1", self.approx._args)
        self.assertIn("m2", self.approx._args)
        self.assertIn("distance", self.approx._args)

    def test_allowed_parameters(self):
        self.assertIsInstance(self.approx.allowed_parameters, list)
        self.assertIn("m1", self.approx.allowed_parameters)


@pytest.mark.skipif(not HAS_LAL, reason="lalsuite not installed")
class TestIMRPhenomPv2(unittest.TestCase):

    def setUp(self):
        self.approx = IMRPhenomPv2()

    def test_initialization(self):
        self.assertIsNotNone(self.approx._args["approximant"])

    def test_approximant_type(self):
        expected = lalsimulation.GetApproximantFromString("IMRPhenomPv2")
        self.assertEqual(self.approx._args["approximant"], expected)


@pytest.mark.skipif(not HAS_LAL, reason="lalsuite not installed")
class TestSEOBNRv3(unittest.TestCase):

    def setUp(self):
        self.approx = SEOBNRv3()

    def test_initialization(self):
        self.assertIsNotNone(self.approx._args["approximant"])


@pytest.mark.skipif(not HAS_LAL, reason="lalsuite not installed")
class TestSEOBNRv4(unittest.TestCase):

    def setUp(self):
        self.approx = SEOBNRv4()

    def test_initialization(self):
        self.assertIsNotNone(self.approx._args["approximant"])

    def test_approximant_type(self):
        expected = lalsimulation.GetApproximantFromString("SEOBNRv4")
        self.assertEqual(self.approx._args["approximant"], expected)


@pytest.mark.skipif(not HAS_LAL, reason="lalsuite not installed")
class TestIMRPhenomD(unittest.TestCase):

    def setUp(self):
        self.approx = IMRPhenomD()

    def test_initialization(self):
        self.assertIsNotNone(self.approx._args["approximant"])

    def test_approximant_type(self):
        expected = lalsimulation.GetApproximantFromString("IMRPhenomD")
        self.assertEqual(self.approx._args["approximant"], expected)


class TestInterpolatedPeakTime(unittest.TestCase):
    """LAL-free checks of the sub-sample peak locator."""

    def setUp(self):
        from heron.models.lalsimulation import interpolated_peak_time
        self.peak = interpolated_peak_time
        self.dt = 1.0 / 4096
        self.t = np.arange(-400, 100) * self.dt

    def _chirp(self, t0):
        amp = np.exp(-0.5 * ((self.t - t0) / 3e-3) ** 2)
        phase = 2 * np.pi * 150.0 * self.t
        return amp * np.cos(phase), amp * np.sin(phase)

    def test_recovers_subsample_peak(self):
        for t0 in (0.0, 37.3e-6, -121.0e-6, 0.4 * self.dt):
            hp, hx = self._chirp(t0)
            self.assertAlmostEqual(self.peak(self.t, hp, hx), t0, delta=1e-7)

    def test_falls_back_to_sample_at_array_edge(self):
        hp, hx = self._chirp(self.t[2])
        self.assertEqual(self.peak(self.t, hp, hx), self.t[2])


@pytest.mark.skipif(not HAS_LAL, reason="lalsuite not installed")
class TestPeakAlignment(unittest.TestCase):
    """LAL snaps t=0 to the sample nearest the peak; at M=60, f_min=20 Hz,
    dt=1/4096 s the IMRPhenomD peak crosses dt/2 between q=0.291 and 0.292,
    so the time axis jumps by one sample there. ``"interpolated"`` must
    remove the jump."""

    @staticmethod
    def _offset(approx, q):
        from astropy import units as u
        from heron.models.lalsimulation import interpolated_peak_time
        wf = approx.time_domain({
            "mass_ratio": q,
            "total_mass": 60 * u.solMass,
            "luminosity_distance": 100 * u.Mpc,
            "f_min": 20 * u.Hertz,
        })
        return interpolated_peak_time(
            wf["plus"].times, wf["plus"].data, wf["cross"].data)

    def test_default_is_lal_convention(self):
        self.assertEqual(IMRPhenomD().peak_alignment, "sample")

    def test_invalid_alignment_raises(self):
        with self.assertRaises(ValueError):
            IMRPhenomD(peak_alignment="nearest")

    def test_sample_alignment_jumps_across_q_0p2915(self):
        dt = 1.0 / 4096
        before = self._offset(IMRPhenomD(), 0.291)
        after = self._offset(IMRPhenomD(), 0.292)
        self.assertLess(before, -0.45 * dt)
        self.assertGreater(after, 0.45 * dt)

    def test_interpolated_alignment_puts_peak_at_zero(self):
        approx = IMRPhenomD(peak_alignment="interpolated")
        for q in (0.291, 0.292, 0.5):
            self.assertAlmostEqual(self._offset(approx, q), 0.0, delta=1e-8)

    def test_alignment_survives_pickling(self):
        import pickle
        approx = pickle.loads(
            pickle.dumps(IMRPhenomD(peak_alignment="interpolated")))
        self.assertEqual(approx.peak_alignment, "interpolated")
