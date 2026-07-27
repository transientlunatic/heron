"""Tests for heron.inference.detectors and detector geometry."""
import numpy as np
import pytest

from heron.detector import (
    antenna_patterns,
    detector_location,
    time_delay_from_geocentre,
    _C_SI,
)
from heron.inference.detectors import Detector, KNOWN_DETECTORS, estimate_psd_welch


RA, DEC, PSI, GPS = 1.95, -1.27, 0.82, 1187008882.4


class TestTimeDelay:

    @pytest.mark.parametrize("prefix", ["H1", "L1", "V1"])
    def test_within_light_travel_bound(self, prefix):
        d = time_delay_from_geocentre(RA, DEC, GPS, prefix)
        r = np.linalg.norm(detector_location(prefix))
        assert abs(d) <= r / _C_SI + 1e-9

    def test_source_along_position_vector_arrives_early(self):
        """A source in the exact direction of the detector position vector
        arrives earliest (most negative delay)."""
        prefix = "H1"
        # Sky direction equal to +position vector: delay should be its minimum.
        r = detector_location(prefix)
        rhat = r / np.linalg.norm(r)
        # Convert rhat (ECEF) to (ra, dec) at GPS via the same GHA convention.
        from heron.detector import gmst_at_gps
        dec = np.arcsin(rhat[2])
        gha = np.arctan2(-rhat[1], rhat[0])
        ra = gmst_at_gps(GPS) - gha
        d = time_delay_from_geocentre(ra, dec, GPS, prefix)
        assert d == pytest.approx(-np.linalg.norm(r) / _C_SI, rel=1e-6)

    def test_matches_lal_when_available(self):
        lal = pytest.importorskip("lal")
        for prefix in ("H1", "L1", "V1"):
            det = lal.cached_detector_by_prefix[prefix]
            gmst = lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(GPS))
            expected = lal.TimeDelayFromEarthCenter(det.location, RA, DEC, lal.LIGOTimeGPS(GPS))
            got = time_delay_from_geocentre(RA, DEC, GPS, prefix)
            assert got == pytest.approx(expected, abs=1e-6)


class TestDetector:

    def test_from_name_default_psd(self):
        from heron.evaluation.psd import aligo_design_psd
        d = Detector.from_name("H1")
        assert d.prefix == "H1"
        assert d.psd_fn is aligo_design_psd

    def test_custom_psd(self, flat_psd):
        d = Detector.from_name("L1", psd_fn=flat_psd)
        assert d.psd_fn is flat_psd

    def test_antenna_patterns_delegate(self):
        d = Detector.from_name("H1")
        assert d.antenna_patterns(RA, DEC, PSI, GPS) == antenna_patterns(RA, DEC, PSI, GPS, "H1")

    def test_time_delay_delegate(self):
        d = Detector.from_name("V1")
        assert d.time_delay_from_geocentre(RA, DEC, GPS) == time_delay_from_geocentre(
            RA, DEC, GPS, "V1"
        )

    def test_registry(self):
        assert set(KNOWN_DETECTORS) == {"H1", "L1", "V1"}
        assert all(isinstance(v, Detector) for v in KNOWN_DETECTORS.values())


class TestWelchPSD:

    def test_returns_callable_with_inf_below_flow(self):
        rng = np.random.default_rng(0)
        fs = 1024.0
        strain = rng.standard_normal(int(8 * fs))
        psd_fn = estimate_psd_welch(strain, dt=1.0 / fs, segment_duration=2.0, f_low=20.0)
        freqs = np.array([5.0, 50.0, 200.0])
        vals = psd_fn(freqs)
        assert np.isinf(vals[0])            # below f_low
        assert np.all(np.isfinite(vals[1:]))  # in band
        assert np.all(vals[1:] > 0)
