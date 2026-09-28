"""Tests for heron.inference.detectors and detector geometry."""
import numpy as np
import pytest

from heron.detector import (
    antenna_patterns,
    detector_location,
    time_delay_from_geocentre,
    _C_SI,
)
from heron.inference.detectors import (
    Detector, KNOWN_DETECTORS, estimate_psd_welch, load_psd_ascii,
)


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

    def test_formula_matches_lal_with_shared_gmst(self):
        """The delay FORMULA matches LAL exactly when fed the same GMST — this
        isolates the geometry from heron's linear-GMST approximation."""
        lal = pytest.importorskip("lal")
        for prefix in ("H1", "L1", "V1"):
            det = lal.cached_detector_by_prefix[prefix]
            t = lal.LIGOTimeGPS(GPS)
            gha = lal.GreenwichMeanSiderealTime(t) - RA
            ehat = np.array([
                np.cos(DEC) * np.cos(gha),
                -np.cos(DEC) * np.sin(gha),
                np.sin(DEC),
            ])
            mine = float(-np.dot(ehat, np.asarray(det.location)) / _C_SI)
            expected = lal.TimeDelayFromEarthCenter(det.location, RA, DEC, t)
            assert mine == pytest.approx(expected, abs=1e-9)

    def test_within_gmst_approximation_of_lal(self):
        """heron's own delay uses its linear GMST (see
        test_detector.TestGMST.test_close_to_lal, < 0.01 rad), so it agrees with
        LAL only to ~|r|/c * 0.01 ≈ 2e-4 s — not to machine precision."""
        lal = pytest.importorskip("lal")
        for prefix in ("H1", "L1", "V1"):
            det = lal.cached_detector_by_prefix[prefix]
            expected = lal.TimeDelayFromEarthCenter(
                det.location, RA, DEC, lal.LIGOTimeGPS(GPS)
            )
            got = time_delay_from_geocentre(RA, DEC, GPS, prefix)
            assert got == pytest.approx(expected, abs=3e-4)


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


class TestLoadPsdAscii:

    @staticmethod
    def _write_psd_file(tmp_path, values):
        freqs = np.linspace(30.0, 500.0, len(values))
        path = tmp_path / "test_psd.dat"
        with open(path, "w") as f:
            f.write("# frequency  psd\n")
            for freq, val in zip(freqs, values):
                f.write(f"{freq} {val}\n")
        return path, freqs

    def test_reproduces_values_in_band(self, tmp_path):
        values = np.linspace(1e-46, 1e-44, 50)
        path, freqs = self._write_psd_file(tmp_path, values)
        psd_fn = load_psd_ascii(path, f_low=20.0)
        got = psd_fn(freqs)
        assert np.allclose(got, values)

    def test_below_f_low_and_outside_range_are_inf(self, tmp_path):
        values = np.linspace(1e-46, 1e-44, 50)
        path, _ = self._write_psd_file(tmp_path, values)
        psd_fn = load_psd_ascii(path, f_low=20.0)
        vals = psd_fn(np.array([5.0, 15.0, 1000.0]))
        assert np.all(np.isinf(vals))

    def test_asd_kind_squares_values(self, tmp_path):
        asd_values = np.linspace(1e-23, 1e-22, 50)
        path, freqs = self._write_psd_file(tmp_path, asd_values)
        psd_fn = load_psd_ascii(path, f_low=20.0, kind="asd")
        got = psd_fn(freqs)
        assert np.allclose(got, asd_values**2)

    def test_bad_kind_raises(self, tmp_path):
        path, _ = self._write_psd_file(tmp_path, np.linspace(1e-46, 1e-44, 10))
        with pytest.raises(ValueError):
            load_psd_ascii(path, kind="bogus")

    def test_drops_into_detector(self, tmp_path):
        values = np.linspace(1e-46, 1e-44, 50)
        path, _ = self._write_psd_file(tmp_path, values)
        det = Detector.from_name("H1", psd_fn=load_psd_ascii(path))
        assert callable(det.psd_fn)
        assert np.isfinite(det.psd_fn(np.array([100.0]))[0])
