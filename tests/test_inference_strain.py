"""Tests for heron.inference.strain.

fetch_gwosc_strain() itself is thin glue over a network call to GWOSC (via
gwpy) and, like scripts/fetch_gw150914_psd.py, is validated by actually
running it (see scripts/fetch_gw150914_strain.py), not by a mocked unit
test. taper_strain() is pure/deterministic and gets full coverage here.
"""
import numpy as np
import pytest

from heron.inference.strain import taper_strain


class TestTaperStrain:

    def test_preserves_length(self):
        strain = np.ones(4096)
        out = taper_strain(strain, dt=1.0 / 4096.0, roll_off=0.4)
        assert out.shape == strain.shape

    def test_edges_taper_towards_zero(self):
        strain = np.ones(4096)
        out = taper_strain(strain, dt=1.0 / 4096.0, roll_off=0.4)
        assert out[0] == pytest.approx(0.0, abs=1e-9)
        assert out[-1] == pytest.approx(0.0, abs=1e-9)

    def test_interior_untouched(self):
        strain = np.ones(4096)
        out = taper_strain(strain, dt=1.0 / 4096.0, roll_off=0.4)
        # Well inside the flat-top region (duration=1s, roll_off=0.4s each
        # end leaves a flat top in the middle).
        mid = len(strain) // 2
        assert out[mid] == pytest.approx(1.0, abs=1e-9)

    def test_zero_roll_off_is_a_no_op(self):
        rng = np.random.default_rng(0)
        strain = rng.standard_normal(1000)
        out = taper_strain(strain, dt=1.0 / 1000.0, roll_off=0.0)
        assert np.allclose(out, strain)

    def test_roll_off_spanning_full_duration_tapers_everything(self):
        # duration = 1s, roll_off = 1s -> alpha clipped to 1 (full Hann-like).
        strain = np.ones(1000)
        out = taper_strain(strain, dt=1.0 / 1000.0, roll_off=10.0)
        assert out[0] == pytest.approx(0.0, abs=1e-9)
        assert out[len(strain) // 2] > 0.9  # still near-flat at the centre

    def test_scales_linearly(self):
        rng = np.random.default_rng(1)
        strain = rng.standard_normal(2048)
        out1 = taper_strain(strain, dt=1.0 / 2048.0, roll_off=0.2)
        out2 = taper_strain(2.0 * strain, dt=1.0 / 2048.0, roll_off=0.2)
        assert np.allclose(2.0 * out1, out2)


class TestFetchGwoscStrainValidation:

    def test_rejects_post_trigger_duration_ge_duration(self):
        from heron.inference.strain import fetch_gwosc_strain

        with pytest.raises(ValueError):
            fetch_gwosc_strain(
                "H1", trigger_time=1126259462.4,
                duration=4.0, post_trigger_duration=4.0,
            )
