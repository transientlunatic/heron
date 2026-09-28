"""Tests for heron.inference.injection.Injection."""
import numpy as np
import pytest

from heron.noise import noise_covariance
from heron.inference.detectors import Detector
from heron.inference.injection import Injection
from heron.inference.network import NetworkLikelihood


TC = 1187008882.4
RA, DEC, PSI, Q = 1.95, -1.27, 0.82, 0.8


def _times(n=256, fs=512.0):
    return TC + (np.arange(n) - n // 2) / fs


def _network(flat_psd):
    return [Detector.from_name("H1", psd_fn=flat_psd),
            Detector.from_name("L1", psd_fn=flat_psd)]


def _params():
    return {"mass_ratio": Q, "tc": TC, "ra": RA, "dec": DEC, "psi": PSI}


class TestGenerate:

    def test_shapes_and_keys(self, stub_surrogate, flat_psd):
        times = _times()
        inj = Injection(times=times, detectors=_network(flat_psd), parameters=_params())
        res = inj.generate(stub_surrogate, rng=np.random.default_rng(0))
        assert set(res.data) == {"H1", "L1"}
        for prefix in ("H1", "L1"):
            assert res.data[prefix].shape == (len(times),)
            assert res.signals[prefix].shape == (len(times),)
        assert res.network_snr > 0

    def test_network_snr_is_quadrature_sum(self, stub_surrogate, flat_psd):
        times = _times()
        inj = Injection(times=times, detectors=_network(flat_psd), parameters=_params())
        res = inj.generate(stub_surrogate, rng=np.random.default_rng(0))
        expected = np.sqrt(sum(v**2 for v in res.snrs.values()))
        assert res.network_snr == pytest.approx(expected)

    def test_snr_matches_manual_optimal(self, stub_surrogate, flat_psd):
        times = _times()
        inj = Injection(times=times, detectors=_network(flat_psd), parameters=_params())
        res = inj.generate(stub_surrogate, rng=np.random.default_rng(0))
        C = noise_covariance(times, flat_psd, f_low=20.0, jitter_rel=1e-8)
        # Manual optimal SNR on the HP-filtered signal.
        sig = res.signals["H1"]
        sig_hp = inj._hp_filter(sig)
        manual = float(np.sqrt(sig_hp @ np.linalg.solve(C, sig_hp)))
        assert res.snrs["H1"] == pytest.approx(manual, rel=1e-9)

    def test_reproducible_with_seed(self, stub_surrogate, flat_psd):
        times = _times()
        inj = Injection(times=times, detectors=_network(flat_psd), parameters=_params())
        a = inj.generate(stub_surrogate, rng=np.random.default_rng(7))
        b = inj.generate(stub_surrogate, rng=np.random.default_rng(7))
        assert np.array_equal(a.data["H1"], b.data["H1"])


class TestSelfInjectionConsistency:
    """A surrogate self-injection must be recovered at truth: with the noiseless
    signal as data, perturbing tc lowers the likelihood."""

    def test_truth_is_a_local_peak(self, stub_surrogate, flat_psd):
        times = _times()
        dets = _network(flat_psd)
        inj = Injection(times=times, detectors=dets, parameters=_params())
        res = inj.generate(stub_surrogate, rng=np.random.default_rng(0))
        # Use the noiseless signal as data => residual zero at truth.
        net = NetworkLikelihood(data=res.signals, times=times, detectors=dets,
                                surrogate=stub_surrogate, use_waveform_uncertainty=False)
        ll_true = net(_params())
        ll_off = net({**_params(), "tc": TC + 0.004})
        assert ll_true > ll_off


class TestDistanceScaling:

    def test_farther_source_is_quieter(self, stub_surrogate, flat_psd):
        times = _times()
        dets = _network(flat_psd)
        near = Injection(times=times, detectors=dets,
                         parameters={**_params(), "luminosity_distance": 100.0})
        far = Injection(times=times, detectors=dets,
                        parameters={**_params(), "luminosity_distance": 400.0})
        snr_near = near.generate(stub_surrogate, rng=np.random.default_rng(0)).network_snr
        snr_far = far.generate(stub_surrogate, rng=np.random.default_rng(0)).network_snr
        # SNR ∝ 1/distance; 4× distance => ~4× quieter.
        assert snr_near == pytest.approx(4.0 * snr_far, rel=1e-6)
