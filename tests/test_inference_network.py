"""Tests for heron.inference.network.NetworkLikelihood."""
import numpy as np
import pytest

from heron.gw_likelihood import GWLikelihood
from heron.detector import antenna_patterns, time_delay_from_geocentre
from heron.inference.detectors import Detector
from heron.inference.network import NetworkLikelihood
from heron.inference.projection import project_polarisations


TC = 1187008882.4
RA, DEC, PSI, Q = 1.95, -1.27, 0.82, 0.8


def _times(n=256, fs=512.0):
    return TC + (np.arange(n) - n // 2) / fs


def _detector_signal(surrogate, det, params, times):
    """Noiseless projected signal for one detector at *params* (geocentre tc)."""
    tc = params["tc"]
    dt_geo = det.time_delay_from_geocentre(params["ra"], params["dec"], tc)
    t_rel = times - (tc + dt_geo)
    wf = surrogate.predict({"mass_ratio": params.get("mass_ratio", Q), "times": t_rel})
    fp, fc = det.antenna_patterns(params["ra"], params["dec"], params["psi"], tc)
    mu, _ = project_polarisations(wf, f_plus=fp, f_cross=fc)
    return mu


class TestGWLikelihoodParity:
    """A single-detector NetworkLikelihood reproduces GWLikelihood. GW uses
    detector-frame tc, Network uses geocentre tc; they map by the geocentre
    delay (the only residual is the sub-µs GMST shift in the antenna patterns)."""

    def test_single_detector_matches(self, stub_surrogate, flat_psd):
        times = _times()
        det_name = "H1"
        fp, fc = antenna_patterns(RA, DEC, PSI, TC, det_name)
        wf = stub_surrogate.predict({"times": times - TC})
        mu_true, _ = project_polarisations(wf, f_plus=fp, f_cross=fc)
        data = mu_true

        gw = GWLikelihood(data=data, times=times, psd_fn=flat_psd,
                          surrogate=stub_surrogate, detector=det_name, f_low=20.0)
        net = NetworkLikelihood(
            data={det_name: data}, times=times,
            detectors=[Detector.from_name(det_name, psd_fn=flat_psd)],
            surrogate=stub_surrogate, f_low=20.0,
        )
        delay = time_delay_from_geocentre(RA, DEC, TC, det_name)
        p = {"mass_ratio": Q, "ra": RA, "dec": DEC, "psi": PSI}
        ll_gw = gw({**p, "tc": TC})
        ll_net = net({**p, "tc": TC - delay})
        assert ll_net == pytest.approx(ll_gw, abs=1e-3)

    def test_bare_array_data_accepted_for_single_detector(self, stub_surrogate, flat_psd):
        times = _times()
        data = _detector_signal(
            stub_surrogate, Detector.from_name("H1"),
            {"tc": TC, "ra": RA, "dec": DEC, "psi": PSI}, times,
        )
        net = NetworkLikelihood(
            data=data, times=times,
            detectors=Detector.from_name("H1", psd_fn=flat_psd),
            surrogate=stub_surrogate,
        )
        ll = net({"mass_ratio": Q, "tc": TC, "ra": RA, "dec": DEC, "psi": PSI})
        assert np.isfinite(ll)


class TestNetworkSum:

    def test_two_detector_equals_sum(self, stub_surrogate, flat_psd):
        times = _times()
        params = {"mass_ratio": Q, "tc": TC, "ra": RA, "dec": DEC, "psi": PSI}
        h1 = Detector.from_name("H1", psd_fn=flat_psd)
        l1 = Detector.from_name("L1", psd_fn=flat_psd)
        data = {d.prefix: _detector_signal(stub_surrogate, d, params, times) for d in (h1, l1)}

        net2 = NetworkLikelihood(data=data, times=times, detectors=[h1, l1],
                                 surrogate=stub_surrogate)
        net_h1 = NetworkLikelihood(data={"H1": data["H1"]}, times=times, detectors=[h1],
                                   surrogate=stub_surrogate)
        net_l1 = NetworkLikelihood(data={"L1": data["L1"]}, times=times, detectors=[l1],
                                   surrogate=stub_surrogate)
        assert net2(params) == pytest.approx(net_h1(params) + net_l1(params), rel=1e-9)


class TestInjectionRecovery:

    @pytest.fixture
    def zero_noise_network(self, stub_surrogate, flat_psd):
        times = _times()
        params = {"mass_ratio": Q, "tc": TC, "ra": RA, "dec": DEC, "psi": PSI}
        dets = [Detector.from_name("H1", psd_fn=flat_psd),
                Detector.from_name("L1", psd_fn=flat_psd)]
        data = {d.prefix: _detector_signal(stub_surrogate, d, params, times) for d in dets}
        net = NetworkLikelihood(data=data, times=times, detectors=dets,
                                surrogate=stub_surrogate)
        return net, params

    def test_truth_beats_tc_perturbation(self, zero_noise_network):
        net, params = zero_noise_network
        ll_true = net(params)
        ll_shift = net({**params, "tc": params["tc"] + 0.005})
        assert ll_true > ll_shift

    def test_truth_beats_sky_perturbation(self, zero_noise_network):
        net, params = zero_noise_network
        ll_true = net(params)
        ll_shift = net({**params, "ra": params["ra"] + 0.4})
        assert ll_true > ll_shift

    def test_no_uncertainty_path(self, zero_noise_network, stub_surrogate, flat_psd):
        net, params = zero_noise_network
        # Rebuild with K=0 and confirm it still evaluates and peaks at truth.
        times = _times()
        dets = [Detector.from_name("H1", psd_fn=flat_psd),
                Detector.from_name("L1", psd_fn=flat_psd)]
        data = {d.prefix: _detector_signal(stub_surrogate, d, params, times) for d in dets}
        net0 = NetworkLikelihood(data=data, times=times, detectors=dets,
                                 surrogate=stub_surrogate, use_waveform_uncertainty=False)
        assert np.isfinite(net0(params))
        assert net0(params) > net0({**params, "tc": params["tc"] + 0.005})


class TestTimeDelayMatters:

    def test_arrival_shifts_with_sky_position(self, stub_surrogate, flat_psd):
        """Data built with one sky position is best-fit at that position; a very
        different sky position (different geocentre delay) fits worse."""
        times = _times()
        det = Detector.from_name("H1", psd_fn=flat_psd)
        params = {"mass_ratio": Q, "tc": TC, "ra": RA, "dec": DEC, "psi": PSI}
        data = _detector_signal(stub_surrogate, det, params, times)
        net = NetworkLikelihood(data={"H1": data}, times=times, detectors=[det],
                                surrogate=stub_surrogate, use_waveform_uncertainty=False)
        ll_true = net(params)
        # Antipodal-ish sky point => different delay => phase-misaligned template.
        ll_off = net({**params, "ra": RA + np.pi, "dec": -DEC})
        assert ll_true > ll_off


class TestKSmoothing:

    def test_envelope_lifts_variance_at_dip(self, flat_psd):
        from tests.conftest import StubSurrogate
        times = _times(n=64)
        surr = StubSurrogate(dip_at=Q, var=0.25, dip_var=1e-12)
        det = Detector.from_name("H1", psd_fn=flat_psd)
        params = {"mass_ratio": Q, "tc": TC, "ra": RA, "dec": DEC, "psi": PSI}
        data = _detector_signal(surr, det, params, times)

        net_raw = NetworkLikelihood(data={"H1": data}, times=times, detectors=[det],
                                    surrogate=surr)
        net_env = NetworkLikelihood(data={"H1": data}, times=times, detectors=[det],
                                    surrogate=surr, k_smoothing_offsets=[0.05, -0.05])
        # At the dip, enveloping over off-dip offsets raises K => changes logL.
        assert abs(net_raw(params) - net_env(params)) > 1.0
