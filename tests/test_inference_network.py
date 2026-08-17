"""Tests for heron.inference.network.NetworkLikelihood."""
import numpy as np
import pytest
from scipy.special import logsumexp

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
    mu, _ = project_polarisations(
        wf, f_plus=fp, f_cross=fc,
        inclination=params.get("inclination", 0.0),
        coalescence_phase=params.get("coalescence_phase", 0.0),
    )
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
        # var=25 (not the smaller value this used to use): once K is
        # band-limited by _project_diag (see network.py/gw_likelihood.py),
        # the comparison that matters is against C's actual *passband*
        # eigenvalues, not its raw time-domain diagonal -- a smaller var is
        # swamped now that the spurious sub-f_low leakage that used to
        # inflate this comparison is removed.
        surr = StubSurrogate(dip_at=Q, var=25.0, dip_var=1e-12)
        det = Detector.from_name("H1", psd_fn=flat_psd)
        params = {"mass_ratio": Q, "tc": TC, "ra": RA, "dec": DEC, "psi": PSI}
        data = _detector_signal(surr, det, params, times)

        net_raw = NetworkLikelihood(data={"H1": data}, times=times, detectors=[det],
                                    surrogate=surr)
        net_env = NetworkLikelihood(data={"H1": data}, times=times, detectors=[det],
                                    surrogate=surr, k_smoothing_offsets=[0.05, -0.05])
        # At the dip, enveloping over off-dip offsets raises K => changes logL.
        assert abs(net_raw(params) - net_env(params)) > 1.0


class TestPhaseMarginalisation:
    """Analytic phase marginalisation vs. brute-force numerical integration.

    ``StubSurrogate`` emits exact sinusoidal quadratures (h+ = A sin, h× = A
    cos) and the fixture PSD is flat, so C is stationary/circulant and K = var
    * I is an isotropic multiple of the identity (commutes with everything).
    Both conditions needed for the closed-form marginalisation to be *exact*
    (see the ``NetworkLikelihood`` module docstring) hold exactly here, not
    just approximately, so brute-force and analytic should agree tightly.
    """

    PHI_TRUE = 1.1

    @staticmethod
    def _brute_force_log_marginal(net, params, n=4096):
        """Riemann-sum ``log E_phi[L(phi)]`` over a fine uniform phase grid."""
        phis = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        logl = np.array([net({**params, "coalescence_phase": phi}) for phi in phis])
        return logsumexp(logl) - np.log(n)

    @pytest.mark.parametrize("use_waveform_uncertainty", [True, False])
    def test_single_detector_matches_brute_force(
        self, stub_surrogate, flat_psd, use_waveform_uncertainty,
    ):
        times = _times()
        det = Detector.from_name("H1", psd_fn=flat_psd)
        params = {"mass_ratio": Q, "tc": TC, "ra": RA, "dec": DEC, "psi": PSI}
        data = _detector_signal(
            stub_surrogate, det, {**params, "coalescence_phase": self.PHI_TRUE}, times,
        )

        net_fixed = NetworkLikelihood(
            data={"H1": data}, times=times, detectors=[det], surrogate=stub_surrogate,
            use_waveform_uncertainty=use_waveform_uncertainty,
        )
        net_marg = NetworkLikelihood(
            data={"H1": data}, times=times, detectors=[det], surrogate=stub_surrogate,
            use_waveform_uncertainty=use_waveform_uncertainty, marginalize_phase=True,
        )

        brute = self._brute_force_log_marginal(net_fixed, params)
        assert net_marg(params) == pytest.approx(brute, abs=1e-2)

    def test_two_detector_network_matches_brute_force(self, stub_surrogate, flat_psd):
        times = _times()
        h1 = Detector.from_name("H1", psd_fn=flat_psd)
        l1 = Detector.from_name("L1", psd_fn=flat_psd)
        params = {"mass_ratio": Q, "tc": TC, "ra": RA, "dec": DEC, "psi": PSI}
        data = {
            d.prefix: _detector_signal(
                stub_surrogate, d, {**params, "coalescence_phase": self.PHI_TRUE}, times,
            )
            for d in (h1, l1)
        }

        net_fixed = NetworkLikelihood(data=data, times=times, detectors=[h1, l1],
                                      surrogate=stub_surrogate)
        net_marg = NetworkLikelihood(data=data, times=times, detectors=[h1, l1],
                                     surrogate=stub_surrogate, marginalize_phase=True)

        # Coherent network marginalisation is NOT the sum of per-detector
        # marginals (phase is shared, not independent per detector) -- the
        # brute-force reference must sum both detectors' logL at each phase
        # *before* integrating, which net_fixed already does internally.
        brute = self._brute_force_log_marginal(net_fixed, params)
        assert net_marg(params) == pytest.approx(brute, abs=1e-2)

    def test_rejects_explicit_phase_param(self, stub_surrogate, flat_psd):
        times = _times()
        det = Detector.from_name("H1", psd_fn=flat_psd)
        params = {"mass_ratio": Q, "tc": TC, "ra": RA, "dec": DEC, "psi": PSI,
                  "coalescence_phase": 0.3}
        data = _detector_signal(stub_surrogate, det, params, times)
        net = NetworkLikelihood(data={"H1": data}, times=times, detectors=[det],
                                surrogate=stub_surrogate, marginalize_phase=True)
        with pytest.raises(ValueError):
            net(params)

    def test_marginal_bounded_by_fixed_phase_extremes(self, stub_surrogate, flat_psd):
        """logL_marg = log E_phi[L(phi)] is a (log-)convex combination of the
        fixed-phase likelihoods over a uniform density, so it must lie between
        the min and max over phase -- a coarse sanity check independent of the
        tighter brute-force integration tests above."""
        times = _times()
        det = Detector.from_name("H1", psd_fn=flat_psd)
        params = {"mass_ratio": Q, "tc": TC, "ra": RA, "dec": DEC, "psi": PSI}
        data = _detector_signal(
            stub_surrogate, det, {**params, "coalescence_phase": self.PHI_TRUE}, times,
        )
        net_fixed = NetworkLikelihood(data={"H1": data}, times=times, detectors=[det],
                                      surrogate=stub_surrogate)
        net_marg = NetworkLikelihood(data={"H1": data}, times=times, detectors=[det],
                                     surrogate=stub_surrogate, marginalize_phase=True)

        grid = np.linspace(0.0, 2 * np.pi, 256, endpoint=False)
        fixed_vals = [net_fixed({**params, "coalescence_phase": phi}) for phi in grid]
        marg = net_marg(params)
        assert min(fixed_vals) - 1e-6 <= marg <= max(fixed_vals) + 1e-6
