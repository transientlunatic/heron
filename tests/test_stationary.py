"""Tests for heron.stationary: exact FFT/Woodbury evaluation of
log N(d; mu, C + P diag(v) P) against the dense MarginalLogLikelihood."""
import numpy as np
import pytest
import torch
from scipy.special import logsumexp

from heron.inference.detectors import Detector, aligo_design_psd
from heron.inference.network import NetworkLikelihood
from heron.inference.projection import variance_window
from heron.likelihood import MarginalLogLikelihood
from heron.noise import noise_covariance
from heron.stationary import StationaryMarginalLikelihood, StationaryNoise

from test_inference_network import DEC, PSI, Q, RA, TC, _detector_signal, _times

N, FS, F_LOW = 512, 512.0, 20.0


def _psd(freqs):
    """aLIGO design, scaled to O(1) so tolerances are easy to read."""
    return aligo_design_psd(freqs) / 1e-46


@pytest.fixture(scope="module")
def setup():
    times = np.arange(N) / FS
    freqs = np.fft.rfftfreq(N, d=1 / FS)
    mask = freqs >= F_LOW
    F = np.fft.rfft(np.eye(N), axis=0)
    F[~mask] = 0.0
    P = np.fft.irfft(F, n=N, axis=0)
    C = noise_covariance(times, _psd, f_low=F_LOW)
    noise = StationaryNoise(times, _psd, f_low=F_LOW)
    rng = np.random.default_rng(42)
    hp = lambda x: np.fft.irfft(np.fft.rfft(x) * mask, n=N)
    mu = hp(np.sin(2 * np.pi * 60 * times) * np.exp(-((times - 0.5) / 0.1) ** 2))
    d = mu + hp(rng.standard_normal(N))
    return dict(times=times, P=P, C=C, noise=noise, mu=mu, d=d, hp=hp, rng=rng)


def _dense(s, v):
    K = 0.0 if v is None else (s["P"] * np.sqrt(v)) @ (s["P"] * np.sqrt(v)).T
    return MarginalLogLikelihood(s["C"], s["mu"], K)


class TestStationaryNoise:
    def test_noise_covariance_is_circulant(self, setup):
        C, noise = setup["C"], setup["noise"]
        np.testing.assert_allclose(noise.dense_covariance(), C, rtol=0, atol=1e-12 * np.abs(C).max())

    def test_log_det(self, setup):
        _, logdet = np.linalg.slogdet(setup["C"])
        assert setup["noise"].log_det() == pytest.approx(logdet, rel=1e-10)

    def test_highpass_matches_fft_mask(self, setup):
        x = setup["rng"].standard_normal(N)
        np.testing.assert_allclose(setup["noise"].highpass(x).numpy(), setup["hp"](x), atol=1e-12)


class TestStationaryMarginal:
    @pytest.mark.parametrize("case", ["none", "constant", "small", "compact", "full"])
    def test_matches_dense(self, setup, case):
        t = setup["times"]
        v = {
            "none": None,
            "constant": np.full(N, 0.3),
            "small": 1e-4 * (1 + np.cos(2 * np.pi * 5 * t)),
            "compact": np.where(np.abs(t - 0.5) < 0.05, 2.0, 0.0),
            "full": 0.5 + 2.0 * np.exp(-((t - 0.5) / 0.1) ** 2),
        }[case]
        expected_method = {"none": "none", "constant": "none", "small": "series",
                           "compact": "support", "full": "support"}[case]
        fast = StationaryMarginalLikelihood(setup["noise"], setup["mu"], v)
        assert fast.method == expected_method
        ref = _dense(setup, v)
        # Dense Cholesky of C (condition ~1e8 from the jitter-level sub-f_low
        # eigenvalues) is itself only good to ~1e-10 relative.
        tol = 1e-9 * abs(ref.log_det) + fast.logdet_error_bound
        assert fast.log_det == pytest.approx(ref.log_det, abs=tol)
        assert fast(setup["d"]) == pytest.approx(ref(setup["d"]), abs=tol)

    def test_dense_fallback(self, setup):
        t = setup["times"]
        v = 0.5 + 2.0 * np.exp(-((t - 0.5) / 0.1) ** 2)
        fast = StationaryMarginalLikelihood(setup["noise"], setup["mu"], v, max_support=16)
        assert fast.method == "dense"
        assert fast(setup["d"]) == pytest.approx(_dense(setup, v)(setup["d"]), rel=1e-9)

    def test_refuses_dense_when_disallowed(self, setup):
        v = np.where(np.arange(N) % 2 == 0, 5.0, 0.0)
        with pytest.raises(RuntimeError):
            StationaryMarginalLikelihood(setup["noise"], setup["mu"], v,
                                         max_support=16, allow_dense=False)

    def test_series_error_within_bound(self, setup):
        t = setup["times"]
        v = 3e-3 * np.exp(-((t - 0.5) / 0.2) ** 2)
        fast = StationaryMarginalLikelihood(setup["noise"], setup["mu"], v, logdet_tol=1.0)
        assert fast.method == "series" and fast.logdet_error_bound > 0
        exact = StationaryMarginalLikelihood(setup["noise"], setup["mu"], v, logdet_tol=0.0)
        assert exact.method == "support"
        assert abs(fast.log_det - exact.log_det) <= fast.logdet_error_bound

    def test_cg_path_quadratic(self, setup):
        # Series log-det uses CG for the quadratic form; check it on its own.
        t = setup["times"]
        v = 1e-4 * (1 + np.cos(2 * np.pi * 5 * t))
        fast = StationaryMarginalLikelihood(setup["noise"], setup["mu"], v)
        assert fast.method == "series"
        ref = _dense(setup, v)
        r = setup["d"] - setup["mu"]
        quad_ref = float(torch.dot(ref.whiten(r), ref.whiten(r)))
        assert fast.inner(r, r) == pytest.approx(quad_ref, rel=1e-10)

    def test_inner_is_bilinear_form(self, setup):
        t = setup["times"]
        v = np.where(np.abs(t - 0.5) < 0.05, 2.0, 0.1)
        fast = StationaryMarginalLikelihood(setup["noise"], setup["mu"], v)
        ref = _dense(setup, v)
        x, y = setup["mu"], setup["d"]
        expected = float(torch.dot(ref.whiten(x), ref.whiten(y)))
        assert fast.inner(x, y) == pytest.approx(expected, rel=1e-9)
        assert fast.inner(x, y) == pytest.approx(fast.inner(y, x), rel=1e-12)


class TestNetworkParity:
    """linalg="stationary" reproduces linalg="dense" end to end."""

    def _pair(self, surrogate, psd, **kw):
        times = _times()
        params = {"mass_ratio": Q, "tc": TC, "ra": RA, "dec": DEC, "psi": PSI}
        dets = [Detector.from_name("H1", psd_fn=psd), Detector.from_name("L1", psd_fn=psd)]
        rng = np.random.default_rng(1)
        data = {d.prefix: _detector_signal(surrogate, d, params, times)
                + 0.1 * rng.standard_normal(len(times)) for d in dets}
        make = lambda linalg: NetworkLikelihood(
            data=data, times=times, detectors=dets, surrogate=surrogate,
            linalg=linalg, **kw)
        return make("stationary"), make("dense"), params

    @pytest.mark.parametrize("use_k", [True, False])
    @pytest.mark.parametrize("inflation", [1.0, 1e4])
    def test_fixed_phase(self, stub_surrogate, flat_psd, use_k, inflation):
        fast, dense, params = self._pair(stub_surrogate, flat_psd,
                                         use_waveform_uncertainty=use_k,
                                         covariance_inflation=inflation)
        for dtc in (0.0, 0.003, -0.01):
            p = {**params, "tc": params["tc"] + dtc}
            assert fast(p) == pytest.approx(dense(p), abs=1e-6)

    @pytest.mark.parametrize("use_k", [True, False])
    def test_marginal_phase(self, stub_surrogate, flat_psd, use_k):
        fast, dense, params = self._pair(stub_surrogate, flat_psd,
                                         use_waveform_uncertainty=use_k,
                                         marginalize_phase=True)
        assert fast(params) == pytest.approx(dense(params), abs=1e-6)

    def test_stationary_is_default(self, stub_surrogate, flat_psd):
        fast, _, _ = self._pair(stub_surrogate, flat_psd)
        assert fast._linalg == "stationary" and fast._P is None

    def test_rejects_unknown_linalg(self, stub_surrogate, flat_psd):
        det = Detector.from_name("H1", psd_fn=flat_psd)
        with pytest.raises(ValueError):
            NetworkLikelihood(data=np.zeros(256), times=_times(), detectors=det,
                              surrogate=stub_surrogate, linalg="magic")


class TestVarianceTaper:
    def test_window_shape(self):
        t = np.array([-1.0, -0.51, -0.5, 0.0, 0.1, 0.11, 0.12, 0.2])
        w = variance_window(t, (-0.5, 0.1), roll_off=0.02)
        np.testing.assert_allclose(w[[2, 3, 4]], 1.0)       # inside
        np.testing.assert_allclose(w[[0, 7]], 0.0)          # well outside
        assert w[5] == pytest.approx(0.5)                    # half-way down
        assert 0.0 < w[1] < 1.0
        np.testing.assert_allclose(variance_window(t, (-0.5, 0.1), 0.0),
                                   [0, 0, 1, 1, 1, 0, 0, 0])

    @pytest.fixture
    def windowed_surrogate(self, stub_surrogate):
        class Windowed(type(stub_surrogate)):
            def training_time_bounds(self, parameters):
                return (-0.1, 0.05)
        return Windowed(var=1e-2)

    def _net(self, surrogate, psd, **kw):
        times = _times()
        dets = [Detector.from_name("H1", psd_fn=psd)]
        params = {"mass_ratio": Q, "tc": TC, "ra": RA, "dec": DEC, "psi": PSI}
        data = {"H1": _detector_signal(surrogate, dets[0], params, times)}
        return NetworkLikelihood(data=data, times=times, detectors=dets,
                                 surrogate=surrogate, **kw), params

    def test_taper_reduces_logdet_and_matches_dense(self, windowed_surrogate, flat_psd):
        tapered, params = self._net(windowed_surrogate, flat_psd)
        untapered, _ = self._net(windowed_surrogate, flat_psd, variance_taper=None)
        dense, _ = self._net(windowed_surrogate, flat_psd, linalg="dense")
        assert tapered(params) != pytest.approx(untapered(params), abs=1e-3)
        assert tapered(params) == pytest.approx(dense(params), abs=1e-6)

    def test_continuous_in_tc(self, windowed_surrogate, flat_psd):
        net, params = self._net(windowed_surrogate, flat_psd)
        tcs = params["tc"] + np.linspace(-2e-3, 2e-3, 41)
        ll = np.array([net({**params, "tc": tc}) for tc in tcs])
        # Sub-sample tc steps must not produce jumps from the window edges.
        assert np.max(np.abs(np.diff(ll, 2))) < 0.05 * np.ptp(ll) + 1e-6

    def test_untapered_without_bounds(self, stub_surrogate, flat_psd):
        a, params = self._net(stub_surrogate, flat_psd)
        b, _ = self._net(stub_surrogate, flat_psd, variance_taper=None)
        assert a(params) == b(params)


class TestShiftedForms:
    """shifted_quadratic / correlate against per-shift brute force."""

    @pytest.mark.parametrize("case", ["none", "small", "compact"])
    def test_shifted_quadratic(self, setup, case):
        t = setup["times"]
        v = {"none": None,
             "small": 1e-4 * np.exp(-((t - 0.5) / 0.1) ** 2),
             "compact": np.where(np.abs(t - 0.5) < 0.05, 2.0, 0.0)}[case]
        fast = StationaryMarginalLikelihood(setup["noise"], setup["mu"], v)
        d = setup["d"]
        shifts = np.arange(-7, 8)
        got = fast.shifted_quadratic(d, shifts).numpy()
        want = [fast.inner(np.roll(d, -k), np.roll(d, -k)) for k in shifts]
        np.testing.assert_allclose(got, want, rtol=1e-10)

    def test_correlate(self, setup):
        from heron.stationary import correlate
        x, y = torch.as_tensor(setup["mu"]), torch.as_tensor(setup["d"])
        shifts = np.arange(-5, 6)
        got = correlate(x, y, shifts).numpy()
        want = [float(np.dot(setup["mu"], np.roll(setup["d"], -k))) for k in shifts]
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


class _CompactStub:
    """Sine-Gaussian stub, compact in time, with a finite training window."""

    distance_factor = 100.0

    def __init__(self, var=1e-2, amp=100.0):
        self.var = var
        self.amp = amp

    def training_time_bounds(self, parameters):
        return (-0.08, 0.04)

    def predict(self, params):
        from heron.types import Waveform, WaveformDict

        t = np.asarray(params["times"], dtype=float)
        env = np.exp(-(t / 0.03) ** 2)
        hp = self.amp * env * np.sin(2 * np.pi * 60 * t)
        hx = self.amp * env * np.cos(2 * np.pi * 60 * t)
        var = self.var * env
        return WaveformDict(
            plus=Waveform(data=hp, times=t, variance=var),
            cross=Waveform(data=hx, times=t, variance=var),
        )


class TestTimeMarginalisation:
    PRIOR = (TC - 0.02, TC + 0.02)

    def _nets(self, psd, var=1e-2, time_resolution=1.0 / 512.0, **kw):
        surrogate = _CompactStub(var=var)
        times = _times(n=512, fs=512.0)
        params = {"mass_ratio": Q, "tc": TC + 0.004, "ra": RA, "dec": DEC, "psi": PSI,
                  "inclination": 0.4, "coalescence_phase": 0.7}
        dets = [Detector.from_name("H1", psd_fn=psd), Detector.from_name("L1", psd_fn=psd)]
        rng = np.random.default_rng(7)
        data = {d.prefix: _detector_signal(surrogate, d, params, times)
                + 5.0 * rng.standard_normal(len(times)) for d in dets}
        make = lambda **extra: NetworkLikelihood(
            data=data, times=times, detectors=dets, surrogate=surrogate, **kw, **extra)
        return (make(marginalize_time=True, time_prior=self.PRIOR,
                     time_resolution=time_resolution), make(), params)

    @pytest.mark.parametrize("use_k,var", [(False, 1e-2), (True, 50.0), (True, 5e4)])
    @pytest.mark.parametrize("marg_phase", [False, True])
    def test_series_matches_brute_force(self, flat_psd, use_k, var, marg_phase):
        tm, plain, params = self._nets(flat_psd, var=var, use_waveform_uncertainty=use_k,
                                       marginalize_phase=marg_phase)
        p = {k: v for k, v in params.items() if k != "tc"}
        if marg_phase:
            p.pop("coalescence_phase")
        grid, series = tm.time_series(p)
        # The only approximation is antenna patterns / delays held at the prior
        # centre (Earth rotation over +-20 ms): ~1e-6 of the logL range.
        tol = 1e-5 * max(1.0, np.ptp(series))
        for i in range(0, len(grid), 3):
            assert series[i] == pytest.approx(plain({**p, "tc": grid[i]}), abs=tol)

    @pytest.mark.parametrize("marg_phase", [False, True])
    def test_subsample_grid_matches_brute_force(self, flat_psd, marg_phase):
        tm, plain, params = self._nets(flat_psd, var=50.0, time_resolution=5e-5,
                                       marginalize_phase=marg_phase)
        assert tm._oversample == 40
        p = {k: v for k, v in params.items() if k != "tc"}
        if marg_phase:
            p.pop("coalescence_phase")
        grid, series = tm.time_series(p)
        # Band-limited fractional shift + interpolated data term: ~1e-5 of range.
        tol = 5e-5 * np.ptp(series)
        for i in np.linspace(0, len(grid) - 1, 15).astype(int):
            assert series[i] == pytest.approx(plain({**p, "tc": grid[i]}), abs=tol)

    def test_marginal_is_grid_average(self, flat_psd):
        tm, plain, params = self._nets(flat_psd)
        p = {k: v for k, v in params.items() if k != "tc"}
        grid, series = tm.time_series(p)
        vals = np.array([plain({**p, "tc": g}) for g in grid])
        assert tm(p) == pytest.approx(logsumexp(vals) - np.log(len(vals)),
                                      abs=1e-5 * np.ptp(vals))
        # The peak lands on the injected tc.
        assert abs(grid[np.argmax(series)] - params["tc"]) <= 1.5 / 512.0

    def test_rejects_tc_and_needs_prior(self, flat_psd, stub_surrogate):
        tm, _, params = self._nets(flat_psd)
        with pytest.raises(ValueError):
            tm(params)
        with pytest.raises(ValueError):
            NetworkLikelihood(data=np.zeros(256), times=_times(),
                              detectors=Detector.from_name("H1", psd_fn=flat_psd),
                              surrogate=stub_surrogate, marginalize_time=True)
