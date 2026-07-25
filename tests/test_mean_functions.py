"""Tests for heron.models.gp.mean — PN mean functions."""

import torch
import numpy as np
import pytest

from heron.models.gp.mean import (
    ZeroMean,
    NewtonianInspiralMean,
    TaylorT2Mean,
    NewtonianInspiralAmplitudeMean,
    NewtonianInspiralPhaseMean,
    TaylorT2AmplitudeMean,
    TaylorT2PhaseMean,
)


class TestZeroMean:

    def test_returns_zeros(self):
        mean = ZeroMean()
        x = torch.randn(20, 2)
        out = mean(x)
        assert out.shape == (20,)
        assert (out == 0).all()


class TestNewtonianInspiralMean:

    def test_output_shape(self):
        mean = NewtonianInspiralMean(total_mass=60.0, distance=100.0)
        x = torch.tensor([
            [0.5, -0.1],
            [0.5, -0.05],
            [0.5, -0.01],
            [0.5, 0.0],
            [0.5, 0.01],
        ])
        out = mean(x)
        assert out.shape == (5,)

    def test_zero_after_merger(self):
        """Post-merger (t > 0) should be zero in the inspiral approximation."""
        mean = NewtonianInspiralMean(total_mass=60.0, distance=100.0, output_scale=1.0)
        x = torch.tensor([[0.5, 0.01], [0.5, 0.1], [0.5, 1.0]])
        out = mean(x)
        assert (out == 0).all()

    def test_nonzero_during_inspiral(self):
        """Pre-merger (t < 0) should produce non-zero values."""
        mean = NewtonianInspiralMean(total_mass=60.0, distance=100.0, output_scale=1.0)
        x = torch.tensor([[0.5, -0.5], [0.5, -0.1], [0.5, -0.01]])
        out = mean(x)
        assert not (out == 0).all()

    def test_amplitude_scales_with_distance(self):
        """Closer distance should give larger amplitude."""
        x = torch.tensor([[0.5, -0.1]])
        near = NewtonianInspiralMean(total_mass=60.0, distance=50.0, output_scale=1.0)
        far = NewtonianInspiralMean(total_mass=60.0, distance=200.0, output_scale=1.0)
        assert abs(float(near(x))) > abs(float(far(x)))

    def test_output_scale_applied(self):
        """Output should scale with output_scale."""
        x = torch.tensor([[0.5, -0.1]])
        m1 = NewtonianInspiralMean(total_mass=60.0, distance=100.0, output_scale=1.0)
        m2 = NewtonianInspiralMean(total_mass=60.0, distance=100.0, output_scale=1e27)
        ratio = float(m2(x)) / float(m1(x))
        assert ratio == pytest.approx(1e27, rel=1e-3)

    def test_with_warping(self):
        """Should unwarp time before evaluating."""
        from heron.models.warping import ChirpTimeWarping

        warping = ChirpTimeWarping(alpha=0.625)
        mean = NewtonianInspiralMean(
            total_mass=60.0, distance=100.0, output_scale=1.0, warping=warping
        )
        # Warped time coordinates
        raw_times = torch.tensor([-0.5, -0.1, -0.01])
        warped_times = warping.warp(raw_times)
        x_warped = torch.stack([torch.full_like(warped_times, 0.5), warped_times], dim=1)
        out = mean(x_warped)
        assert out.shape == (3,)
        assert not (out == 0).all()


class TestTaylorT2Mean:

    def test_output_shape(self):
        mean = TaylorT2Mean(total_mass=60.0, distance=100.0)
        x = torch.tensor([[0.5, -0.1], [0.8, -0.05], [1.0, -0.01]])
        out = mean(x)
        assert out.shape == (3,)

    def test_differs_from_newtonian(self):
        """1PN corrections should make TaylorT2 differ from Newtonian."""
        x = torch.tensor([[0.5, -0.1]])
        newt = NewtonianInspiralMean(total_mass=60.0, distance=100.0, output_scale=1.0)
        t2 = TaylorT2Mean(total_mass=60.0, distance=100.0, output_scale=1.0)
        # They should differ (1PN phase correction)
        assert abs(float(newt(x)) - float(t2(x))) > 1e-30


class TestAmplitudePhaseMeans:
    """The standalone amplitude/phase means (for PhaseAmplitudeGPSurrogate)
    must reconstruct the combined strain means exactly, since they're
    derived from the same shared PN helper — this is the key check that
    the mean.py refactor didn't change the underlying physics."""

    x = torch.tensor([[0.5, -0.3], [0.7, -0.1], [0.9, -0.02]])

    def test_newtonian_amplitude_phase_reconstructs_strain_mean(self):
        strain_mean = NewtonianInspiralMean(total_mass=60.0, distance=100.0, output_scale=1.0)
        amp_mean = NewtonianInspiralAmplitudeMean(total_mass=60.0, distance=100.0)
        phase_mean = NewtonianInspiralPhaseMean(total_mass=60.0, distance=100.0)

        reconstructed = torch.exp(amp_mean(self.x)) * torch.cos(phase_mean(self.x))
        np.testing.assert_allclose(
            reconstructed.numpy(), strain_mean(self.x).numpy(), atol=1e-25
        )

    def test_taylort2_amplitude_phase_reconstructs_strain_mean(self):
        strain_mean = TaylorT2Mean(total_mass=60.0, distance=100.0, output_scale=1.0)
        amp_mean = TaylorT2AmplitudeMean(total_mass=60.0, distance=100.0)
        phase_mean = TaylorT2PhaseMean(total_mass=60.0, distance=100.0)

        reconstructed = torch.exp(amp_mean(self.x)) * torch.cos(phase_mean(self.x))
        np.testing.assert_allclose(
            reconstructed.numpy(), strain_mean(self.x).numpy(), atol=1e-25
        )

    def test_amplitude_output_shape(self):
        mean = NewtonianInspiralAmplitudeMean(total_mass=60.0, distance=100.0)
        assert mean(self.x).shape == (3,)

    def test_phase_output_shape(self):
        mean = NewtonianInspiralPhaseMean(total_mass=60.0, distance=100.0)
        assert mean(self.x).shape == (3,)

    def test_taylort2_phase_differs_from_newtonian_phase(self):
        newt_phase = NewtonianInspiralPhaseMean(total_mass=60.0, distance=100.0)
        t2_phase = TaylorT2PhaseMean(total_mass=60.0, distance=100.0)
        assert abs(float(newt_phase(self.x)[0]) - float(t2_phase(self.x)[0])) > 1e-10

    def test_with_warping(self):
        from heron.models.warping import ChirpTimeWarping

        warping = ChirpTimeWarping(alpha=0.625)
        amp_mean = NewtonianInspiralAmplitudeMean(total_mass=60.0, distance=100.0, warping=warping)
        phase_mean = NewtonianInspiralPhaseMean(total_mass=60.0, distance=100.0, warping=warping)

        raw_times = torch.tensor([-0.5, -0.1, -0.01])
        warped_times = warping.warp(raw_times)
        x_warped = torch.stack([torch.full_like(warped_times, 0.5), warped_times], dim=1)

        assert amp_mean(x_warped).shape == (3,)
        assert phase_mean(x_warped).shape == (3,)


class TestMeanConfigRoundtrip:
    """mean_to_config / mean_from_config — checkpoint (de)serialization."""

    def test_none_and_zero_means_serialize_to_none(self):
        from heron.models.gp.mean import mean_to_config
        import gpytorch

        assert mean_to_config(None) is None
        assert mean_to_config(ZeroMean()) is None
        assert mean_to_config(gpytorch.means.ZeroMean()) is None
        assert mean_to_config(gpytorch.means.ConstantMean()) is None

    def test_none_config_rebuilds_to_none(self):
        from heron.models.gp.mean import mean_from_config

        assert mean_from_config(None) is None

    @pytest.mark.parametrize("cls,extra", [
        (NewtonianInspiralMean, {"output_scale": 1e26}),
        (TaylorT2Mean, {"output_scale": 1e26}),
        (NewtonianInspiralAmplitudeMean, {}),
        (NewtonianInspiralPhaseMean, {}),
        (TaylorT2AmplitudeMean, {}),
        (TaylorT2PhaseMean, {}),
    ])
    def test_pn_mean_roundtrip(self, cls, extra):
        from heron.models.gp.mean import mean_to_config, mean_from_config
        from heron.models.warping import ChirpTimeWarping

        warping = ChirpTimeWarping(alpha=0.625)
        mean = cls(total_mass=42.0, distance=250.0, warping=warping, **extra)
        config = mean_to_config(mean)
        rebuilt = mean_from_config(config, warping=warping)

        assert type(rebuilt) is cls
        assert rebuilt.total_mass == 42.0
        assert rebuilt.distance == 250.0
        assert rebuilt.warping is warping
        if "output_scale" in extra:
            assert rebuilt.output_scale == extra["output_scale"]

    def test_lal_approximant_mean_roundtrip(self):
        # Construction is lazy (no LAL call until forward), so this needs
        # no lalsuite.
        from heron.models.gp.mean import (
            LALApproximantAmplitudeMean,
            LALApproximantPhaseMean,
            mean_to_config,
            mean_from_config,
        )

        for cls in (LALApproximantAmplitudeMean, LALApproximantPhaseMean):
            mean = cls(approximant="IMRPhenomXAS", total_mass=42.0, distance=250.0)
            config = mean_to_config(mean)
            assert config["type"] == "approximant"
            assert config["approximant"] == "IMRPhenomXAS"
            rebuilt = mean_from_config(config)
            assert type(rebuilt) is cls
            assert rebuilt.approximant == "IMRPhenomXAS"
            assert rebuilt.total_mass == 42.0

    def test_lal_approximant_waveform_mean_roundtrip(self):
        # Construction is lazy (no LAL call until forward), so this needs
        # no lalsuite.
        from heron.models.gp.mean import (
            LALApproximantPlusMean,
            LALApproximantCrossMean,
            mean_to_config,
            mean_from_config,
        )

        for cls, target in (
            (LALApproximantPlusMean, "plus"),
            (LALApproximantCrossMean, "cross"),
        ):
            mean = cls(
                approximant="IMRPhenomXAS", total_mass=42.0, distance=250.0,
                output_scale=1e26, phase_correction=-2.15,
            )
            config = mean_to_config(mean)
            assert config["type"] == "approximant"
            assert config["target"] == target
            assert config["output_scale"] == 1e26
            assert config["phase_correction"] == -2.15
            rebuilt = mean_from_config(config)
            assert type(rebuilt) is cls
            assert rebuilt.approximant == "IMRPhenomXAS"
            assert rebuilt.total_mass == 42.0
            assert rebuilt.output_scale == 1e26
            assert rebuilt.phase_correction == -2.15

    def test_lal_approximant_waveform_mean_phase_correction_defaults_zero(self):
        from heron.models.gp.mean import LALApproximantPlusMean

        mean = LALApproximantPlusMean(approximant="IMRPhenomXAS")
        assert mean.phase_correction == 0.0

    def test_unknown_mean_raises(self):
        from heron.models.gp.mean import mean_to_config
        import gpytorch

        class Strange(gpytorch.means.Mean):
            def forward(self, x):
                return torch.zeros(x.shape[0])

        with pytest.raises(ValueError):
            mean_to_config(Strange())


class TestLALApproximantMeans:
    """Full-IMR approximant mean functions (require lalsuite)."""

    @pytest.fixture(scope="class")
    def phase_mean(self):
        pytest.importorskip("lalsimulation")
        from heron.models.gp.mean import LALApproximantPhaseMean
        from heron.models.warping import ChirpTimeWarping

        warping = ChirpTimeWarping(alpha=0.625)
        return LALApproximantPhaseMean(
            approximant="IMRPhenomXAS", total_mass=60.0, distance=100.0,
            warping=warping,
        )

    def _x(self, warping, q=0.5, times=(-0.5, -0.1, -0.01, 0.005)):
        raw = torch.tensor(times, dtype=torch.float64)
        warped = warping.warp(raw)
        return torch.stack([torch.full_like(warped, q), warped], dim=1)

    def test_forward_shape_dtype_device(self, phase_mean):
        x = self._x(phase_mean.warping).to(torch.float32)
        out = phase_mean(x)
        assert out.shape == (4,)
        assert out.dtype == torch.float32
        assert out.device == x.device

    def test_phase_is_monotonic_through_inspiral(self, phase_mean):
        """Phase convention h+ - i hx = A e^{-i Phi}: Phi increases with
        time through the inspiral (matches the training-target unwrap)."""
        times = np.linspace(-1.0, -0.01, 50)
        x = self._x(phase_mean.warping, times=tuple(times))
        out = phase_mean(x).numpy()
        diffs = np.diff(out)
        assert (diffs > 0).all() or (diffs < 0).all()

    def test_amplitude_mean_tracks_ringdown(self):
        pytest.importorskip("lalsimulation")
        from heron.models.gp.mean import LALApproximantAmplitudeMean
        from heron.models.warping import ChirpTimeWarping

        warping = ChirpTimeWarping(alpha=0.625)
        mean = LALApproximantAmplitudeMean(
            approximant="IMRPhenomXAS", total_mass=60.0, distance=100.0,
            warping=warping,
        )
        # Log-amplitude must decay after merger (ringdown), not freeze.
        x = self._x(warping, times=(-0.001, 0.003, 0.006))
        out = mean(x).numpy()
        assert out[2] < out[0]

    def test_forward_is_cached(self, phase_mean):
        x = self._x(phase_mean.warping)
        out1 = phase_mean(x)
        assert len(phase_mean._forward_cache) >= 1
        out2 = phase_mean(x)
        np.testing.assert_allclose(out1.numpy(), out2.numpy())

    def test_deepcopy_works_after_forward(self, phase_mean):
        import copy

        x = self._x(phase_mean.warping)
        _ = phase_mean(x)  # populate LAL generator + caches
        clone = copy.deepcopy(phase_mean)
        assert clone.approximant == phase_mean.approximant
        np.testing.assert_allclose(
            clone(x).numpy(), phase_mean(x).numpy(), rtol=1e-12
        )


class TestLALApproximantWaveformMeans:
    """Plus/cross full-IMR approximant means for ExactGPSurrogate (require
    lalsuite)."""

    def _x(self, warping, q=0.5, times=(-0.5, -0.1, -0.01, 0.005)):
        raw = torch.tensor(times, dtype=torch.float64)
        warped = warping.warp(raw)
        return torch.stack([torch.full_like(warped, q), warped], dim=1)

    @pytest.fixture(scope="class")
    def plus_mean(self):
        pytest.importorskip("lalsimulation")
        from heron.models.gp.mean import LALApproximantPlusMean
        from heron.models.warping import ChirpTimeWarping

        warping = ChirpTimeWarping(alpha=0.625)
        return LALApproximantPlusMean(
            approximant="IMRPhenomXAS", total_mass=60.0, distance=100.0,
            warping=warping, output_scale=1e21,
        )

    def test_forward_shape_dtype_and_scale(self, plus_mean):
        x = self._x(plus_mean.warping).to(torch.float32)
        out = plus_mean(x)
        assert out.shape == (4,)
        assert out.dtype == torch.float32
        # output_scale=1e21 applied to raw strain (~1e-21) -> O(1).
        assert out.abs().max() < 1e6

    def test_zero_outside_native_support(self, plus_mean):
        """Unlike the amplitude/phase means (frozen boundary
        extrapolation), strain must go to (near) zero for query times well
        outside the waveform's native support -- freezing at a nonzero
        boundary value would inject a spurious constant offset."""
        far_past = self._x(plus_mean.warping, times=(-1e6,))
        out = plus_mean(far_past).numpy()
        assert abs(out[0]) < 1e-8

    def test_plus_and_cross_differ(self):
        pytest.importorskip("lalsimulation")
        from heron.models.gp.mean import LALApproximantPlusMean, LALApproximantCrossMean
        from heron.models.warping import ChirpTimeWarping

        warping = ChirpTimeWarping(alpha=0.625)
        plus = LALApproximantPlusMean(
            approximant="IMRPhenomXAS", total_mass=60.0, distance=100.0,
            warping=warping,
        )
        cross = LALApproximantCrossMean(
            approximant="IMRPhenomXAS", total_mass=60.0, distance=100.0,
            warping=warping,
        )
        x = self._x(warping, times=(-0.1, -0.05, -0.01))
        out_plus = plus(x).numpy()
        out_cross = cross(x).numpy()
        assert not np.allclose(out_plus, out_cross)

    def test_deepcopy_preserves_output_scale(self, plus_mean):
        import copy

        x = self._x(plus_mean.warping)
        _ = plus_mean(x)  # populate LAL generator + caches
        clone = copy.deepcopy(plus_mean)
        assert clone.output_scale == plus_mean.output_scale
        np.testing.assert_allclose(
            clone(x).numpy(), plus_mean(x).numpy(), rtol=1e-12
        )

    def test_deepcopy_preserves_phase_correction(self, plus_mean):
        import copy

        clone = copy.deepcopy(plus_mean)
        assert clone.phase_correction == plus_mean.phase_correction

    def test_phase_correction_shifts_reconstructed_phase(self):
        """A pure phase_correction change must rotate h_plus/h_cross
        together the way `A*cos(Phi-c)`/`A*sin(Phi-c)` predicts -- not
        change the envelope."""
        pytest.importorskip("lalsimulation")
        from heron.models.gp.mean import LALApproximantPlusMean, LALApproximantCrossMean
        from heron.models.warping import ChirpTimeWarping

        warping = ChirpTimeWarping(alpha=0.625)
        x = self._x(warping, times=(-0.1, -0.05, -0.01))
        c = 1.3

        plus0 = LALApproximantPlusMean(
            approximant="IMRPhenomXAS", total_mass=60.0, distance=100.0, warping=warping,
        )
        cross0 = LALApproximantCrossMean(
            approximant="IMRPhenomXAS", total_mass=60.0, distance=100.0, warping=warping,
        )
        plusc = LALApproximantPlusMean(
            approximant="IMRPhenomXAS", total_mass=60.0, distance=100.0, warping=warping,
            phase_correction=c,
        )
        crossc = LALApproximantCrossMean(
            approximant="IMRPhenomXAS", total_mass=60.0, distance=100.0, warping=warping,
            phase_correction=c,
        )

        hp0, hx0 = plus0(x).numpy(), cross0(x).numpy()
        hpc, hxc = plusc(x).numpy(), crossc(x).numpy()

        # Envelope amplitude is unchanged by a pure phase rotation.
        np.testing.assert_allclose(hp0**2 + hx0**2, hpc**2 + hxc**2, rtol=1e-8)
        # h_plus - i*h_cross picks up a factor e^{+i*c} under Phi -> Phi-c.
        z0 = hp0 - 1j * hx0
        zc = hpc - 1j * hxc
        np.testing.assert_allclose(zc, z0 * np.exp(1j * c), rtol=1e-6)


class TestComputePhaseCorrection:
    """compute_phase_correction -- the D-vs-XAS phase-convention offset
    measurement motivating LALApproximantPlusMean/CrossMean's
    phase_correction (see grid_snap/phase-alignment session notes)."""

    def test_is_near_constant_across_mass_ratio(self):
        pytest.importorskip("lalsimulation")
        from heron.models.gp.mean import compute_phase_correction

        values = [
            compute_phase_correction(
                "IMRPhenomXAS", "IMRPhenomD", reference_mass_ratio=q,
            )
            for q in (0.2, 0.5, 0.8)
        ]
        # Measured empirically at -2.24..-2.13 rad over q=0.15-0.9 -- a
        # phi_ref/f_ref convention difference, not a per-q physical effect.
        for v in values:
            assert -2.5 < v < -1.8
        assert max(values) - min(values) < 0.3

    def test_self_comparison_is_zero(self):
        pytest.importorskip("lalsimulation")
        from heron.models.gp.mean import compute_phase_correction

        c = compute_phase_correction("IMRPhenomD", "IMRPhenomD", reference_mass_ratio=0.5)
        assert abs(c) < 1e-6
