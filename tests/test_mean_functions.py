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
