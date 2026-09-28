"""Tests for heron.training — dataset, sampling, and active learning."""

import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

from heron.training.dataset import TrainingSet
from heron.training.sampling import (
    sobol_sample,
    latin_hypercube_sample,
    jittered_grid_sample,
)


class TestTrainingSet:

    def _make_set(self, n=100):
        x = torch.randn(n, 2)
        y_plus = torch.randn(n)
        y_cross = torch.randn(n)
        return TrainingSet(x=x, y_plus=y_plus, y_cross=y_cross)

    def test_len(self):
        ts = self._make_set(50)
        assert len(ts) == 50

    def test_n_parameters(self):
        ts = self._make_set()
        assert ts.n_parameters == 1  # 2 columns, last is time

    def test_append(self):
        a = self._make_set(50)
        b = self._make_set(30)
        c = a.append(b)
        assert len(c) == 80

    def test_save_load_roundtrip(self):
        ts = self._make_set(40)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "training.h5"
            ts.save(path)
            loaded = TrainingSet.load(path)

            assert len(loaded) == 40
            torch.testing.assert_close(ts.x, loaded.x, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(ts.y_plus, loaded.y_plus, atol=1e-6, rtol=1e-6)


class TestSobolSampling:

    def test_correct_count(self):
        samples = sobol_sample(
            {"mass_ratio": (0.1, 1.0)}, n_samples=50, seed=42
        )
        assert len(samples["mass_ratio"]) == 50

    def test_within_bounds(self):
        bounds = {"mass_ratio": (0.1, 1.0), "spin": (-0.5, 0.5)}
        samples = sobol_sample(bounds, n_samples=100, seed=42)
        assert samples["mass_ratio"].min() >= 0.1
        assert samples["mass_ratio"].max() <= 1.0
        assert samples["spin"].min() >= -0.5
        assert samples["spin"].max() <= 0.5

    def test_multidimensional(self):
        bounds = {"a": (0, 1), "b": (0, 1), "c": (0, 1)}
        samples = sobol_sample(bounds, n_samples=64, seed=42)
        assert len(samples) == 3
        for v in samples.values():
            assert len(v) == 64


class TestLatinHypercubeSampling:

    def test_correct_count(self):
        samples = latin_hypercube_sample(
            {"mass_ratio": (0.1, 1.0)}, n_samples=50, seed=42
        )
        assert len(samples["mass_ratio"]) == 50

    def test_within_bounds(self):
        bounds = {"mass_ratio": (0.1, 1.0)}
        samples = latin_hypercube_sample(bounds, n_samples=100, seed=42)
        assert samples["mass_ratio"].min() >= 0.1
        assert samples["mass_ratio"].max() <= 1.0


class TestJitteredGridSampling:

    def test_correct_count(self):
        samples = jittered_grid_sample(
            {"mass_ratio": (0.1, 1.0)}, n_samples=50, seed=42
        )
        assert len(samples["mass_ratio"]) == 50

    def test_within_bounds(self):
        bounds = {"mass_ratio": (0.1, 1.0)}
        samples = jittered_grid_sample(bounds, n_samples=100, seed=42)
        assert samples["mass_ratio"].min() >= 0.1
        assert samples["mass_ratio"].max() <= 1.0

    def test_stratified_one_per_cell(self):
        # Each equal-width cell should contain exactly one sample — no
        # clumps, no voids (the whole point vs pure random).
        n = 40
        lo, hi = 0.1, 0.97
        samples = np.sort(
            jittered_grid_sample({"q": (lo, hi)}, n_samples=n, seed=7)["q"]
        )
        edges = np.linspace(lo, hi, n + 1)
        for i, s in enumerate(samples):
            assert edges[i] <= s <= edges[i + 1]

    def test_reproducible(self):
        a = jittered_grid_sample({"q": (0.1, 1.0)}, n_samples=30, seed=123)["q"]
        b = jittered_grid_sample({"q": (0.1, 1.0)}, n_samples=30, seed=123)["q"]
        np.testing.assert_array_equal(a, b)

    def test_multidimensional(self):
        bounds = {"a": (0, 1), "b": (0, 1), "c": (0, 1)}
        samples = jittered_grid_sample(bounds, n_samples=32, seed=42)
        assert len(samples) == 3
        for v in samples.values():
            assert len(v) == 32


class TestScatteredGeneration:
    """generate_training_data_scattered — sampling/assembly logic, exercised
    with a lightweight fake approximant (no lalsuite needed)."""

    def _settings(self, **overrides):
        s = {
            "approximant": "IMRPhenomD",  # replaced by the monkeypatch below
            "total_mass": 60.0,
            "distance": 100.0,
            "q_bounds": [0.1, 0.97],
            "n_mass_ratios": 12,
            "n_samples": 8,
            "q_sampling": "sobol",
            "seed": 1234,
            "warping": {"type": "chirp", "alpha": 0.625},
        }
        s.update(overrides)
        return s

    def _patch_approx(self, monkeypatch):
        from heron.models.testing import SineGaussianWaveform
        import heron.train as train_mod
        monkeypatch.setattr(
            train_mod, "_get_approximant", lambda name: SineGaussianWaveform()
        )

    def test_distinct_q_count_and_size(self, monkeypatch):
        self._patch_approx(monkeypatch)
        from heron.train import generate_training_data_scattered

        ts = generate_training_data_scattered(self._settings())
        q = ts.x[:, 0].numpy()
        assert len(ts) == 12 * 8
        assert len(np.unique(q)) == 12  # 12 DISTINCT q's — not a tensor grid
        assert q.min() >= 0.1 and q.max() <= 0.97
        assert ts.metadata["source"] == "scattered"
        assert ts.metadata["n_mass_ratios"] == 12

    def test_more_distinct_q_than_grid(self, monkeypatch):
        # The whole motivation: many more distinct q's than the 30-node grid.
        self._patch_approx(monkeypatch)
        from heron.train import generate_training_data_scattered

        ts = generate_training_data_scattered(
            self._settings(n_mass_ratios=90, n_samples=4)
        )
        assert len(np.unique(ts.x[:, 0].numpy())) == 90

    def test_reproducible_with_seed(self, monkeypatch):
        self._patch_approx(monkeypatch)
        from heron.train import generate_training_data_scattered

        a = generate_training_data_scattered(self._settings())
        b = generate_training_data_scattered(self._settings())
        torch.testing.assert_close(a.x, b.x)

    def test_sampling_methods(self, monkeypatch):
        self._patch_approx(monkeypatch)
        from heron.train import generate_training_data_scattered

        for method in ("sobol", "lhs", "jittered"):
            ts = generate_training_data_scattered(
                self._settings(q_sampling=method)
            )
            assert len(np.unique(ts.x[:, 0].numpy())) == 12

    def test_requires_q_bounds(self, monkeypatch):
        self._patch_approx(monkeypatch)
        from heron.train import generate_training_data_scattered

        s = self._settings()
        del s["q_bounds"]
        with pytest.raises(ValueError, match="q_bounds"):
            generate_training_data_scattered(s)

    def test_unknown_sampling_raises(self, monkeypatch):
        self._patch_approx(monkeypatch)
        from heron.train import generate_training_data_scattered

        with pytest.raises(ValueError, match="Unknown q_sampling"):
            generate_training_data_scattered(self._settings(q_sampling="random"))
