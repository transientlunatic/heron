"""Tests for heron.training — dataset, sampling, and active learning."""

import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

from heron.training.dataset import TrainingSet
from heron.training.sampling import sobol_sample, latin_hypercube_sample


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
