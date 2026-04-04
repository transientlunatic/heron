"""Tests for heron.models.warping — time coordinate transformations."""

import torch
import numpy as np
import pytest

from heron.models.warping import (
    SimpleWarping,
    ChirpTimeWarping,
    get_warping,
)


class TestSimpleWarping:

    def test_roundtrip(self):
        w = SimpleWarping(scale=2.0)
        t = torch.linspace(-1.0, 1.0, 100)
        recovered = w.unwarp(w.warp(t))
        torch.testing.assert_close(recovered, t, atol=1e-6, rtol=1e-6)

    def test_positive_times_unchanged(self):
        w = SimpleWarping(scale=3.0)
        t = torch.tensor([0.0, 0.1, 0.5, 1.0])
        torch.testing.assert_close(w.warp(t), t)

    def test_negative_times_compressed(self):
        w = SimpleWarping(scale=2.0)
        t = torch.tensor([-1.0, -0.5])
        warped = w.warp(t)
        assert (warped > t).all()  # compressed toward zero


class TestChirpTimeWarping:

    def test_roundtrip(self):
        w = ChirpTimeWarping(alpha=0.625, t_ref=0.1)
        t = torch.linspace(-1.0, 1.0, 200)
        recovered = w.unwarp(w.warp(t))
        torch.testing.assert_close(recovered, t, atol=1e-5, rtol=1e-5)

    def test_preserves_ordering(self):
        w = ChirpTimeWarping(alpha=0.625)
        t = torch.linspace(-2.0, 0.5, 100)
        warped = w.warp(t)
        # Warped times should be monotonically increasing
        assert (warped[1:] >= warped[:-1]).all()

    def test_default_alpha(self):
        w = ChirpTimeWarping()
        assert w.alpha == 0.375  # Newtonian scaling


class TestGetWarping:

    def test_simple(self):
        w = get_warping("simple", scale=3.0)
        assert isinstance(w, SimpleWarping)

    def test_chirp(self):
        w = get_warping("chirp", alpha=0.5, t_ref=0.2)
        assert isinstance(w, ChirpTimeWarping)
        assert w.alpha == 0.5
