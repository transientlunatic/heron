"""Tests for heron.models.lalsimulation — requires lalsuite."""

import unittest
import pytest

try:
    import lalsimulation
    from heron.models.lalsimulation import (
        LALSimulationApproximant,
        IMRPhenomD,
        IMRPhenomPv2,
        SEOBNRv3,
    )
    HAS_LAL = True
except ImportError:
    HAS_LAL = False


@pytest.mark.skipif(not HAS_LAL, reason="lalsuite not installed")
class TestLALSimulationApproximant(unittest.TestCase):

    def setUp(self):
        self.approx = LALSimulationApproximant()

    def test_initialization(self):
        self.assertIn("m1", self.approx._args)
        self.assertIn("m2", self.approx._args)
        self.assertIn("distance", self.approx._args)

    def test_allowed_parameters(self):
        self.assertIsInstance(self.approx.allowed_parameters, list)
        self.assertIn("m1", self.approx.allowed_parameters)


@pytest.mark.skipif(not HAS_LAL, reason="lalsuite not installed")
class TestIMRPhenomPv2(unittest.TestCase):

    def setUp(self):
        self.approx = IMRPhenomPv2()

    def test_initialization(self):
        self.assertIsNotNone(self.approx._args["approximant"])

    def test_approximant_type(self):
        expected = lalsimulation.GetApproximantFromString("IMRPhenomPv2")
        self.assertEqual(self.approx._args["approximant"], expected)


@pytest.mark.skipif(not HAS_LAL, reason="lalsuite not installed")
class TestSEOBNRv3(unittest.TestCase):

    def setUp(self):
        self.approx = SEOBNRv3()

    def test_initialization(self):
        self.assertIsNotNone(self.approx._args["approximant"])


@pytest.mark.skipif(not HAS_LAL, reason="lalsuite not installed")
class TestIMRPhenomD(unittest.TestCase):

    def setUp(self):
        self.approx = IMRPhenomD()

    def test_initialization(self):
        self.assertIsNotNone(self.approx._args["approximant"])

    def test_approximant_type(self):
        expected = lalsimulation.GetApproximantFromString("IMRPhenomD")
        self.assertEqual(self.approx._args["approximant"], expected)
