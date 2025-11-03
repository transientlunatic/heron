"""
These tests are designed to test the GPyTorch-based models in heron.
"""

import unittest

import astropy.units as u
import torch
import numpy as np
import gpytorch

from heron.training.makedata import make_manifold, make_optimal_manifold
from heron.models.gpytorch import HeronNonSpinningApproximant
from heron.models.lalsimulation import SEOBNRv3, IMRPhenomPv2
from heron.models.lalnoise import AdvancedLIGO
from heron.filters import Overlap
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from astropy import units as u
import lal
import lalsimulation

from heron.models.lalsimulation import (
    LALSimulationApproximant,
    IMRPhenomPv2,
    SEOBNRv3,
    IMRPhenomPv2_FakeUncertainty
)
from heron.types import Waveform, WaveformDict

from . import _GenericWaveform

class test_IMRPhenomPv2(_GenericWaveform):

    @classmethod
    def setUpClass(cls):
        # initialize likelihood and model
        cls.model = IMRPhenomPv2()


class test_SEOBNRv3(_GenericWaveform):

    @classmethod
    def setUpClass(cls):
        # initialize likelihood and model
        cls.model = SEOBNRv3()

class TestLALSimulationApproximant(unittest.TestCase):
    """Test the base LALSimulationApproximant class."""

    def setUp(self):
        self.approx = LALSimulationApproximant()

    def test_initialization(self):
        """Test that approximant initializes with correct default arguments."""
        self.assertIn('m1', self.approx._args)
        self.assertIn('m2', self.approx._args)
        self.assertIn('distance', self.approx._args)
        self.assertEqual(self.approx._args['S1x'], 0.0)
        self.assertEqual(self.approx._args['S1y'], 0.0)
        self.assertEqual(self.approx._args['S1z'], 0.0)

    def test_allowed_parameters(self):
        """Test that allowed_parameters list is populated."""
        self.assertIsInstance(self.approx.allowed_parameters, list)
        self.assertGreater(len(self.approx.allowed_parameters), 0)
        self.assertIn('m1', self.approx.allowed_parameters)
        self.assertIn('m2', self.approx.allowed_parameters)

    def test_supported_conversions(self):
        """Test that supported conversions are defined."""
        expected = {'mass_ratio', 'total_mass', 'luminosity_distance'}
        self.assertEqual(self.approx.supported_converstions, expected)

    def test_convert_units_with_quantity(self):
        """Test unit conversion with astropy quantities."""
        args = {
            'm1': 30.0 * u.solMass,
            'm2': 25.0 * u.solMass,
            'distance': 100.0 * u.Mpc
        }
        converted = self.approx._convert_units(args)
        
        # Check that values are converted to SI units
        self.assertIsInstance(converted['m1'], float)
        self.assertIsInstance(converted['m2'], float)
        self.assertIsInstance(converted['distance'], float)
        
        # Mass should be in kg
        self.assertAlmostEqual(converted['m1'], (30.0 * u.solMass).to(u.kg).value, places=5)
        self.assertAlmostEqual(converted['m2'], (25.0 * u.solMass).to(u.kg).value, places=5)

    def test_convert_units_without_quantity(self):
        """Test unit conversion with plain numbers (assumes default units)."""
        args = {
            'm1': 30.0,  # Assumed to be solar masses
            'm2': 25.0,
            'distance': 100.0  # Assumed to be Mpc
        }
        converted = self.approx._convert_units(args)
        
        # Should convert assuming default units
        self.assertIsInstance(converted['m1'], float)
        self.assertIsInstance(converted['m2'], float)
        self.assertGreater(converted['m1'], 1e30)  # Should be in kg

    def test_convert_units_frequency_parameters(self):
        """Test conversion of frequency parameters."""
        args = {
            'f_min': 20.0 * u.Hertz,
            'f_ref': 20.0 * u.Hertz
        }
        converted = self.approx._convert_units(args)
        
        self.assertEqual(converted['f_min'], 20.0)
        self.assertEqual(converted['f_ref'], 20.0)

    def test_args_property(self):
        """Test the args property returns properly formatted arguments."""
        self.approx._args['m1'] = 30.0 * u.solMass
        self.approx._args['m2'] = 25.0 * u.solMass
        
        args = self.approx.args
        
        # Should be converted to base units
        self.assertIn('m1', args)
        self.assertIn('m2', args)
        self.assertIsInstance(args['m1'], float)

    def test_args_removes_extrinsic_parameters(self):
        """Test that extrinsic parameters are removed from args."""
        self.approx._args.update({
            'ra': 1.0,
            'dec': 0.5,
            'phase': 0.0,
            'psi': 0.5,
            'theta_jn': 1.0
        })
        
        args = self.approx.args
        
        # Extrinsic parameters should be removed
        self.assertNotIn('ra', args)
        self.assertNotIn('dec', args)
        self.assertNotIn('phase', args)
        self.assertNotIn('psi', args)
        self.assertNotIn('theta_jn', args)


class TestIMRPhenomPv2(unittest.TestCase):
    """Test the IMRPhenomPv2 approximant."""

    def setUp(self):
        self.approx = IMRPhenomPv2()

    def test_initialization(self):
        """Test that IMRPhenomPv2 initializes correctly."""
        self.assertIsNotNone(self.approx._args['approximant'])

    def test_approximant_type(self):
        """Test that the correct approximant is set."""
        expected_approx = lalsimulation.GetApproximantFromString("IMRPhenomPv2")
        self.assertEqual(self.approx._args['approximant'], expected_approx)

    @patch('lalsimulation.SimInspiralChooseTDWaveform')
    def test_time_domain_generation(self, mock_sim):
        """Test time domain waveform generation."""
        # Mock the LAL waveform output
        mock_hp = Mock()
        mock_hp.data.data = np.random.randn(1000)
        mock_hp.deltaT = 1.0 / 4096.0
        mock_hp.epoch = 0
        
        mock_hx = Mock()
        mock_hx.data.data = np.random.randn(1000)
        mock_hx.deltaT = 1.0 / 4096.0
        mock_hx.epoch = 0
        
        mock_sim.return_value = (mock_hp, mock_hx)
        
        parameters = {
            'm1': 30.0,
            'm2': 25.0,
            'distance': 100.0
        }
        
        waveform = self.approx.time_domain(parameters)
        
        # Check that waveform is a WaveformDict
        self.assertIsInstance(waveform, WaveformDict)
        self.assertIn('plus', waveform.waveforms)
        self.assertIn('cross', waveform.waveforms)
        
        # Check that LALSimulation was called
        mock_sim.assert_called_once()

    @patch('lalsimulation.SimInspiralChooseTDWaveform')
    def test_time_domain_with_times(self, mock_sim):
        """Test time domain generation with specified times."""
        # Mock the LAL waveform output
        mock_hp = Mock()
        mock_hp.data.data = np.sin(np.linspace(0, 10*np.pi, 1000))
        mock_hp.deltaT = 1.0 / 4096.0
        mock_hp.epoch.ns.return_value = 0
        
        mock_hx = Mock()
        mock_hx.data.data = np.cos(np.linspace(0, 10*np.pi, 1000))
        mock_hx.deltaT = 1.0 / 4096.0
        mock_hx.epoch.ns.return_value = 0
        
        mock_sim.return_value = (mock_hp, mock_hx)
        
        parameters = {
            'm1': 30.0,
            'm2': 25.0,
            'distance': 100.0
        }
        
        times = np.linspace(0, 1, 500)
        waveform = self.approx.time_domain(parameters, times=times)
        
        # Check that output times match input times
        self.assertEqual(len(waveform['plus'].times), len(times))
        self.assertEqual(len(waveform['cross'].times), len(times))

    @patch('lalsimulation.SimInspiralChooseTDWaveform')
    def test_time_domain_with_epoch(self, mock_sim):
        """Test time domain generation with GPS time/epoch."""
        mock_hp = Mock()
        mock_hp.data.data = np.random.randn(1000)
        mock_hp.deltaT = 1.0 / 4096.0
        mock_hp.epoch = 0
        
        mock_hx = Mock()
        mock_hx.data.data = np.random.randn(1000)
        mock_hx.deltaT = 1.0 / 4096.0
        mock_hx.epoch = 0
        
        mock_sim.return_value = (mock_hp, mock_hx)
        
        parameters = {
            'm1': 30.0,
            'm2': 25.0,
            'distance': 100.0,
            'gpstime': 1000000000.0
        }
        
        waveform = self.approx.time_domain(parameters)
        
        self.assertIsInstance(waveform, WaveformDict)

    @patch('lalsimulation.SimInspiralChooseTDWaveform')
    def test_time_domain_caching(self, mock_sim):
        """Test that waveforms are cached when parameters don't change."""
        mock_hp = Mock()
        mock_hp.data.data = np.random.randn(1000)
        mock_hp.deltaT = 1.0 / 4096.0
        mock_hp.epoch = 0
        
        mock_hx = Mock()
        mock_hx.data.data = np.random.randn(1000)
        mock_hx.deltaT = 1.0 / 4096.0
        mock_hx.epoch = 0
        
        mock_sim.return_value = (mock_hp, mock_hx)
        
        parameters = {
            'm1': 30.0,
            'm2': 25.0,
            'distance': 100.0
        }
        
        # Generate waveform twice with same parameters
        waveform1 = self.approx.time_domain(parameters)
        waveform2 = self.approx.time_domain(parameters)
        
        # Should only call LALSimulation once due to caching
        self.assertEqual(mock_sim.call_count, 1)
        self.assertIs(waveform1, waveform2)

    @patch('lalsimulation.SimInspiralChooseTDWaveform')
    def test_time_domain_cache_invalidation(self, mock_sim):
        """Test that cache is invalidated when parameters change."""
        mock_hp = Mock()
        mock_hp.data.data = np.random.randn(1000)
        mock_hp.deltaT = 1.0 / 4096.0
        mock_hp.epoch = 0
        
        mock_hx = Mock()
        mock_hx.data.data = np.random.randn(1000)
        mock_hx.deltaT = 1.0 / 4096.0
        mock_hx.epoch = 0
        
        mock_sim.return_value = (mock_hp, mock_hx)
        
        parameters1 = {
            'm1': 30.0,
            'm2': 25.0,
            'distance': 100.0
        }
        
        parameters2 = {
            'm1': 35.0,  # Different mass
            'm2': 25.0,
            'distance': 100.0
        }
        
        waveform1 = self.approx.time_domain(parameters1)
        waveform2 = self.approx.time_domain(parameters2)
        
        # Should call LALSimulation twice with different parameters
        self.assertEqual(mock_sim.call_count, 2)


class TestSEOBNRv3(unittest.TestCase):
    """Test the SEOBNRv3 approximant."""

    def setUp(self):
        self.approx = SEOBNRv3()

    def test_initialization(self):
        """Test that SEOBNRv3 initializes correctly."""
        self.assertIsNotNone(self.approx._args['approximant'])

    def test_approximant_type(self):
        """Test that the correct approximant is set."""
        expected_approx = lalsimulation.GetApproximantFromString("SEOBNRv3")
        self.assertEqual(self.approx._args['approximant'], expected_approx)


class TestIMRPhenomPv2FakeUncertainty(unittest.TestCase):
    """Test the IMRPhenomPv2_FakeUncertainty approximant."""

    def setUp(self):
        self.covariance = 1e-24
        self.approx = IMRPhenomPv2_FakeUncertainty(covariance=self.covariance)

    def test_initialization(self):
        """Test initialization with covariance parameter."""
        self.assertEqual(self.approx.covariance, self.covariance)
        self.assertIsNotNone(self.approx._args['approximant'])

    def test_default_covariance(self):
        """Test default covariance value."""
        approx = IMRPhenomPv2_FakeUncertainty()
        self.assertEqual(approx.covariance, 1e-24)

    @patch('lalsimulation.SimInspiralChooseTDWaveform')
    def test_time_domain_adds_covariance(self, mock_sim):
        """Test that covariance matrix is added to waveforms."""
        N = 100
        mock_hp = Mock()
        mock_hp.data.data = np.random.randn(N)
        mock_hp.deltaT = 1.0 / 4096.0
        mock_hp.epoch.ns.return_value = 0
        
        mock_hx = Mock()
        mock_hx.data.data = np.random.randn(N)
        mock_hx.deltaT = 1.0 / 4096.0
        mock_hx.epoch.ns.return_value = 0
        
        mock_sim.return_value = (mock_hp, mock_hx)
        
        parameters = {
            'm1': 30.0,
            'm2': 25.0,
            'distance': 100.0
        }
        
        times = np.linspace(0, 1, N)
        waveform = self.approx.time_domain(parameters, times=times)
        
        # Check that covariance is added
        self.assertTrue(hasattr(waveform['plus'], 'covariance'))
        self.assertTrue(hasattr(waveform['cross'], 'covariance'))
        
        # Check covariance shape
        self.assertEqual(waveform['plus'].covariance.shape, (N, N))
        self.assertEqual(waveform['cross'].covariance.shape, (N, N))

    @patch('lalsimulation.SimInspiralChooseTDWaveform')
    def test_covariance_matrix_properties(self, mock_sim):
        """Test properties of the covariance matrix."""
        N = 50
        mock_hp = Mock()
        mock_hp.data.data = np.random.randn(N)
        mock_hp.deltaT = 1.0 / 4096.0
        mock_hp.epoch.ns.return_value = 0
        
        mock_hx = Mock()
        mock_hx.data.data = np.random.randn(N)
        mock_hx.deltaT = 1.0 / 4096.0
        mock_hx.epoch.ns.return_value = 0
        
        mock_sim.return_value = (mock_hp, mock_hx)
        
        parameters = {
            'm1': 30.0,
            'm2': 25.0,
            'distance': 100.0
        }
        
        times = np.linspace(0, 1, N)
        waveform = self.approx.time_domain(parameters, times=times)
        
        cov = waveform['plus'].covariance
        
        # Covariance should be symmetric
        self.assertTrue(np.allclose(cov, cov.T))
        
        # Covariance should be positive semi-definite
        eigenvalues = np.linalg.eigvals(cov)
        self.assertTrue(np.all(eigenvalues >= -1e-10))  # Allow small numerical errors
        
        # Diagonal should be positive
        self.assertTrue(np.all(np.diag(cov) > 0))

    @patch('lalsimulation.SimInspiralChooseTDWaveform')
    def test_covariance_scaling(self, mock_sim):
        """Test that covariance scales with the covariance parameter."""
        N = 50
        mock_hp = Mock()
        mock_hp.data.data = np.random.randn(N)
        mock_hp.deltaT = 1.0 / 4096.0
        mock_hp.epoch.ns.return_value = 0
        
        mock_hx = Mock()
        mock_hx.data.data = np.random.randn(N)
        mock_hx.deltaT = 1.0 / 4096.0
        mock_hx.epoch.ns.return_value = 0
        
        mock_sim.return_value = (mock_hp, mock_hx)
        
        parameters = {
            'm1': 30.0,
            'm2': 25.0,
            'distance': 100.0
        }
        
        times = np.linspace(0, 1, N)
        
        # Test with two different covariance values
        approx1 = IMRPhenomPv2_FakeUncertainty(covariance=1e-24)
        approx2 = IMRPhenomPv2_FakeUncertainty(covariance=1e-23)
        
        waveform1 = approx1.time_domain(parameters, times=times)
        waveform2 = approx2.time_domain(parameters, times=times)
        
        # Larger covariance parameter should give larger covariance matrix
        ratio = np.max(waveform2['plus'].covariance) / np.max(waveform1['plus'].covariance)
        self.assertAlmostEqual(ratio, 10.0, places=1)


class TestWaveformValidation(unittest.TestCase):
    """Integration tests for waveform generation and validation."""

    @patch('lalsimulation.SimInspiralChooseTDWaveform')
    def test_waveform_length_consistency(self, mock_sim):
        """Test that plus and cross polarizations have same length."""
        N = 1000
        mock_hp = Mock()
        mock_hp.data.data = np.random.randn(N)
        mock_hp.deltaT = 1.0 / 4096.0
        mock_hp.epoch = 0
        
        mock_hx = Mock()
        mock_hx.data.data = np.random.randn(N)
        mock_hx.deltaT = 1.0 / 4096.0
        mock_hx.epoch = 0
        
        mock_sim.return_value = (mock_hp, mock_hx)
        
        approx = IMRPhenomPv2()
        parameters = {'m1': 30.0, 'm2': 25.0, 'distance': 100.0}
        
        waveform = approx.time_domain(parameters)
        
        self.assertEqual(len(waveform['plus'].data), len(waveform['cross'].data))

    @patch('lalsimulation.SimInspiralChooseTDWaveform')
    def test_waveform_parameters_stored(self, mock_sim):
        """Test that parameters are stored in WaveformDict."""
        mock_hp = Mock()
        mock_hp.data.data = np.random.randn(1000)
        mock_hp.deltaT = 1.0 / 4096.0
        mock_hp.epoch = 0
        
        mock_hx = Mock()
        mock_hx.data.data = np.random.randn(1000)
        mock_hx.deltaT = 1.0 / 4096.0
        mock_hx.epoch = 0
        
        mock_sim.return_value = (mock_hp, mock_hx)
        
        approx = IMRPhenomPv2()
        parameters = {
            'm1': 30.0,
            'm2': 25.0,
            'distance': 100.0,
            'inclination': 0.5
        }
        
        waveform = approx.time_domain(parameters)
        
        self.assertEqual(waveform.parameters['m1'], 30.0)
        self.assertEqual(waveform.parameters['m2'], 25.0)
        self.assertEqual(waveform.parameters['inclination'], 0.5)


if __name__ == '__main__':
    unittest.main()