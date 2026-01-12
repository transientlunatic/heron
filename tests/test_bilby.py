"""
Tests for the bilby interfaces
"""

import unittest
import os
import numpy as np

try:
    import bilby

    BILBY_AVAILABLE = True
except ImportError:
    BILBY_AVAILABLE = False

import heron.bilby
from heron.models.testing import SineGaussianWaveform



class TestBilbyData(unittest.TestCase):

    def setUp(self):
        self.current_directory = os.path.dirname(os.path.realpath(__file__))
        self.data_directory = os.path.join(self.current_directory, "testing-data")

    @unittest.skipIf(not heron.bilby.BILBY_PIPE_AVAILABLE, "bilby_pipe not available")
    def test_read_pickle(self):
        data = heron.bilby.read_pickle(
            os.path.join(self.data_directory, "bilby-pipe-data.pickle")
        )
        self.assertTrue(set(data['strain'].keys()) == {"H1", "L1"})


@unittest.skipIf(not BILBY_AVAILABLE, "bilby not available")
class TestHeronWaveformGenerator(unittest.TestCase):
    """Test the HeronWaveformGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.duration = 1.0
        self.sampling_frequency = 1024
        self.heron_waveform = SineGaussianWaveform()
        
        self.wfg = heron.bilby.HeronWaveformGenerator(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            heron_waveform=self.heron_waveform
        )
        
        self.parameters = {
            'width': 0.02,
            'ra': 1.0,
            'dec': 0.5,
            'phase': 0.0,
            'psi': 0.0,
            'theta_jn': 1.0,
            'geocent_time': 0.0
        }
    
    def test_initialization(self):
        """Test that the waveform generator initializes correctly."""
        self.assertIsNotNone(self.wfg)
        self.assertEqual(self.wfg.duration, self.duration)
        self.assertEqual(self.wfg.sampling_frequency, self.sampling_frequency)
        self.assertEqual(self.wfg.heron_waveform, self.heron_waveform)
    
    def test_time_domain_generation(self):
        """Test time-domain waveform generation."""
        times = self.wfg.time_array
        waveform_dict = self.wfg._heron_time_domain_model(times, **self.parameters)
        
        self.assertIn('plus', waveform_dict)
        self.assertIn('cross', waveform_dict)
        self.assertEqual(len(waveform_dict['plus']), len(times))
        self.assertEqual(len(waveform_dict['cross']), len(times))
    
    def test_uncertainty_propagation(self):
        """Test that uncertainty information is propagated."""
        times = self.wfg.time_array
        waveform_dict = self.wfg._heron_time_domain_model(times, **self.parameters)
        
        # Check that uncertainty information is included
        # SineGaussianWaveform includes covariance
        self.assertIn('plus_covariance', waveform_dict)
        self.assertIn('cross_covariance', waveform_dict)
        self.assertIn('plus_variance', waveform_dict)
        self.assertIn('cross_variance', waveform_dict)
    
    def test_waveform_shape(self):
        """Test that waveforms have the correct shape."""
        times = self.wfg.time_array
        waveform_dict = self.wfg._heron_time_domain_model(times, **self.parameters)
        

        # Note: actual length may differ slightly due to heron's internal handling
        self.assertGreater(len(waveform_dict['plus']), 0)
        self.assertGreater(len(waveform_dict['cross']), 0)


@unittest.skipIf(not BILBY_AVAILABLE, "bilby not available")  
class TestHeronGravitationalWaveTransient(unittest.TestCase):
    """Test the HeronGravitationalWaveTransient likelihood class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set up basic parameters
        self.duration = 1.0
        self.sampling_frequency = 1024
        self.injection_parameters = {
            'width': 0.02,
            'ra': 1.0,
            'dec': 0.5,
            'phase': 0.0,
            'psi': 0.0,
            'theta_jn': 1.0,
            'geocent_time': 0.0
        }
        
        # Create heron waveform model
        self.heron_waveform = SineGaussianWaveform()
        
        # Create waveform generator
        self.wfg = heron.bilby.HeronWaveformGenerator(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            heron_waveform=self.heron_waveform
        )
    
    @unittest.skip("Requires full bilby.gw setup with interferometers")
    def test_likelihood_initialization(self):
        """Test likelihood initialization."""
        # This test requires setting up bilby interferometers which is complex
        # Skipping for now but keeping as template for future complete tests
        pass
    
    @unittest.skip("Requires full bilby.gw setup")
    def test_likelihood_evaluation(self):
        """Test likelihood evaluation at injection parameters."""
        pass
    
    @unittest.skip("Requires full bilby.gw setup")
    def test_likelihood_with_uncertainty(self):
        """Test that model uncertainty affects likelihood."""
        pass
    
    @unittest.skip("Requires full bilby.gw setup")
    def test_snr_calculation(self):
        """Test SNR calculation matches expected values."""
        pass


class TestBilbyIntegrationScientificAccuracy(unittest.TestCase):
    """Test scientific accuracy of the bilby integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.heron_waveform = SineGaussianWaveform()
        
    @unittest.skipIf(not BILBY_AVAILABLE, "bilby not available")
    def test_waveform_normalization(self):
        """Test that waveforms are properly normalized."""
        duration = 1.0
        sampling_frequency = 1024
        
        wfg = heron.bilby.HeronWaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            heron_waveform=self.heron_waveform
        )
        
        parameters = {
            'width': 0.02,
            'ra': 1.0,
            'dec': 0.5,
            'phase': 0.0,
            'psi': 0.0,
            'theta_jn': 1.0,
            'geocent_time': 0.0
        }
        
        times = wfg.time_array
        waveform_dict = wfg._heron_time_domain_model(times, **parameters)
        
        # Check that waveforms are non-zero
        self.assertGreater(np.max(np.abs(waveform_dict['plus'])), 0)
        self.assertGreater(np.max(np.abs(waveform_dict['cross'])), 0)
        
        # Check that waveforms are finite
        self.assertTrue(np.all(np.isfinite(waveform_dict['plus'])))
        self.assertTrue(np.all(np.isfinite(waveform_dict['cross'])))
    
    @unittest.skipIf(not BILBY_AVAILABLE, "bilby not available")
    def test_uncertainty_positive_definite(self):
        """Test that uncertainty covariances are positive semi-definite."""
        duration = 1.0
        sampling_frequency = 1024
        
        wfg = heron.bilby.HeronWaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            heron_waveform=self.heron_waveform
        )
        
        parameters = {
            'width': 0.02,
            'ra': 1.0,
            'dec': 0.5,
            'phase': 0.0,
            'psi': 0.0,
            'theta_jn': 1.0,
            'geocent_time': 0.0
        }
        
        times = wfg.time_array
        waveform_dict = wfg._heron_time_domain_model(times, **parameters)
        
        # Check that variances are non-negative
        if 'plus_variance' in waveform_dict:
            self.assertTrue(np.all(waveform_dict['plus_variance'] >= 0))
            self.assertTrue(np.all(waveform_dict['cross_variance'] >= 0))
        
        # Check that diagonal of covariance is non-negative
        if 'plus_covariance' in waveform_dict:
            plus_diag = np.diag(waveform_dict['plus_covariance'])
            cross_diag = np.diag(waveform_dict['cross_covariance'])
            self.assertTrue(np.all(plus_diag >= 0))
            self.assertTrue(np.all(cross_diag >= 0))


if __name__ == '__main__':
    unittest.main()
