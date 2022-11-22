"""
CUDA Waveform Tests
-------------------

This test suite contains tests to ensure that the output of CUDA-based waveform models
both complies with the API and also behave as expected.

These tests do not attempt to test the scientific accuracy of the waveforms.
"""

import unittest
from unittest import mock

from heron.models import torchbased

import torch

models = {"HeronCUDA": torchbased.HeronCUDA}

class TestCUDAWaveform(unittest.TestCase):

    def test_load_default_model(self):
        """Check that a default model and training data are loaded if nothing is specified."""
        for model_name, model in models.items():
            with self.subTest(model=model_name):
                generative_model = model()
    
    def test_prediction_mean(self):
        """Check that the model can produce a predictive waveform."""
        times = torch.linspace(-0.01, 0.05, 100)
        p = {"mass ratio": 0.9}
        for model_name, model in models.items():
            with self.subTest(model=model_name):
                generative_model = model(datafile="training_data.h5")
                prediction_mean, prediction_var, prediction_covar = generative_model._predict(p=p, times=times)

                # A mean waveform should be produced.
                self.assertTrue(isinstance(prediction_mean, torch.Tensor))
                # The length of the produced waveform should be the same as the input times
                self.assertEqual(len(prediction_mean), len(times))

    def test_prediction_variance(self):
        """Check that the model can produce a predictive waveform."""
        times = torch.linspace(-0.01, 0.05, 100)
        p = {"mass ratio": 0.9}
        for model_name, model in models.items():
            with self.subTest(model=model_name):
                generative_model = model(datafile="training_data.h5")
                prediction_mean, prediction_var, prediction_covar = generative_model._predict(p=p, times=times)

                # A mean waveform should be produced.
                self.assertTrue(isinstance(prediction_var, torch.Tensor))
                # The length of the produced waveform should be the same as the input times
                self.assertEqual(len(prediction_var), len(times))

    def test_prediction_covariance(self):
        """Check that the model can produce a predictive waveform."""
        times = torch.linspace(-0.01, 0.05, 100)
        p = {"mass ratio": 0.9}
        for model_name, model in models.items():
            with self.subTest(model=model_name):
                generative_model = model(datafile="training_data.h5")
                prediction_mean, prediction_var, prediction_covar = generative_model._predict(p=p, times=times)

                # A mean waveform should be produced.
                self.assertTrue(isinstance(prediction_covar, torch.Tensor))
                # The covariance should be two-dimensional
                self.assertEqual(prediction_covar.ndim, 2)
                # The length of the produced waveform should be the same as the input times
                self.assertEqual(len(prediction_covar[:,0]), len(times))

    def test_distribution(self):
        """Check that a distribution of predictive waveforms can be produced."""
        times = torch.linspace(-0.01, 0.05, 100)
        p = {"mass ratio": 0.9}
        for model_name, model in models.items():
            with self.subTest(model=model_name):
                generative_model = model(datafile="training_data.h5")

                waveforms = generative_model.distribution(times=times, p=p)
                # A list of waveforms is produced
                self.assertTrue(isinstance(waveforms, list))
                # Check that one of these waveforms looks sane
                self.assertEqual(len(waveforms[0]), len(times))

    def test_timedomain_interface(self):
        """check that the timedomain interface returns a waveform in the anticipated format."""
        times = torch.linspace(-0.01, 0.05, 100)
        p = {"mass ratio": 0.9}
        for model_name, model in models.items():
            with self.subTest(model=model_name):
                generative_model = model(datafile="training_data.h5")

                waveform = generative_model.time_domain_waveform(p=p, times=times)

                # We've not asked for a specific polarisation, so we should expect the output in a dict
                self.assertTrue(isinstance(waveform, dict))
                # Let's check that one polarisation is a sensible-looking waveform
                self.assertEqual(len(waveform['plus'].data), len(times))
                
    def test_freqdomain_interface(self):
        """check that the timedomain interface returns a waveform in the anticipated format."""
        times = torch.linspace(-0.01, 0.05, 100)
        p = {"mass ratio": 0.9}
        for model_name, model in models.items():
            with self.subTest(model=model_name):
                generative_model = model(datafile="training_data.h5")

                waveform = generative_model.frequency_domain_waveform(p=p, times=times)

                # We've not asked for a specific polarisation, so we should expect the output in a dict
                self.assertTrue(isinstance(waveform, dict))
                # Let's check that one polarisation is a sensible-looking waveform
                self.assertEqual(len(waveform['plus'].data), 1+len(times)/2)
                
