import unittest

from heron.models.torchbased import HeronCUDA, train
from heron.likelihood import InnerProduct, Likelihood, CUDALikelihood
from heron.models.testing import TestModel, CUDATestModel
from elk.waveform import Timeseries

from torch import tensor
import torch
import gpytorch

import numpy as np
import numpy.testing as npt
np.random.seed(90)


class TestCUDALikelihood(unittest.TestCase):
    device = torch.device("cuda")

    @classmethod
    def setUpClass(cls):
        cls.generator = HeronCUDA(datafile="notebooks/new-data-interface/test_file_2.h5", datalabel="IMR training", 
                          device=cls.device, 
                          noise=[0.000001, 0.0001],
                          lengths={"mass ratio": [0.001, 0.25],
                                   "time": [0.001, 20]}
        )
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.max_cg_iterations(1500):
            train(cls.generator, iterations=50)
        
    
    def setUp(self):

        data_length = 164
        fft_length = int(1+(data_length/2))
        
        generator = self.generator
        window = torch.blackman_window

        self.psd = torch.ones(fft_length, device=self.device)
        self.times = times = torch.linspace(0.05, 0.005, data_length)

        noise = 5e-20*torch.randn(data_length, device=self.device)
        signal = self.generator.time_domain_waveform(times=self.times, p={"mass ratio":0.7})
        detection = Timeseries(data=(torch.tensor(signal['plus'].data, device=self.device) + noise), times=signal['plus'].times)
        
        self.likelihood = CUDALikelihood(model=generator,
                                         detector_prefix="H1",
                                         window=window,
                                         data=detection,
                                         psd=self.psd,
                                         device=self.device
        )

    def test_inner_data_data(self):

        inner_product = InnerProduct(self.likelihood.psd.clone(),
                                     duration=self.likelihood.duration,
                                     f_min=self.likelihood.f_min, f_max=self.likelihood.f_max)
        ip = inner_product(self.likelihood.data, self.likelihood.data)
        self.assertGreater(inner_product(self.likelihood.data, self.likelihood.data), 0)
        self.assertFalse(torch.isnan(ip))

    def test_factor(self):
        inner_product = InnerProduct(self.likelihood.psd.clone(),
                                     duration=self.likelihood.duration,
                                     f_min=self.likelihood.f_min, f_max=self.likelihood.f_max)
        factor = torch.logdet(inner_product.metric.abs()[1:-1, 1:-1])
        self.assertGreater(factor, 0)

    def test_inner_waveform_waveform(self):

        inner_product = InnerProduct(self.likelihood.psd.clone(),
                                     duration=self.likelihood.duration,
                                     f_min=self.likelihood.f_min, f_max=self.likelihood.f_max)

        
        polarisations = self.likelihood._call_model({"mass ratio": 0.9})
        waveform_mean = polarisations['plus'].data
        self.assertGreater(inner_product(waveform_mean, waveform_mean), 0)

    def test_waveform_magnitude(self):
        polarisations = self.likelihood._call_model({"mass ratio": 0.9})
        self.assertGreater(polarisations['plus'].data[0].real, 0)
        self.assertLess(polarisations['plus'].data[0].real, 1e-20)
        
    def test_inner_waveform_data(self):

        inner_product = InnerProduct(self.likelihood.psd.clone(),
                                     duration=self.likelihood.duration,
                                     f_min=self.likelihood.f_min, f_max=self.likelihood.f_max)

        
        polarisations = self.likelihood._call_model({"mass ratio": 0.9})
        waveform_mean = polarisations['plus'].data

        #torch.testing.assert_close(waveform_mean, self.likelihood.data)
        
        product = inner_product(waveform_mean, self.likelihood.data)
        self.assertGreater(product, 0)

    def test_inner_data_data_multiple(self):
        """Check that the data,data inner product is always constant."""
        inner_product = InnerProduct(self.likelihood.psd.clone(),
                                     duration=self.likelihood.duration,
                                     f_min=self.likelihood.f_min, f_max=self.likelihood.f_max)

        parameters = np.linspace(0.1, 1.0, 100)
        products = []
        for parameter in parameters:
            polarisations = self.likelihood._call_model({"mass ratio": parameter})
            waveform_mean = polarisations['plus'].data
            products.append(inner_product(self.likelihood.data, self.likelihood.data))
        products = torch.tensor(products)
        self.assertTrue(torch.all(products==products[0]))

    def test_inner_waveform_data_multiple(self):
        """Check that the inner product between the data and the waveform has a maximum close to the true value."""
        inner_product = InnerProduct(self.likelihood.psd.clone(),
                                     duration=self.likelihood.duration,
                                     f_min=self.likelihood.f_min, f_max=self.likelihood.f_max)

        parameters = np.linspace(0., 1.0, 100)
        products = []

        #torch.testing.assert_close(self.likelihood._call_model({"a": 0.9})['plus'].data,
        #                           self.likelihood.data)
        
        for parameter in parameters:
            polarisations = self.likelihood._call_model({"mass ratio": parameter})
            waveform_mean = polarisations['plus'].data
            products.append(inner_product(waveform_mean, self.likelihood.data)-inner_product(waveform_mean, waveform_mean))
        self.assertLess(parameters[torch.argmax(torch.tensor(products))] - 0.9, 0.05)


    def test_inner_waveform_waveform_multiple(self):
        """Check that the waveform,waveform inner product makes sense."""
        inner_product = InnerProduct(self.likelihood.psd.clone(),
                                     duration=self.likelihood.duration,
                                     f_min=self.likelihood.f_min, f_max=self.likelihood.f_max)

        parameters = np.linspace(0.1, 1., 100)
        products = []

        polarisations = self.likelihood._call_model({"mass ratio": 0.9})
        waveform_mean = polarisations['plus'].data

        #torch.testing.assert_close(waveform_mean, self.likelihood.data)
        
        for parameter in parameters:
            polarisations = self.likelihood._call_model({"mass ratio": parameter})
            waveform_mean = polarisations['plus'].data
            products.append(inner_product(waveform_mean, waveform_mean))
        products = torch.tensor(products, device=self.device)
        self.assertTrue(torch.all(products<10*inner_product(self.likelihood.data, self.likelihood.data)))

        
    def test_simple(self):
        """Check that the correct inference is made for a simple signal."""
        parameters = np.linspace(0, 1., 100)
        likes = torch.tensor([self.likelihood({"mass ratio": parameter}, model_var=False) for parameter in parameters])
        self.assertLess(parameters[torch.argmax(likes)] - 0.7, 0.05)
