import unittest

from heron.likelihood import InnerProduct, Likelihood, CUDALikelihood
from heron.models.testing import TestModel, CUDATestModel
from elk.waveform import Timeseries

from torch import tensor
import torch

import numpy as np
import numpy.testing as npt
np.random.seed(90)


class TestInnerProduct(unittest.TestCase):
    device=torch.device("cuda")
    def setUp(self):
        self.vector_length = 180
        self.window = torch.blackman_window(self.vector_length)
        asd = torch.ones((int(self.vector_length/2)), device=self.device, dtype=torch.cfloat)
        self.psd = asd * asd

    def test_simple(self):
        ip = InnerProduct(psd=self.psd, duration=1)
        a = torch.ones(int(self.vector_length/2), device=self.device, dtype=torch.cfloat)
        b = torch.ones(int(self.vector_length/2), device=self.device, dtype=torch.cfloat)
        self.assertEqual((ip(a, b)), 360)

    def test_two_psds(self):
        """Test that everything works when a signal PSD is included."""
        ip = InnerProduct(psd=self.psd, signal_psd=self.psd, duration=1)
        a = torch.ones(int(self.vector_length/2), device=self.device, dtype=torch.cfloat)
        b = torch.ones(int(self.vector_length/2), device=self.device, dtype=torch.cfloat)
        self.assertEqual((ip(a, b)), 180)

class TestInnerProductCPU(TestInnerProduct):
    device = torch.device("cpu")


class TestLikelihood(unittest.TestCase):
    device = torch.device("cuda")

    def setUp(self):
        self.likelihood = Likelihood()

    def test_antenna_function(self):
        """Test that the antenna function returns the right sort of numbers."""
        antenna = self.likelihood._get_antenna_response("H1", ra=0, dec=45, psi=0, time=1000) 
        self.assertEqual(-0.5540661589835211, antenna.plus)
        self.assertEqual( 0.7995900298333787, antenna.cross)

class TestCUDALikelihood(unittest.TestCase):
    device = torch.device("cuda")

    def setUp(self):

        data_length = 164
        fft_length = int(1+(data_length/2))
        
        generator = CUDATestModel(device=self.device)
        window = torch.blackman_window

        self.psd = torch.ones(fft_length, device=self.device)
        self.times = times = torch.linspace(0, 5, data_length)
        self.test_data = generator.time_domain_waveform(dict(a=0.9),
                                                        times=times
        )
        #self.test_data_plus_noise = self.test_data + torch.randn(164, device=self.device)

        #data = Timeseries(self.test_data_plus_noise, times=times)
        
        self.likelihood = CUDALikelihood(model=generator,
                                         detector_prefix="H1",
                                         window=window,
                                         data=self.test_data,
                                         psd=self.psd,
                                         device=self.device
        )

    def test_inner_data_data(self):

        inner_product = InnerProduct(self.likelihood.psd.clone(),
                                     duration=self.likelihood.duration,
                                     f_min=self.likelihood.f_min, f_max=self.likelihood.f_max)
        self.assertGreater(inner_product(self.likelihood.data, self.likelihood.data), 0)

    def test_inner_waveform_waveform(self):

        inner_product = InnerProduct(self.likelihood.psd.clone(),
                                     duration=self.likelihood.duration,
                                     f_min=self.likelihood.f_min, f_max=self.likelihood.f_max)

        
        polarisations = self.likelihood._call_model({"a": 0.9})
        waveform_mean = polarisations['plus'].data
        self.assertGreater(inner_product(waveform_mean, waveform_mean), 0)

    def test_inner_waveform_data(self):

        inner_product = InnerProduct(self.likelihood.psd.clone(),
                                     duration=self.likelihood.duration,
                                     f_min=self.likelihood.f_min, f_max=self.likelihood.f_max)

        
        polarisations = self.likelihood._call_model({"a": 0.9})
        waveform_mean = polarisations['plus'].data

        #torch.testing.assert_close(waveform_mean, self.likelihood.data)
        
        product = inner_product(waveform_mean, self.likelihood.data)
        self.assertGreater(product, 0)

    def test_inner_data_data_multiple(self):
        """Check that the data,data inner product is always constant."""
        inner_product = InnerProduct(self.likelihood.psd.clone(),
                                     duration=self.likelihood.duration,
                                     f_min=self.likelihood.f_min, f_max=self.likelihood.f_max)

        parameters = torch.linspace(0.1, 1.5, 100)
        products = []
        for parameter in parameters:
            polarisations = self.likelihood._call_model({"a": parameter})
            waveform_mean = polarisations['plus'].data
            products.append(inner_product(self.likelihood.data, self.likelihood.data))
        products = torch.tensor(products)
        self.assertTrue(torch.all(products==products[0]))

    def test_inner_waveform_data_multiple(self):
        """Check that the inner product between the data and the waveform has a maximum close to the true value."""
        inner_product = InnerProduct(self.likelihood.psd.clone(),
                                     duration=self.likelihood.duration,
                                     f_min=self.likelihood.f_min, f_max=self.likelihood.f_max)

        parameters = torch.linspace(0., 2.0, 100)
        products = []

        #torch.testing.assert_close(self.likelihood._call_model({"a": 0.9})['plus'].data,
        #                           self.likelihood.data)
        
        for parameter in parameters:
            polarisations = self.likelihood._call_model({"a": parameter})
            waveform_mean = polarisations['plus'].data
            products.append(inner_product(waveform_mean, self.likelihood.data)-inner_product(waveform_mean, waveform_mean))
        self.assertLess(parameters[torch.argmax(torch.tensor(products))] - 0.9, 0.05)


    def test_inner_waveform_waveform_multiple(self):
        """Check that the waveform,waveform inner product makes sense."""
        inner_product = InnerProduct(self.likelihood.psd.clone(),
                                     duration=self.likelihood.duration,
                                     f_min=self.likelihood.f_min, f_max=self.likelihood.f_max)

        parameters = torch.linspace(0.1, 1.5, 100)
        products = []

        polarisations = self.likelihood._call_model({"a": 0.9})
        waveform_mean = polarisations['plus'].data

        #torch.testing.assert_close(waveform_mean, self.likelihood.data)
        
        for parameter in parameters:
            polarisations = self.likelihood._call_model({"a": parameter})
            waveform_mean = polarisations['plus'].data
            products.append(inner_product(waveform_mean, waveform_mean))
        products = torch.tensor(products, device=self.device)
        self.assertTrue(torch.all(products<10*inner_product(self.likelihood.data, self.likelihood.data)))

        
    def test_simple(self):
        """Check that the correct inference is made for a simple signal."""
        parameters = torch.linspace(0, 2, 100)
        likes = torch.tensor([self.likelihood({"a": parameter}, model_var=False) for parameter in parameters])
        self.assertLess(parameters[torch.argmax(likes)] - 0.9, 0.05)


class TestCUDALikelihoodFlatNoise(TestCUDALikelihood):
    device = torch.device("cuda")

    def setUp(self):

        data_length = 164
        fft_length = int(1+(data_length/2))
        
        generator = CUDATestModel(device=self.device)
        window = torch.blackman_window

        self.psd = torch.ones(fft_length, device=self.device)
        self.times = times = torch.linspace(0, 5, data_length)
        self.test_data = generator.time_domain_waveform(dict(a=0.9),
                                                        times=times
        )
        self.test_data_plus_noise = self.test_data.data + 1.2*torch.randn(164, device=self.device)

        data = Timeseries(self.test_data_plus_noise, times=times)
        
        self.likelihood = CUDALikelihood(model=generator,
                                         detector_prefix="H1",
                                         window=window,
                                         data=data,
                                         psd=self.psd,
                                         device=self.device
        )
