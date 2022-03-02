import unittest

from heron.matched import InnerProduct, Likelihood
from heron.utils import Complex
from heron.models.torchbased import HeronCUDAIMR
from elk.waveform import Timeseries

from torch import tensor
import torch

import numpy as np
import numpy.testing as npt
np.random.seed(90)


class TestInnerProduct(unittest.TestCase):
    def setUp(self):
        self.window = torch.blackman_window(180)
        self.inner_flat = InnerProduct(noise=Complex(torch.ones(len(self.window),2)))
        self.inner_model = InnerProduct(noise=Complex(torch.ones(len(self.window),2)),
                                        noise2=Complex(torch.ones(len(self.window),2))
        )

    def test_ones(self):
        """Check arrays of ones return the correct result."""
        a = Complex(torch.ones((len(self.window), 2)))
        b = Complex(torch.ones((len(self.window), 2)))

        npt.assert_almost_equal(self.inner_flat(a,b, len(self.window)).numpy(), 4.0)

    def test_ones_model(self):
        """Check arrays of ones give the correct result when there is also model uncertainty."""
        a = Complex(torch.ones((len(self.window), 2)))
        b = Complex(torch.ones((len(self.window), 2)))

        npt.assert_almost_equal(self.inner_model(a,b, len(self.window)).numpy(), 2.0)


class TestLikelihood(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = HeronCUDAIMR(device=self.device)
        self.window = torch.blackman_window(180, device=self.device)
        self.empty = Timeseries(data=torch.zeros(len(self.window), device=self.device),
                                times=np.linspace(0,1, len(self.window))
        )
        self.asd = Complex(torch.ones((len(self.window),2), device=self.device))
        self.likelihood = Likelihood(self.model, self.empty, self.window, self.asd.clone(), device=self.device)

    def test_model_call(self):
        """Check that the model is called correctly."""
        p = {"mass ratio": 0.9}
        result = self.model.time_domain_waveform(p,
                                                 times=np.linspace(0,1,len(self.window)))
        result_fft = torch.tensor(self.window*result[0].data).rfft(1)

        npt.assert_array_almost_equal(self.likelihood._call_model(p)[0].tensor, result_fft[1:]*1e19)
