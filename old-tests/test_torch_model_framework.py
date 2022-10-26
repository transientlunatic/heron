import unittest

from heron.models.torchbased import ExactGPModel, HeronCUDA, train
from heron.likelihood import InnerProduct, Likelihood, CUDALikelihood
from heron.models.testing import TestModel, CUDATestModel
from elk.waveform import Timeseries

from torch import tensor
import torch
import gpytorch
from gpytorch.kernels import RBFKernel
from gpytorch.constraints import GreaterThan, LessThan, Interval

import numpy as np
import numpy.testing as npt
np.random.seed(90)


class TestCUDALikelihoodCPU(unittest.TestCase):
    device = torch.device("cpu")

    @classmethod
    def setUpClass(cls):
        x = torch.randn(100, device=cls.device)
        y = torch.randn(100, device=cls.device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        cls.model = ExactGPModel(x, y, likelihood)
        cls.model.eval()


    def test_mean_device(self):
        """Check that the mean is produced on the correct device."""
        points = torch.linspace(0, 1, 10, device=self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var(num_probe_vectors=10), gpytorch.settings.max_root_decomposition_size(5000):
                f_preds = self.model(points)

        self.assertEqual(f_preds.mean.device.type, self.device.type)


class TestCUDALikelihoodGPU(TestCUDALikelihoodCPU):
    device = torch.device("cuda")

    @classmethod
    def setUpClass(cls):
        x = torch.randn(100, device=cls.device)
        y = torch.randn(100, device=cls.device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(device=cls.device)
        cls.model = ExactGPModel(x, y, likelihood, device=cls.device).cuda()
        cls.model.eval()

