"""
Testing models
--------------

These models are designed purely for helping with unittests and other tests.
They aren't designed for use in production analyses!

"""
import torch

from .gw import HofTSurrogate, BBHNonSpinSurrogate, FrequencyMixin
from scipy.stats import norm
from elk.waveform import Timeseries


class TestModel(BBHNonSpinSurrogate, HofTSurrogate, FrequencyMixin):
    strain_input_factor = 1

    polarisations = ["plus"]

    def __init__(self, device):
        super().__init__()
        self.device = device

    def _predict(self, times, args, polarisation=None):
        args_a = dict(a=1, b=0.3)
        args_a.update(args)
        data = norm.pdf(times, args_a["a"], args_a["b"])
        output = Timeseries(
            data=torch.tensor(data, device=self.device),
            times=times,
            variance=None,
            covariance=None,
        )
        return output, None, None

    def mean(self, times, args, polarisation="plus"):
        return self._predict(times, args)[0]


class CUDATestModel(TestModel):
    pass
