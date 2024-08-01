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
matplotlib.use("agg")

from torch.cuda import is_available

CUDA_NOT_AVAILABLE = not is_available()


@unittest.skipIf(CUDA_NOT_AVAILABLE, "CUDA is not installed on this system")
class TestGPyTorchFundamentals(unittest.TestCase):

    train_data_plus, train_data_cross = make_optimal_manifold(
        approximant=IMRPhenomPv2,
        warp_factor=3,
        varied={"mass_ratio": dict(lower=0.1, upper=1, step=0.05)},
                                       fixed={"total_mass": 60*u.solMass,
                                              "f_min": 10*u.Hertz,
                                              "delta T": 1/(1024*u.Hertz)})

    @classmethod
    def setUpClass(cls):
        train_data_plus = torch.tensor(cls.train_data_plus.array(parameter="mass_ratio"), device="cuda", dtype=torch.float32)
        cls.train_x_plus = train_data_plus[:,[0,1]]
        cls.train_y_plus = train_data_plus[:,2]

        train_data_cross = torch.tensor(cls.train_data_cross.array(parameter="mass_ratio", component="cross"), device="cuda", dtype=torch.float32)
        cls.train_x_cross = train_data_cross[:,[0,1]]
        cls.train_y_cross = train_data_cross[:,2]

        # initialize likelihood and model
        cls.model = HeronNonSpinningApproximant(train_x_plus=cls.train_x_plus.float(),
                                                train_y_plus=cls.train_y_plus.float(),
                                                train_x_cross=cls.train_x_cross.float(),
                                                train_y_cross=cls.train_y_cross.float(),
                                                total_mass=(60*u.solMass),
                                                distance=(1*u.Mpc).to(u.meter).value,
                                                warp_scale=2,
                                                training=200,
                                                )

    def test_training(self):

        mean, _, points = self.model.time_domain_manifold(parameters={
            "mass_ratio": {"lower": 0.1, "upper": 1.0, "number": 20},
            "time": {"lower": -0.5, "upper": 0.1, "number": 150}
        })['plus']

        import matplotlib.pyplot as plt
        f, ax = plt.subplots(2,1)
        ax[0].scatter(points[:,1].cpu(), points[:,0].cpu(), c=mean.cpu(), marker='.',
                    vmax=self.train_y_plus.max(),
                    vmin=self.train_y_plus.min())
        ax[1].scatter(self.train_x_plus[:,1].cpu(), self.train_x_plus[:,0].cpu(), c=self.train_y_plus.cpu(), marker='.')
        ax[1].set_xlim([-0.125, 0.1])
        ax[0].set_xlim([-0.125, 0.1])
        f.savefig("evaluation.png")
        
    def test_evaluation(self):
        wf = self.model.time_domain(parameters={
            "mass_ratio": 1.0,
            "time": {"lower": -0.1, "upper": 0.05, "number": 350}
        })
        f = wf['plus'].plot()
        f = wf['cross'].plot()
        f.savefig("waveform_plot.png")

    def test_overlaps(self):
        """Test overlaps between the IMRPhenomPv2 model and the heron mode trained from it."""
        overlap = Overlap()
        overlaps = {}
        for mass_ratio in np.linspace(0.1, 1.0, 20):
            parameters = {
                "mass_ratio": mass_ratio,
                "total_mass": 60*u.solMass,
                "time": {"lower": -0.1, "upper": 0.05, "number": 350}
                }

            wf_gp = self.model.time_domain(parameters=parameters.copy())["plus"]
            wf_ap = IMRPhenomPv2().time_domain(parameters=parameters.copy())["plus"]
            overlaps[mass_ratio] = overlap(wf_gp, wf_ap)
        self.assertTrue(np.sum(np.array(list(overlaps.values())) > 0.8) > 10)

    def test_overlaps_aligo(self):
        """Test overlaps between the IMRPhenomPv2 model and the heron mode trained from it."""
        overlap = Overlap(psd=AdvancedLIGO())
        overlaps = {}
        for mass_ratio in np.linspace(0.1, 1.0, 20):
            parameters = {
                "mass_ratio": mass_ratio,
                "total_mass": 60*u.solMass,
                "distance": 4*u.megaparsec,
                "time": {"lower": -0.1, "upper": 0.05, "number": 350}
                }

            wf_gp = self.model.time_domain(parameters=parameters.copy())["plus"]
            wf_ap = IMRPhenomPv2().time_domain(parameters=parameters.copy())["plus"]
            overlaps[mass_ratio] = overlap(wf_gp, wf_ap)
        self.assertTrue(np.all(np.array(list(overlaps.values())) > 0.99))
