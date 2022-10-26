import unittest

import os

import gpytorch
import torch
import numpy as np

import pycbc.waveform
from pycbc.waveform import get_td_waveform
import numpy as np
import matplotlib.pyplot as plt

from elk.catalogue import PPCatalogue

from heron.data import DataWrapper
import heron.testing
from heron.models.torchbased import HeronCUDA, train
from heron.likelihood import Match

class TestSimpleModel(unittest.TestCase):
    """
    These tests are designed to verify that heron can build an extremely simple model,
    and then have the functions produced from it verified with the same tests we use
    for waveforms.
    """

    @classmethod
    def setUpClass(cls):
        data = DataWrapper.create("test_training_data.h5")
        qs = np.logspace(np.log(0.1), 0, 30)
        apx = "IMRPhenomPv2"
        M = 20

        for q in qs:

            m1 = M / (1+q)
            m2 = M / (1+1/q)
            assert ((m1 + m2) - M ) < 1e-4

            hp, hc = get_td_waveform(approximant=apx,
                                         mass1=m1,
                                         mass2=m2,
                                         spin1z=0,
                                         delta_t=1.0/8192,
                                         f_lower=20)

            idx = (hp.sample_times > -0.05) & (hp.sample_times < 0.02)
            inspiral = (hp.sample_times > -0.15) & (hp.sample_times < -0.005)
            merger = (hp.sample_times >= -0.0075) & (hp.sample_times < 0.005)
            new_times = np.hstack([hp.sample_times[inspiral][::4], hp.sample_times[merger]])
            new_strain = np.hstack([hp[inspiral][::4], hp[merger]])
            data.add_waveform(group="IMR training",
                          polarisation="+",
                          reference_mass=M,
                          source=apx,
                          locations={"mass ratio": q},
                          times=new_times,
                          data=new_strain
                         )


        cls.imr_cat = PPCatalogue("IMRPhenomPv2", total_mass=M, fmin=10, )

    @classmethod
    def tearDownClass(cls):
        os.remove("test_training_data.h5")

    def setUp(self):
        pass

    def test_train_model(self):

        model = HeronCUDA(datafile="test_training_data.h5",
                          datalabel="IMR training", 
                          device=torch.device("cuda")
                 )

        with gpytorch.settings.fast_pred_var(), gpytorch.settings.max_cg_iterations(5000):
            train(model, iterations=5000)

        # for kernel in model.model_plus.covar_module.base_kernel.kernels:
        #     print(f"Dim: {kernel.active_dims}: {kernel.lengthscale.item():.3f}")

        matches = []
        
        times = torch.linspace(-0.05, 0.02, 400)

        for q in np.linspace(0.1, 1.0, 10):
            with self.subTest(q=q):
                p = {"mass ratio": q}
                a = model.mean(p=p, times=times)['plus']
                #print(a, type(a))
                b = self.imr_cat.waveform(p, [-0.05, 0.02, 400])[0]

                matcher = Match(psd=None, duration=0.07, window=torch.blackman_window)

                match = matcher(a, b)
                
                matches.append(float(torch.max(match.abs())))
                self.assertGreater(float(torch.max(match.abs())), 0.95)
                
            
        print(matches)
