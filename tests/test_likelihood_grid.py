import unittest
from heron.likelihood import determine_overlap
from elk.waveform import Timeseries
import torch

import heron.injection
import scipy.signal

import heron.models.lalinference
from heron.models.torchbased import HeronCUDA,  train
from heron.likelihood import CUDATimedomainLikelihood

import lalsimulation

class TestLikelihoodTimeWindowing(unittest.TestCase):

    def generate_injection(self):
        times = {"duration": 0.10,
                 "sample rate": 4096,
                 "before": 0.3,
                 "after": 0.01,
                 }

        psd = heron.injection.psd_from_lalinference(
            "SimNoisePSDaLIGOZeroDetHighPower",
            frequencies=heron.injection.frequencies_from_times(times),
        )

        settings = {}
        settings['injection'] = {"mass ratio": 0.6,
                                 "total mass": 65.,
                                 "ra": 1.79,
                                 "dec": -1.22,
                                 "psi": 1.47,
                                 "gpstime": 1000,
                                 "detector": "L1",
                                 "distance": 1000,
                                 "before": 0.3,
                                 }
        settings['injection'].update(times)
        
        noise = heron.injection.create_noise_series(psd, times)
        signal = heron.models.lalinference.IMRPhenomPv2(torch.device("cuda")).time_domain_waveform(
            p=settings["injection"]
        )
        detection = Timeseries(data=signal.data + noise, times=signal.times)
        sos = scipy.signal.butter(
            10,
            20,
            "hp",
            fs=float(1 / (detection.times[1] - detection.times[0])),
            output="sos",
        )
        detection.data = torch.tensor(
            scipy.signal.sosfilt(sos, detection.data.cpu()),
            device=detection.data.device,
        )

        return detection, psd
    
    def setUp(self):
        DISABLE_CUDA = False
        if not DISABLE_CUDA and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.model = HeronCUDA(datafile="training_data.h5", 
                               datalabel="IMR training linear", 
                               name="Heron IMR Non-spinning",
                               device=self.device,
                               )
        self.model.eval()

    def test_exploring_mass_space(self):
        p = {
            "sample rate": 4096,
            "mass ratio": 0.7,
            "total mass": 40.0,
            "ra": 1.79,
            "dec": -1.22,
            "psi": 1.47,
            "gpstime": 1000.,
            "detector": "L1",
            "distance": 1000.,
            "after": 0.2,
            "before": 0.3,
        }
        signal = self.model.time_domain_waveform(p=p)
        detection, psd = self.generate_injection()
        self.assertNotEqual(detection.times[0] - signal.times[0], 0)
        parameters = p.copy()
        
        l = CUDATimedomainLikelihood(self.model, data=detection, detector_prefix="L1", psd=psd)

        likes = []
        mass_differences = torch.linspace(-5, 5, 25)
        for mass_difference in mass_differences:
            parameters_updated = parameters.copy()
            parameters_updated['total mass'] += float(mass_difference)
            likes.append(l(parameters_updated, model_var=True).cpu())
            #print(parameters_updated['mass ratio'], likes[-1])
            self.assertTrue(likes[-1].isfinite())

        print("Maximum likelihood", float(max(likes)),
              "at",
              "total mass",
              parameters['total mass'] + float(mass_differences[torch.argmax(torch.tensor(likes))])) 
            
        self.assertTrue(max(likes) > likes[0])
        self.assertTrue(max(likes) > likes[-1])

        
    def test_exploring_mass_ratio_space(self):
        p = {
            "sample rate": 4096,
            "mass ratio": 0.7,
            "total mass": 40.0,
            "ra": 1.79,
            "dec": -1.22,
            "psi": 1.47,
            "gpstime": 1000.,
            "detector": "L1",
            "distance": 1000.,
            "after": 0.2,
            "before": 0.3,
        }
        signal = self.model.time_domain_waveform(p=p)
        detection, psd = self.generate_injection()
        self.assertNotEqual(detection.times[0] - signal.times[0], 0)
        parameters = p.copy()
        
        l = CUDATimedomainLikelihood(self.model, data=detection, detector_prefix="L1", psd=psd)

        likes = []
        mass_differences = torch.linspace(-0.1, 0.1, 25)
        for mass_difference in mass_differences:
            parameters_updated = parameters.copy()
            parameters_updated['mass ratio'] += float(mass_difference)
            likes.append(l(parameters_updated, model_var=True).cpu())
            #print(parameters_updated['mass ratio'], likes[-1])
            self.assertTrue(likes[-1].isfinite())

        print("Maximum likelihood", float(max(likes)),
              "at",
              "mass ratio",
              parameters['mass ratio'] + float(mass_differences[torch.argmax(torch.tensor(likes))])) 
            
        self.assertTrue(max(likes) > likes[0])
        self.assertTrue(max(likes) > likes[-1])


    def test_exploring_distance_space(self):
        p = {
            "sample rate": 4096,
            "mass ratio": 0.7,
            "total mass": 40.0,
            "ra": 1.79,
            "dec": -1.22,
            "psi": 1.47,
            "gpstime": 1000.,
            "detector": "L1",
            "distance": 1000.,
            "after": 0.2,
            "before": 0.3,
        }
        signal = self.model.time_domain_waveform(p=p)
        detection, psd = self.generate_injection()
        self.assertNotEqual(detection.times[0] - signal.times[0], 0)
        parameters = p.copy()
        
        l = CUDATimedomainLikelihood(self.model, data=detection, detector_prefix="L1", psd=psd)

        likes = []
        mass_differences = torch.linspace(10, 10000, 25)
        for mass_difference in mass_differences:
            parameters_updated = parameters.copy()
            parameters_updated['distance'] += float(mass_difference)
            likes.append(l(parameters_updated, model_var=True).cpu())
            #print(parameters_updated['mass ratio'], likes[-1])
            self.assertTrue(likes[-1].isfinite())

        print("Maximum likelihood", float(max(likes)),
              "at",
              "distance",
              parameters['distance'] + float(mass_differences[torch.argmax(torch.tensor(likes))])) 
            
        self.assertTrue(max(likes) > likes[0])
        self.assertTrue(max(likes) > likes[-1])
