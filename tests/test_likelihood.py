import unittest
import torch
import scipy

from asimov.utils import update

from elk.waveform import Timeseries

from heron.models.torchbased import HeronCUDA
from heron.likelihood import CUDATimedomainLikelihood, device
from heron.types import PSD
import heron.injection

from hypothesis import given
import hypothesis.strategies as st


class TestTimeDomainLikelihood(unittest.TestCase):
    """Interrogate the functionality of the time-domain likelihood class"""

    def _create_injection(self, **kwargs):
        """
        Create an injection for the tests.
        """

        defaults = {"noise model": "SimNoisePSDaLIGOZeroDetHighPower",
                    "times": {"duration": 2,
                              "sample rate": 4096,
                              "before": 0.05,
                              "after": 0.01,
                              },
                    "parameters": {
                        "mass ratio": 0.6,
                        "total mass": 65,
                        "ra": 1.79,
                        "dec": -1.22,
                        "psi": 1.47,
                        "gpstime": 1126259462,
                        "detector": "L1",
                        "distance": 1000,
                        },
                    }
        defaults.update(kwargs)
        
        psd = heron.injection.psd_from_lalinference(
            defaults['noise model'],
            frequencies=heron.injection.frequencies_from_times(defaults['times']),
        )
        noise = heron.injection.create_noise_series(psd, defaults['times'])
        signal = self.model.time_domain_waveform(
            p=dict(defaults['parameters'], **defaults['times']),
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

    def _create_clean_waveform(self, **kwargs):
        """
        Create an injection for the tests.
        """

        defaults = {"noise model": "SimNoisePSDaLIGOZeroDetHighPower",
                    "times": {"duration": 2,
                              "sample rate": 4096,
                              "before": 0.05,
                              "after": 0.01,
                              },
                    "parameters": {
                        "mass ratio": 0.6,
                        "total mass": 65,
                        "ra": 1.79,
                        "dec": -1.22,
                        "psi": 1.47,
                        "gpstime": 1126259462,
                        "detector": "L1",
                        "distance": 1000,
                        },
                    }
        update(defaults, kwargs)
        signal = self.model.time_domain_waveform(
            p=dict(defaults['parameters'], **defaults['times']),
        )
        detection = Timeseries(data=signal.data, times=signal.times)
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
        return detection
    
    def setUp(self):
        
        self.model = HeronCUDA(
            datafile="training_data.h5",
            datalabel="IMR training linear",
            name="Heron IMR Non-spinning",
            device=torch.device("cuda"),
        )

        self.detection, self.psd = self._create_injection()
        times = {"duration": 2,
                 "sample rate": 4096,
                 "before": 0.05,
                 "after": 0.01,
                 }
        
        self.likelihood = CUDATimedomainLikelihood(
            self.model,
            data=self.detection,
            detector_prefix="L1",
            generator_args=times,
            psd=self.psd,
        )

    def test_waveform_draw(self):
        defaults = {"noise model": "SimNoisePSDaLIGOZeroDetHighPower",
                    "times": {"duration": 2,
                              "sample rate": 4096,
                              "before": 0.05,
                              "after": 0.01,
                              },
                    "parameters": {
                        "mass ratio": 0.6,
                        "total mass": 65,
                        "ra": 1.79,
                        "dec": -1.22,
                        "psi": 1.47,
                        "gpstime": 1126259462,
                        "detector": "L1",
                        "distance": 1000,
                        },
                    }
        a = self.model.time_domain_waveform(
            p=dict(defaults['parameters'], **defaults['times']),
        )
        self.assertTrue(1e-25 < torch.mean(a.data) < 1e-22)


    @given(gpstime=st.floats(min_value=100, max_value=2126259462,
                             allow_nan=False, allow_infinity=False))
    def test_evaluate_model_changing_time_alignment(self, gpstime):
       """Test that the injection and the likelihood waveform are aligned"""
       test_ts = self._create_clean_waveform(parameters={'gpstime':gpstime})
       p = {
           "duration": 2,
           "sample rate": 4096,
           "before": 0.05,
           "after": 0.01,
           "mass ratio": 0.6,
           "total mass": 65,
           "ra": 1.79,
           "dec": -1.22,
           "psi": 1.47,
           "gpstime": gpstime,
           "detector": "L1",
           "distance": 1000,
       }
       ts = self.likelihood._call_model(p, times=test_ts.times)
       self.assertEqual(ts.times[0], gpstime-p["before"])


    def test_normalisation__part1_simple(self):
        """Check that the first part of the normalisation is correct."""
        # This requires a covariance matrix, so this test will fake this first off.
        K = torch.eye(10)
        self.assertEqual(self.likelihood._normalisation_A(K),
                         -torch.log(torch.sqrt((torch.tensor(2)*torch.pi)**10)))

    def test_normalisation__part2_simple(self):
        """Check the second part of the normalisation is correct."""
        K = torch.eye(10)
        S = torch.eye(10)
        a = (2 * torch.pi)**10 / torch.det(torch.inverse(K) + torch.inverse(S))
        self.assertEqual(self.likelihood._normalisation_B(K, S), torch.log(torch.sqrt(a)))

    def test_normalisation__data(self):
        """Check the data-data component of the likelihood"""
        times = {"duration": 2,
                 "sample rate": 4096,
                 "before": 0.05,
                 "after": 0.01,
                 }

        likelihood = CUDATimedomainLikelihood(
            self.model,
            data= Timeseries(data=torch.ones(10, device=device, dtype=torch.float64),
                             times=torch.linspace(0, 10., 10, device=device)), 
            detector_prefix="L1",
            generator_args=times,
            psd=PSD(data=torch.ones(10, device=device, dtype=torch.float64),
                    frequencies=torch.linspace(0, 10., 10, device=device)),
        )
        
        self.assertEqual(likelihood._weighted_data(), -0.5*10/likelihood.C[0,0])
        
    def test_normalisation__model(self):
        """Check the model-model component of the likelihood"""
        times = {"duration": 2,
                 "sample rate": 4096,
                 "before": 0.05,
                 "after": 0.01,
                 }

        mu = torch.ones(10)
        K = torch.eye(10)
        
        self.assertEqual(self.likelihood._weighted_model(mu, K), -0.5*10)

    def test_normalisation_cross(self):
        """Check the data-model component of the likelihood"""
        times = {"duration": 2,
                 "sample rate": 4096,
                 "before": 0.05,
                 "after": 0.01,
                 }

        likelihood = CUDATimedomainLikelihood(
            self.model,
            data= Timeseries(data=torch.ones(10, device=device, dtype=torch.float64),
                             times=torch.linspace(0, 10., 10, device=device)), 
            detector_prefix="L1",
            generator_args=times,
            psd=PSD(data=torch.ones(10, device=device, dtype=torch.float64),
                    frequencies=torch.linspace(0, 10., 10, device=device)),
        )
        mu = torch.ones(10, device=device)
        K = torch.eye(10, device=device)
        self.assertTrue(likelihood._weighted_cross(mu, K).cpu()-(5+0.5*10/likelihood.C[0,0]).cpu() < 0.001)
    

    @given(gpstime=st.floats(min_value=100, max_value=2126259462, allow_nan=False, allow_infinity=False))
    def test_likelihood_evaluation(self, gpstime):
        """Test the process of evaluating the likelihood function at any value."""
        p =  {
            "duration": 2,
            "sample rate": 4096,
            "before": 0.05,
            "after": 0.01,
            "mass ratio": 0.6,
            "total mass": 65,
            "ra": 1.79,
            "dec": -1.22,
            "psi": 1.47,
            "gpstime": gpstime,
            "detector": "L1",
            "distance": 1000,
        }
        self.assertTrue(self.likelihood(p) < 0)

    @given(gpstime=st.floats(min_value=1126259460, max_value=1126259464, allow_nan=False, allow_infinity=False),
           mass_ratio=st.floats(min_value=0.001, max_value=1, allow_nan=False, allow_infinity=False),
           total_mass=st.floats(min_value=0.001, max_value=300, allow_nan=False, allow_infinity=False),
           duration=st.floats(min_value=0.5, max_value=4, allow_nan=False, allow_infinity=False),
           before=st.floats(min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False),
           )
    def test_likelihood_evaluation_sensible(self, gpstime, mass_ratio, total_mass, duration, before):
        """Test the process of evaluating the likelihood function at any value."""
        p =  {
            "duration": duration,
            "sample rate": 4096,
            "before": before,
            "after": 0.01,
            "mass ratio": mass_ratio,
            "total mass": total_mass,
            "ra": 1.79,
            "dec": -1.22,
            "psi": 1.47,
            "gpstime": gpstime,
            "detector": "L1",
            "distance": 1000,
        }
        self.assertTrue(self.likelihood(p) < 0)
        
    def test_injection_creation(self):
        """Check that injections are created correctly within heron."""
        
        injection_ts, injection_psd = self._create_injection()

        self.assertEqual(int(injection_ts.times[0]), 1126259462 - 1)
        
    @given(gpstime=st.floats(min_value=100, max_value=2126259462, allow_nan=False, allow_infinity=False))
    def test_evaluate_model_changing_time(self, gpstime):
        """Test that the waveform behaves in a sane fashion when the evaluated."""
        test_ts = self._create_clean_waveform(parameters={'gpstime':gpstime})
        # Check that the gps time is actually in the timeseries which is generated.
        self.assertTrue(float(test_ts.times[0]) < gpstime < float(test_ts.times[-1]))
        # Check that the signal duration is what we expect
        self.assertEqual(len(test_ts.times), int(4096 * 0.06))

    @given(before=st.floats(min_value=0.001, max_value=0.5, allow_nan=False, allow_infinity=False),
           after=st.floats(min_value=0.01, max_value=0.05, allow_nan=False, allow_infinity=False))
    def test_evaluate_model_changing_time_limits(self, before, after):
        """Test that the waveform behaves in a sane fashion when the evaluated."""
        test_ts = self._create_clean_waveform(times={'before': before, 'after': after})
        # Check that the signal duration is what we expect
        self.assertEqual(len(test_ts.times), int(4096 * (before + after)))
