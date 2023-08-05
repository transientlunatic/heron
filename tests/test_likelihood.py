import unittest
import torch
import scipy

from asimov.utils import update

from elk.waveform import Timeseries

from heron.models.torchbased import HeronCUDA
from heron.likelihood import CUDATimedomainLikelihood
import heron.injection

from hypothesis import given
import hypothesis.strategies as st


class test_TimeDomainLikelihood(unittest.TestCase):
    """
    These tests are designed to interrogate the functioning of the time domain likelihood class.
    """

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

    @given(gpstime=st.floats(min_value=100, max_value=2126259462,
                            allow_nan=False, allow_infinity=False))
    def test_evaluate_model_changing_time_alignment(self, gpstime):
       """Test that the injection and the likelihood waveform are aligned"""
       test_ts = self._create_clean_waveform(parameters={'gpstime':gpstime})
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
       ts = self.likelihood._call_model(p)
       self.assertEqual(ts.times[0], test_ts.times[0])


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
           after=st.floats(min_value=0.001, max_value=0.05, allow_nan=False, allow_infinity=False))
    def test_evaluate_model_changing_time(self, before, after):
        """Test that the waveform behaves in a sane fashion when the evaluated."""
        test_ts = self._create_clean_waveform(times={'before': before, 'after': after})
        # Check that the signal duration is what we expect
        self.assertEqual(len(test_ts.times), int(4096 * (before + after)))
