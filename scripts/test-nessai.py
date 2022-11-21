#!/usr/bin/env python

import numpy as np
import lalsimulation

from heron.models.torchbased import HeronCUDA,  train
from heron.likelihood import CUDALikelihood, InnerProduct, CUDATimedomainLikelihood
from heron.data import DataWrapper
import gpytorch
import torch
from elk.waveform import Timeseries
from heron.models.torchbased import HeronCUDA,  train

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger

import scipy.signal

class PSD:
    def __init__(self, data, frequencies):
        self.data = data
        self.frequencies = frequencies
        self.df = frequencies[1] - frequencies[0]


def noise_psd(N, frequencies, psd = lambda f: 1):
    """
    Generate noise with a given PSD
    """
    reals = np.random.randn(N//2+1)
    imags = np.random.randn(N//2+1)

    T = 1/(frequencies[1]-frequencies[0])

    S = np.sqrt(N*N/4/(T) * np.array([psd(float(f)) for f in frequencies]))

    noise_r =  S * (reals)
    noise_i =  S * (imags)

    noise_f = noise_r + 1j * noise_i

    return np.fft.irfft(noise_f, n=(N));

# model = HeronCUDA(datafile="test_file_2.h5", datalabel="IMR training linear",
#                   device=torch.device("cuda"),
model = HeronCUDA(datafile="training_data.h5", 
                  datalabel="IMR training linear", 
                  name="Heron IMR Non-spinning",
                  device=torch.device("cuda"),
                 )
#train(model, iterations=1000)                 )
#train(model, iterations=500)

# We want to sample at a sample rate of 2048-Hz
srate = 4096
# This means that we'll have 2048 samples per second, or a sample spacing of 1/srate
dt = 1./srate
# This gives us a time axis:
times = torch.linspace(-0.05, 0.005, int((0.005+0.05)*srate))
#
# If we work in the Fourier domain then the maximum frequency will be half the sampling rate
f_max = srate / 2
df = 1./(times[-1]-times[0])
frequencies = torch.arange((len(times) +1 ) // 2) / (dt * len(times))
#
 #
psd_func = lalsimulation.SimNoisePSDaLIGOZeroDetHighPower # lalsimulation.SimNoisePSDaLIGOEarlyHighSensitivityP1200087
masked_psd = lambda f:psd_func(f) if f>=20 else 0
psd = np.array([masked_psd(float(f)) for f in frequencies])
psd = PSD(data=psd, frequencies=frequencies)

p = {
    "mass ratio": 0.5,
    "total mass": 60,
    #"mass 1":35.,
    #"mass 2": 32.,
    "ra": 1.79,
    "dec": -1.22,
    "psi": 1.47,
    "gpstime": 1126259462,
    "detector": "L1",
    "distance": 1000,
}

signal = model.time_domain_waveform(times=times, p=p)
noise = torch.tensor(noise_psd(len(times), frequencies=frequencies, psd=masked_psd), device="cuda")


detection = Timeseries(data=torch.tensor(signal.data)+noise, times=signal.times)
sos = scipy.signal.butter(10, 20, 'hp', fs=float(1/(detection.times[1] - detection.times[0])), output='sos')
detection.data = torch.tensor(scipy.signal.sosfilt(sos, detection.data.cpu()), device=detection.data.device)

heron_likelihood = CUDATimedomainLikelihood(
    model, times=times, data=detection, detector_prefix="L1", psd=psd
)
# check likelihood works
masses = np.linspace(0.1,1.0,100)
likes = torch.tensor([heron_likelihood({
    "mass ratio": m,
    "total mass": 40,
    "ra": 1.79,
    "dec": -1.22,
    "psi": 1.47,
    "gpstime": 1126259462,
    "detector": "L1",
    "distance": 400,
}, model_var=True).cpu()
        for m in masses])


output = 'heron_test_2'
logger = setup_logger(output=output, label=output, log_level='WARNING')

priors = {
    "mass ratio": [0.1, 1.0],
    #"distance": [100, 1000],
    #"psi": [0, 2*np.pi],
    "total mass": [20, 100],
    #"gpstime": [1126259461.9, 1126259462.1],
}

device = heron_likelihood.device

base_p = {
    "mass ratio": 0.6,
    "total mass": 40,
    "ra": 1.79,
    "dec": -1.22,
    "psi": 1.47,
    "gpstime": 1126259462,
    "detector": "L1",
    "distance": 400,
}


class HeronModel(Model):
    """Nessai mode for Heron Likelihoods.

    This simple model uses uniform priors on all parameters.

    Parameters
    ----------
    heron_likelihood
        Instance of heron likelihood.
    priors
        Prior dictionary.
    """
    allow_vectorised = False

    def __init__(self, heron_likelihood, priors):
        # Names of parameters to sample
        self.names = list(priors.keys())
        self.bounds = priors
        self.heron_likelihood = heron_likelihood

    def log_prior(self, x):
        log_p = np.log(self.in_bounds(x), dtype=float)
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def log_likelihood(self, x):
        with torch.inference_mode():
            # Need to convert from numpy floats to python floats
            base_p.update({n: float(x[n]) for n in self.names})
            return self.heron_likelihood(base_p, model_var=True).cpu().numpy()


nessai_model = HeronModel(heron_likelihood, priors)


fp = FlowSampler(
    nessai_model,
    nlive=2000,
    maximum_uninformed=4000,
    output=output,
    resume=False,
    seed=1234,
    flow_class='FlowProposal',
    signal_handling=False,
)

fp.run()
