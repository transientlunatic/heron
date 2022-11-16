import unittest
from heron.likelihood import CUDATimedomainLikelihood
# from heron.models.torchbased
import torch
from elk.waveform import Timeseries
import lalsimulation
from heron.models.torchbased import HeronCUDA,  train

class PseudoModelIMRPhenomP:
    def __init__(self, device="cpu"):
        self.device=device

    def time_domain_waveform(self, p, times):
        detection = generate_imr_waveform(q=p['mass ratio'])
        detection = Timeseries(data=torch.tensor(detection[1], device=self.device), times=torch.tensor(detection[0], device=self.device))
        detection.variance = torch.zeros(len(detection.times), device=self.device)
        detection.covariance = torch.zeros((len(detection.times), len(detection.times)), device=self.device)
        return detection

def noise_psd(N, frequencies, psd = lambda f: 1):
    """
    Generate noise with a given PSD
    """
    M = N //2
    if N%2==1: M+=1
    reals = torch.randn(N, device=frequencies.device)
    imags = torch.randn(N, device=frequencies.device)
    
    T = 1/(frequencies[1]-frequencies[0])
    
    S = torch.sqrt(N*N/4/(T) * torch.tensor([float(psd(float(f))) for f in frequencies], device=T.device))
    
    noise_r =  S * (reals)
    noise_i =  S * (imags)

    noise_f = noise_r + 1j * noise_i
    
    return torch.fft.irfft(noise_f, n=(N));

from pycbc.waveform import get_td_waveform
def generate_imr_waveform(q, M=20):
    apx = "IMRPhenomPv2"
    unaligned = {}
    m1 = M / (1+q)
    m2 = M / (1+1/q)
    assert ((m1 + m2) - M ) < 1e-4
    unaligned['hp'], unaligned['hc'] = get_td_waveform(approximant=apx,
                                                       mass1=m1,
                                                       mass2=m2,
                                                       spin1z=0,
                                                       
                                                       distance=2000,
                                                       delta_t=1.0/4096,
                                                       f_lower=20)
    waveform_a = unaligned
    idx = (waveform_a['hp'].sample_times >= -0.05) & (waveform_a['hp'].sample_times <= 0.02)
    times = waveform_a['hp'].sample_times[idx]
    return times, waveform_a['hp'][idx], waveform_a['hp'][idx]

class PSD:
    def __init__(self, data, frequencies):
        self.data = data
        self.frequencies = frequencies
        self.df = frequencies[1] - frequencies[0]

class TestTimeDomainLikelihoodNoNoise(unittest.TestCase):

    def setUp(self):
        model = PseudoModelIMRPhenomP(device="cuda")
        srate = 4096
        dt = 1./srate
        detection = generate_imr_waveform(q=0.6)
        detection = Timeseries(data=torch.tensor(detection[1], device="cuda"), times=torch.tensor(detection[0], device="cuda"))
        psd = torch.ones(int(len(detection.times)/2+1), device="cuda")*1e-44
        f_max = srate / 2
        df = 1./(detection.times[-1]-detection.times[0])
        frequencies = torch.arange((len(detection.times) +1 ) // 2).cuda() / (dt * len(detection.times))
        psd = PSD(data=psd, frequencies=frequencies)
        self.likelihood = CUDATimedomainLikelihood(model, times=detection.times, data=detection, detector_prefix="L1", psd=psd)

    def test_residuals_with_self(self):
        """Ensure that the residual of a signal and itself with no noise is zero."""
        M = 20
        m = 0.6
        p = {
            "mass ratio": float(m),
            "total mass": M,
            "ra": 0,
            "dec": 0,
            "psi": 0,
            "gpstime": 0,
            "detector": "L1",
            "distance": 2000,}
        draw = self.likelihood._call_model(p)
        residual = self.likelihood._residual(draw).cpu()
        self.assertEqual(torch.sum(residual), 0)

    def test_weighted_residual_power_self(self):
        """Ensure that the residual power is zero for a signal with itself with no noise."""
        M = 20
        m = 0.6
        p = {
            "mass ratio": float(m),
            "total mass": M,
            "ra": 0,
            "dec": 0,
            "psi": 0,
            "gpstime": 0,
            "detector": "L1",
            "distance": 2000,}
        draw = self.likelihood._call_model(p)
        residual = self.likelihood._residual(draw)
        residual_power = self.likelihood._weighted_residual_power(residual, weight=self.likelihood.C).cpu()
        self.assertEqual(residual_power, 0)

    def test_residual_power_other(self):
        """Check that the residual power is minimum for self"""
        M = 20
        masses = torch.linspace(0.3, 1, 20)
        likes = []
        for m in masses:
            p = {
                "mass ratio": float(m),
                "total mass": M,
                "ra": 0,
                "dec": 0,
                "psi": 0,
                "gpstime": 0,
                "detector": "L1",
                "distance": 2000,}
            draw = self.likelihood._call_model(p)
            residual = self.likelihood._residual(draw)
            likes.append([self.likelihood._residual_power(residual).cpu()])
        likes = torch.tensor(likes)
        self.assertLess(masses[torch.argmin(likes)] - 0.6, 0.05)

    def test_covariance_constant(self):
        """Check that the covariance matrix is contant if no waveform uncertainty is included."""
        M = 20
        masses = torch.linspace(0.3, 1, 20)
        likes = []
        for m in masses:
            p = {
                "mass ratio": float(m),
                "total mass": M,
                "ra": 0,
                "dec": 0,
                "psi": 0,
                "gpstime": 0,
                "detector": "L1",
                "distance": 2000,}
            draw = self.likelihood._call_model(p)
            residual = self.likelihood._residual(draw)
            likes.append(torch.det(self.likelihood.C.cpu()))
        likes = torch.tensor(likes)
        self.assertTrue(torch.all(likes==likes[0]))

    def test_inverse_covariance_constant(self):
        """Check that the covariance matrix is contant if no waveform uncertainty is included."""
        M = 20
        masses = torch.linspace(0.3, 1, 20)
        likes = []
        for m in masses:
            p = {
                "mass ratio": float(m),
                "total mass": M,
                "ra": 0,
                "dec": 0,
                "psi": 0,
                "gpstime": 0,
                "detector": "L1",
                "distance": 2000,}
            draw = self.likelihood._call_model(p)
            residual = self.likelihood._residual(draw)
            likes.append(torch.inverse(self.likelihood.C.cpu()))
        #likes = torch.tensor(likes)
        for inverse in likes:
            self.assertTrue(torch.all(inverse == likes[0]))
        
    def test_weighted_residual_power_other(self):
        """Check that the weighted residual power is minimum for self"""
        M = 20
        masses = torch.linspace(0.3, 1, 20)
        likes = []
        for m in masses:
            p = {
                "mass ratio": float(m),
                "total mass": M,
                "ra": 0,
                "dec": 0,
                "psi": 0,
                "gpstime": 0,
                "detector": "L1",
                "distance": 2000,}
            draw = self.likelihood._call_model(p)
            residual = self.likelihood._residual(draw)
            likes.append([self.likelihood._weighted_residual_power(residual, weight=self.likelihood.C).cpu()])
        likes = torch.tensor(likes)
        self.assertTrue(torch.abs(masses[torch.argmin(likes)] - 0.6) < 0.05)

    def test_inversion_of_c_matrix(self):

        self.assertTrue(float(torch.det(torch.inverse(self.likelihood.C)@self.likelihood.C)) - 1 < 0.00001)
        
    def test_self_with_self(self):
        """Check that the maximum likelihood is produced by a model evaluated with itself at the correct parameters"""
        M = 20
        masses = torch.linspace(0.3, 1, 20)
        likes = torch.tensor([self.likelihood({
            "mass ratio": float(m),
            "total mass": M,
            "ra": 0,
            "dec": 0,
            "psi": 0,
            "gpstime": 0,
            "detector": "L1",
            "distance": 2000,}, model_var=False).cpu() 
        for m in masses])
        self.assertTrue(masses[torch.argmax(likes)] - 0.6 < 0.05)

class TestTimeDomainLikelihoodNoise(unittest.TestCase):

    def setUp(self):
        model = PseudoModelIMRPhenomP(device="cuda")
        srate = 4096
        dt = 1./srate
        signal = generate_imr_waveform(q=0.6)
        times = signal[0]
        df = 1/(times[-1] - times[0])
        frequencies = torch.arange(0, df+1/(times[1] - times[0]), df)
        psd_func = lalsimulation.SimNoisePSDaLIGOZeroDetHighPower
        masked_psd = lambda f:psd_func(f) if f >= 20 else psd_func(20)
        
        noise = torch.tensor(noise_psd(len(signal[0]), frequencies=frequencies, psd=masked_psd), device="cuda")
        detection = Timeseries(data=torch.tensor(torch.tensor(signal[1]).to(device=noise.device))+noise,
                               times=torch.tensor(signal[0]).to(device=noise.device))

        psd = torch.tensor([masked_psd(float(f)) for f in frequencies], dtype=torch.float64)        
        psd = PSD(data=psd, frequencies=frequencies)
        self.likelihood = CUDATimedomainLikelihood(model, times=detection.times, data=detection, detector_prefix="L1", psd=psd)

    def test_residuals_with_self(self):
        """Ensure that the residual of a signal and itself with no noise is zero."""
        M = 20
        m = 0.6
        p = {
            "mass ratio": float(m),
            "total mass": M,
            "ra": 0,
            "dec": 0,
            "psi": 0,
            "gpstime": 0,
            "detector": "L1",
            "distance": 2000,}
        draw = self.likelihood._call_model(p)
        residual = self.likelihood._residual(draw).cpu()
        self.assertEqual(torch.sum(residual), 0)

    def test_weighted_residual_power_self(self):
        """Ensure that the residual power is zero for a signal with itself with no noise."""
        M = 20
        m = 0.6
        p = {
            "mass ratio": float(m),
            "total mass": M,
            "ra": 0,
            "dec": 0,
            "psi": 0,
            "gpstime": 0,
            "detector": "L1",
            "distance": 2000,}
        draw = self.likelihood._call_model(p)
        residual = self.likelihood._residual(draw)
        residual_power = self.likelihood._weighted_residual_power(residual, weight=self.likelihood.C).cpu()
        self.assertEqual(residual_power, 0)

    def test_residual_power_other(self):
        """Check that the residual power is minimum for self"""
        M = 20
        masses = torch.linspace(0.3, 1, 20)
        likes = []
        for m in masses:
            p = {
                "mass ratio": float(m),
                "total mass": M,
                "ra": 0,
                "dec": 0,
                "psi": 0,
                "gpstime": 0,
                "detector": "L1",
                "distance": 2000,}
            draw = self.likelihood._call_model(p)
            residual = self.likelihood._residual(draw)
            likes.append([self.likelihood._residual_power(residual).cpu()])
        likes = torch.tensor(likes)
        self.assertLess(masses[torch.argmin(likes)] - 0.6, 0.05)

    def test_covariance_constant(self):
        """Check that the covariance matrix is contant if no waveform uncertainty is included."""
        M = 20
        masses = torch.linspace(0.3, 1, 20)
        likes = []
        for m in masses:
            p = {
                "mass ratio": float(m),
                "total mass": M,
                "ra": 0,
                "dec": 0,
                "psi": 0,
                "gpstime": 0,
                "detector": "L1",
                "distance": 2000,}
            draw = self.likelihood._call_model(p)
            residual = self.likelihood._residual(draw)
            likes.append(torch.det(self.likelihood.C.cpu()))
        likes = torch.tensor(likes)
        self.assertTrue(torch.all(likes==likes[0]))

    def test_inverse_covariance_constant(self):
        """Check that the covariance matrix is contant if no waveform uncertainty is included."""
        M = 20
        masses = torch.linspace(0.3, 1, 20)
        likes = []
        for m in masses:
            p = {
                "mass ratio": float(m),
                "total mass": M,
                "ra": 0,
                "dec": 0,
                "psi": 0,
                "gpstime": 0,
                "detector": "L1",
                "distance": 2000,}
            draw = self.likelihood._call_model(p)
            residual = self.likelihood._residual(draw)
            likes.append(torch.inverse(self.likelihood.C.cpu()))
        #likes = torch.tensor(likes)
        for inverse in likes:
            self.assertTrue(torch.all(inverse == likes[0]))
        
    def test_weighted_residual_power_other(self):
        """Check that the weighted residual power is minimum for self"""
        M = 20
        masses = torch.linspace(0.3, 1, 20)
        likes = []
        for m in masses:
            p = {
                "mass ratio": float(m),
                "total mass": M,
                "ra": 0,
                "dec": 0,
                "psi": 0,
                "gpstime": 0,
                "detector": "L1",
                "distance": 2000,}
            draw = self.likelihood._call_model(p)
            residual = self.likelihood._residual(draw)
            likes.append([self.likelihood._weighted_residual_power(residual, weight=self.likelihood.C).cpu()])
        likes = torch.tensor(likes)
        self.assertTrue(torch.abs(masses[torch.argmin(likes)] - 0.6) < 0.05)

    def test_inversion_of_c_matrix(self):

        self.assertTrue(float(torch.det(torch.inverse(self.likelihood.C)@self.likelihood.C)) - 1 < 0.00001)
        
    def test_self_with_self(self):
        """Check that the maximum likelihood is produced by a model evaluated with itself at the correct parameters"""
        M = 20
        masses = torch.linspace(0.3, 1, 20)
        likes = torch.tensor([self.likelihood({
            "mass ratio": float(m),
            "total mass": M,
            "ra": 0,
            "dec": 0,
            "psi": 0,
            "gpstime": 0,
            "detector": "L1",
            "distance": 2000,}, model_var=False).cpu() 
        for m in masses])
        self.assertTrue(masses[torch.argmax(likes)] - 0.6 < 0.05)


class TestTimeDomainLikelihoodNoNoiseGPR(unittest.TestCase):

    def setUp(self):
        model = HeronCUDA(datafile="training_data.h5", 
                  datalabel="IMR training linear", 
                  name="Heron IMR Non-spinning",
                  device=torch.device("cuda"),
                 )
        #train(model, iterations=1000)
        srate = 4096
        
        times = torch.linspace(-0.05, 0.005, int((0.005+0.05)*srate))
        dt = 1./srate
        #signal = generate_imr_waveform(q=0.6)
        M = 20
        p = {
            "mass ratio": 0.6,
            "total_mass": M,
            #"mass_1":35,
            #"mass_2": 30,
            "ra": 0,
            "dec": 0,
            "psi": 0,
            "gpstime": 0,
            "detector": "L1",
            "distance": 2000,
        }
        signal = model.time_domain_waveform(times=times, p=p)

        psd = torch.ones(int(len(signal.times)/2+1), device="cuda")*1e-46
        f_max = srate / 2
        df = 1./(signal.times[-1]-signal.times[0])
        frequencies = torch.arange((len(signal.times) + 1) // 2).cuda() / (dt * len(signal.times))
        psd = PSD(data=psd, frequencies=frequencies)
        
        times = signal.times
        df = 1/(times[-1] - times[0])
        detection = Timeseries(data=torch.tensor(torch.tensor(signal.data).to(device="cuda")),
                               times=torch.tensor(signal.times).to(device="cuda"))

        self.likelihood = CUDATimedomainLikelihood(model, times=detection.times, data=detection, detector_prefix="L1", psd=psd)

    def test_residuals_with_self(self):
        """Ensure that the residual of a signal and itself with no noise is zero."""
        M = 20
        m = 0.6
        p = {
            "mass ratio": float(m),
            "total mass": M,
            "ra": 0,
            "dec": 0,
            "psi": 0,
            "gpstime": 0,
            "detector": "L1",
            "distance": 2000,}
        draw = self.likelihood._call_model(p)
        residual = self.likelihood._residual(draw).cpu()
        self.assertEqual(torch.sum(residual), 0)

    def test_weighted_residual_power_self(self):
        """Ensure that the residual power is zero for a signal with itself with no noise."""
        M = 20
        m = 0.6
        p = {
            "mass ratio": float(m),
            "total mass": M,
            "ra": 0,
            "dec": 0,
            "psi": 0,
            "gpstime": 0,
            "detector": "L1",
            "distance": 2000,}
        draw = self.likelihood._call_model(p)
        residual = self.likelihood._residual(draw)
        residual_power = self.likelihood._weighted_residual_power(residual, weight=self.likelihood.C).cpu()
        self.assertEqual(residual_power, 0)

    def test_residual_power_other(self):
        """Check that the residual power is minimum for self"""
        M = 20
        masses = torch.linspace(0.3, 1, 20)
        likes = []
        for m in masses:
            p = {
                "mass ratio": float(m),
                "total mass": M,
                "ra": 0,
                "dec": 0,
                "psi": 0,
                "gpstime": 0,
                "detector": "L1",
                "distance": 2000,}
            draw = self.likelihood._call_model(p)
            residual = self.likelihood._residual(draw)
            likes.append([self.likelihood._residual_power(residual).cpu()])
        likes = torch.tensor(likes)
        self.assertLess(masses[torch.argmin(likes)] - 0.6, 0.05)

    def test_covariance_constant(self):
        """Check that the covariance matrix is contant if no waveform uncertainty is included."""
        M = 20
        masses = torch.linspace(0.3, 1, 20)
        likes = []
        for m in masses:
            p = {
                "mass ratio": float(m),
                "total mass": M,
                "ra": 0,
                "dec": 0,
                "psi": 0,
                "gpstime": 0,
                "detector": "L1",
                "distance": 2000,}
            draw = self.likelihood._call_model(p)
            residual = self.likelihood._residual(draw)
            likes.append(torch.det(self.likelihood.C.cpu()))
        likes = torch.tensor(likes)
        self.assertTrue(torch.all(likes==likes[0]))

    def test_inverse_covariance_constant(self):
        """Check that the covariance matrix is contant if no waveform uncertainty is included."""
        M = 20
        masses = torch.linspace(0.3, 1, 20)
        likes = []
        for m in masses:
            p = {
                "mass ratio": float(m),
                "total mass": M,
                "ra": 0,
                "dec": 0,
                "psi": 0,
                "gpstime": 0,
                "detector": "L1",
                "distance": 2000,}
            draw = self.likelihood._call_model(p)
            residual = self.likelihood._residual(draw)
            likes.append(torch.inverse(self.likelihood.C.cpu()))
        #likes = torch.tensor(likes)
        for inverse in likes:
            self.assertTrue(torch.all(inverse == likes[0]))
        
    def test_weighted_residual_power_other(self):
        """Check that the weighted residual power is minimum for self"""
        M = 20
        masses = torch.linspace(0.3, 1, 20)
        likes = []
        for m in masses:
            p = {
                "mass ratio": float(m),
                "total mass": M,
                "ra": 0,
                "dec": 0,
                "psi": 0,
                "gpstime": 0,
                "detector": "L1",
                "distance": 2000,}
            draw = self.likelihood._call_model(p)
            residual = self.likelihood._residual(draw)
            likes.append([self.likelihood._weighted_residual_power(residual, weight=self.likelihood.C).cpu()])
        likes = torch.tensor(likes)
        self.assertTrue(torch.abs(masses[torch.argmin(likes)] - 0.6) < 0.05)

    def test_inversion_of_c_matrix(self):

        self.assertTrue(float(torch.det(torch.inverse(self.likelihood.C)@self.likelihood.C)) - 1 < 0.00001)
        
    def test_self_with_self(self):
        """Check that the maximum likelihood is produced by a model evaluated with itself at the correct parameters"""
        M = 20
        masses = torch.linspace(0.3, 1, 20)
        likes = torch.tensor([self.likelihood({
            "mass ratio": float(m),
            "total mass": M,
            "ra": 0,
            "dec": 0,
            "psi": 0,
            "gpstime": 0,
            "detector": "L1",
            "distance": 2000,}, model_var=False).cpu() 
        for m in masses])
        self.assertTrue(masses[torch.argmax(likes)] - 0.6 < 0.05)

class TestTimeDomainLikelihoodNoiseIMR(unittest.TestCase):

    def setUp(self):
        model = HeronCUDA(datafile="training_data.h5", 
                  datalabel="IMR training linear", 
                  name="Heron IMR Non-spinning",
                  device=torch.device("cuda"),
                 )
        #train(model, iterations=1000)
        srate = 4096
        dt = 1./srate
        signal = generate_imr_waveform(q=0.6)
        times = signal[0]
        M = 20
        df = 1/(times[-1] - times[0])
        frequencies = torch.arange(0, df+1/(times[1] - times[0]), df)
        psd_func = lalsimulation.SimNoisePSDaLIGOZeroDetHighPower
        masked_psd = lambda f:psd_func(f) if f >= 20 else psd_func(20)
        
        noise = torch.tensor(noise_psd(len(signal[0]), frequencies=frequencies, psd=masked_psd), device="cuda")
        detection = Timeseries(data=torch.tensor(torch.tensor(signal[1]).to(device=noise.device))+noise,
                               times=torch.tensor(signal[0]).to(device=noise.device))

        psd = torch.tensor([masked_psd(float(f)) for f in frequencies], dtype=torch.float64)
        psd = PSD(data=psd, frequencies=frequencies)
        self.likelihood = CUDATimedomainLikelihood(model, times=detection.times, data=detection, detector_prefix="L1", psd=psd)

    def test_residual_power_other(self):
        """Check that the residual power is minimum for self"""
        M = 20
        masses = torch.linspace(0.3, 1, 20)
        likes = []
        for m in masses:
            p = {
                "mass ratio": float(m),
                "total mass": M,
                "ra": 0,
                "dec": 0,
                "psi": 0,
                "gpstime": 0,
                "detector": "L1",
                "distance": 2000,}
            draw = self.likelihood._call_model(p)
            residual = self.likelihood._residual(draw)
            likes.append([self.likelihood._residual_power(residual).cpu()])
        likes = torch.tensor(likes)
        self.assertLess(masses[torch.argmin(likes)] - 0.6, 0.05)

    def test_covariance_constant(self):
        """Check that the covariance matrix is contant if no waveform uncertainty is included."""
        M = 20
        masses = torch.linspace(0.3, 1, 20)
        likes = []
        for m in masses:
            p = {
                "mass ratio": float(m),
                "total mass": M,
                "ra": 0,
                "dec": 0,
                "psi": 0,
                "gpstime": 0,
                "detector": "L1",
                "distance": 2000,}
            draw = self.likelihood._call_model(p)
            residual = self.likelihood._residual(draw)
            likes.append(torch.det(self.likelihood.C.cpu()))
        likes = torch.tensor(likes)
        self.assertTrue(torch.all(likes==likes[0]))

    def test_inverse_covariance_constant(self):
        """Check that the covariance matrix is contant if no waveform uncertainty is included."""
        M = 20
        masses = torch.linspace(0.3, 1, 20)
        likes = []
        for m in masses:
            p = {
                "mass ratio": float(m),
                "total mass": M,
                "ra": 0,
                "dec": 0,
                "psi": 0,
                "gpstime": 0,
                "detector": "L1",
                "distance": 2000,}
            draw = self.likelihood._call_model(p)
            residual = self.likelihood._residual(draw)
            likes.append(torch.inverse(self.likelihood.C.cpu()))
        #likes = torch.tensor(likes)
        for inverse in likes:
            self.assertTrue(torch.all(inverse == likes[0]))
        
    def test_weighted_residual_power_other(self):
        """Check that the weighted residual power is minimum for self"""
        M = 20
        masses = torch.linspace(0.3, 1, 20)
        likes = []
        for m in masses:
            p = {
                "mass ratio": float(m),
                "total mass": M,
                "ra": 0,
                "dec": 0,
                "psi": 0,
                "gpstime": 0,
                "detector": "L1",
                "distance": 2000,}
            draw = self.likelihood._call_model(p)
            residual = self.likelihood._residual(draw)
            likes.append([self.likelihood._weighted_residual_power(residual, weight=self.likelihood.C).cpu()])
        likes = torch.tensor(likes)
        self.assertTrue(torch.abs(masses[torch.argmin(likes)] - 0.6) < 0.05)

    def test_inversion_of_c_matrix(self):

        self.assertTrue(float(torch.det(torch.inverse(self.likelihood.C)@self.likelihood.C)) - 1 < 0.00001)
        
    def test_self_with_self(self):
        """Check that the maximum likelihood is produced by a model evaluated with itself at the correct parameters"""
        M = 20
        masses = torch.linspace(0.3, 1, 20)
        likes = torch.tensor([self.likelihood({
            "mass ratio": float(m),
            "total mass": M,
            "ra": 0,
            "dec": 0,
            "psi": 0,
            "gpstime": 0,
            "detector": "L1",
            "distance": 2000,}, model_var=False).cpu() 
        for m in masses])
        self.assertTrue(masses[torch.argmax(likes)] - 0.6 < 0.05)
