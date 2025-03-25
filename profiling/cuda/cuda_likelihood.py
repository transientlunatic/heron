import numpy as np
import astropy.units as u
from heron.models.lalnoise import AdvancedLIGO
from heron.injection import make_injection_zero_noise
from heron.detector import AdvancedLIGOHanford
from heron.likelihood import TimeDomainLikelihoodPyTorch
from heron.models.lalsimulation import IMRPhenomPv2, IMRPhenomPv2_FakeUncertainty


def profile_likelihood_pytorch_nouncert():
    waveform = IMRPhenomPv2()
    psd_model = AdvancedLIGO()

    injections = make_injection_zero_noise(waveform=IMRPhenomPv2,
                                     injection_parameters={"m1": 35*u.solMass,
                                                          "m2": 30*u.solMass,
                                                          "gpstime": 4000,
                                                          "distance": 410 * u.megaparsec},
                                     detectors={"AdvancedLIGOHanford": "AdvancedLIGO",
                                                "AdvancedLIGOLivingston": "AdvancedLIGO"}
                                     )

    data = injections['H1']

    likelihood = TimeDomainLikelihoodPyTorch(data, psd=psd_model)
    print(likelihood.device)

    
    test_waveform = waveform.time_domain(parameters={"m1": 30*u.solMass,
                                                     "m2": 30*u.solMass,
                                                     "gpstime": 4000,
                                                     "distance": 410 * u.megaparsec}, times=data.times)

    projected_waveform = test_waveform.project(AdvancedLIGOHanford(),
                                                          ra=0, dec=0,
                                                          phi_0=0, psi=0,
                                                          iota=0)

    log_like = likelihood.log_likelihood(projected_waveform)


from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_likelihood"):

        profile_likelihood_pytorch_nouncert()

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
