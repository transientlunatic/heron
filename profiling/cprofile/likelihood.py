import numpy as np
import astropy.units as u
from heron.models.lalnoise import AdvancedLIGO
from heron.injection import make_injection_zero_noise
from heron.detector import AdvancedLIGOHanford
from heron.likelihood import TimeDomainLikelihood
from heron.models.lalsimulation import IMRPhenomPv2


def profile_likelihood():
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

    likelihood = TimeDomainLikelihood(data, psd=psd_model)

    for m1 in np.linspace(20, 50, 100):
        test_waveform = waveform.time_domain(parameters={"m1": 35*u.solMass,
                                                              "m2": 30*u.solMass,
                                                              "gpstime": 4000,
                                                              "distance": 410 * u.megaparsec}, times=data.times)

        projected_waveform = test_waveform.project(AdvancedLIGOHanford(),
                                                              ra=0, dec=0,
                                                              phi_0=0, psi=0,
                                                              iota=0)

        log_like = likelihood.log_likelihood(projected_waveform)


profile_likelihood()
