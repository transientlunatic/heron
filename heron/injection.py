"""
Code to create injections using the various models supported by heron.
"""
import numpy as np

import astropy.units as u

from heron.models.lalsimulation import SEOBNRv3, IMRPhenomPv2
from heron.models.lalnoise import AdvancedLIGO


def make_injection(waveform=IMRPhenomPv2,
                   injection_parameters={"mass ratio": 1.0,
                                         "ra": 1,
                                         "dec": 1,
                                         "psi": 1,
                                         "theta_jn": 0,
                                         "phase": 0,
                                         "total mass": 30 * u.solMass,
                                         "distance": 500 * u.Mpc},
                   detectors=None,
                   framefile=None):

    waveform = waveform()

    
    times = np.linspace(-0.5, 0.05, int(0.555*4096))
    waveform = waveform.time_domain(
        injection_parameters,
        times=times,
    )

    injections = {}
    for detector, psd_model in detectors.items():
        psd_model = psd_model()
        data = psd_model.time_domain(times)
        injection = data + waveform.project(detector())
        injections[detector.abbreviation] = injection

        if framefile:
            injection.write(f"{detector.abbreviation}_{framefile}.gwf")

    return injections

