"""
Code to create injections using the various models supported by heron.
"""

import logging

import click

import numpy as np

import astropy.units as u

from heron.models.lalsimulation import SEOBNRv3, IMRPhenomPv2
from heron.models.lalnoise import KNOWN_PSDS
from heron.likelihood import TimeDomainLikelihood
from heron.detector import KNOWN_IFOS
from heron.utils import load_yaml

logger = logging.getLogger("heron.injection")


def make_injection(
    waveform=IMRPhenomPv2,
    injection_parameters={},
    times=None,
    detectors=None,
    framefile=None,
):

    parameters = {"ra": 0, "dec": 0, "psi": 0, "theta_jn": 0, "phase": 0}
    parameters.update(injection_parameters)

    waveform = waveform()

    if times is None:
        times = np.linspace(-0.5, 0.1, int(0.6 * 4096))
    waveform = waveform.time_domain(
        parameters,
        times=times,
    )

    injections = {}
    for detector, psd_model in detectors.items():
        logger.info(f"Making injection for {detector}")
        psd_model = KNOWN_PSDS[psd_model]()
        detector = KNOWN_IFOS[detector]()
        data = psd_model.time_series(times)

        channel = f"{detector.abbreviation}:Injection"
        injection = data + waveform.project(detector)
        injection.channel = channel
        injections[detector.abbreviation] = injection
        likelihood = TimeDomainLikelihood(injection, psd=psd_model)
        snr = likelihood.snr(waveform.project(detector))

        logger.info(f"Optimal Filter SNR: {snr}")

        if framefile:
            filename = f"{detector.abbreviation}_{framefile}.gwf"
            logger.info(f"Saving framefile to {filename}")
            injection.write(filename, format="gwf")

    return injections


def injection_parameters_add_units(parameters):
    UNITS = {"luminosity_distance": u.megaparsec, "m1": u.solMass, "m2": u.solMass}

    for parameter, value in parameters.items():
        if not isinstance(value, u.Quantity):
            parameters[parameter] = value * UNITS[parameter]
    return parameters


@click.command()
@click.option("--settings")
def injection(settings):
    settings = load_yaml(settings)

    if "logging" in settings:

        level = settings.get("logging", {}).get("level", "warning")

        LOGGER_LEVELS = {
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "warning": logging.WARNING,
        }

        logging.basicConfig(level=LOGGER_LEVELS[level])

    settings = settings["injection"]
    parameters = injection_parameters_add_units(settings["parameters"])

    detector_dict = {
        settings["interferometers"][ifo]: settings["psds"][ifo]
        for ifo in settings["interferometers"]
    }
    injections = make_injection(
        waveform=IMRPhenomPv2,
        injection_parameters=parameters,
        detectors=detector_dict,
        framefile="injection",
    )
    data = injections
