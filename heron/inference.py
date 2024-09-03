"""
Logic to allow heron to complete signal inference
"""

import logging

import click

from .types import TimeSeries
import astropy.units as u

from nessai.flowsampler import FlowSampler

from heron.detector import KNOWN_IFOS
from heron.models.lalnoise import KNOWN_PSDS
from heron.likelihood import TimeDomainLikelihood, MultiDetector
import heron.priors

from heron.sampling import NessaiSampler

from heron.injection import make_injection, injection_parameters_add_units
from heron.models.lalsimulation import (
    SEOBNRv3,
    IMRPhenomPv2,
    IMRPhenomPv2_FakeUncertainty,
)
from heron.utils import load_yaml

logger = logging.getLogger("heron.inference")

KNOWN_LIKELIHOODS = {
    "TimeDomainLikelihood": TimeDomainLikelihood,
}
KNOWN_WAVEFORMS = {
    "IMRPhenomPv2": IMRPhenomPv2,
}


def parse_dict(settings):
    # Inference settings are in the `settings` part of the dict.
    other_settings = settings.copy()
    settings = settings["inference"]

    # Load interferometers
    ifos = {}
    psds = {}
    for name, ifo in settings["interferometers"].items():
        ifos[name] = KNOWN_IFOS[ifo]
        psds[name] = KNOWN_PSDS[settings["psds"][name]]

    settings["interferometers"] = ifos
    settings["psds"] = psds

    return settings, other_settings


def heron_inference(settings):

    settings = load_yaml(settings)
    settings, other_settings = parse_dict(settings)

    if "logging" in other_settings:

        level = other_settings.get("logging", {}).get("level", "warning")

        LOGGER_LEVELS = {
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "warning": logging.WARNING,
        }

        logging.basicConfig(level=LOGGER_LEVELS[level])
        logging.getLogger("heron.likelihood").setLevel(LOGGER_LEVELS[level])
        logging.getLogger("heron.likelihood.TimeDomainLikelihood").setLevel(
            LOGGER_LEVELS[level]
        )

    import matplotlib

    matplotlib.use("agg")

    data = {}

    if "data files" in settings.get("data", {}):
        # Load frame files from disk
        for ifo in settings["interferometers"]:
            print(f"Loading {ifo} data")
            logger.info(
                f"Loading {ifo} data from "
                f"{settings['data']['data files'][ifo]}/{settings['data']['channels'][ifo]}"
            )
            data[ifo] = TimeSeries.read(
                source=settings["data"]["data files"][ifo],
                channel=settings["data"]["channels"][ifo],
                format="gwf",
            )
    #elif "injection" in other_settings:
    #    pass

    # Make Likelihood
    if len(settings["interferometers"]) > 1:
        likelihoods = []
        print("Creating likelihoods")
        waveform_model = KNOWN_WAVEFORMS[settings["waveform"]["model"]]()
        for ifo in settings["interferometers"]:
            print(f"\t {ifo}")
            likelihoods.append(
                KNOWN_LIKELIHOODS[settings.get("likelihood").get("function")](
                    data[ifo],
                    psd=settings["psds"][ifo](),
                    waveform=waveform_model,
                    detector=settings["interferometers"][ifo](),
                    fixed_parameters=settings["fixed_parameters"],
                    timing_basis=settings["likelihood"].get(
                        "timing basis", ["H1", "L1"]
                    ),
                )
            )
        likelihood = MultiDetector(*likelihoods)

    priors = heron.priors.PriorDict()
    priors.from_dictionary(settings["priors"])

    if settings["sampler"]["sampler"] == "nessai":
        nessai_model = NessaiSampler(
            likelihood,
            priors,
            injection_parameters_add_units(other_settings["injection"]["parameters"]),
        )

        fp = FlowSampler(
            nessai_model,
            nlive=settings.get("sampler", {}).get("live points", 1000),
            maximum_uninformed=settings.get("sampler", {}).get(
                "maximum uninformed", 2000
            ),
            output=settings["name"],
            resume=settings.get("sampler", {}).get("resume", True),
            checkpointing=settings.get("sampler", {}).get("checkpointing", True),
            checkpoint_interval=settings.get("sampler", {}).get(
                "checkpointing interval", 3600
            ),
            logging_interval=settings.get("sampler", {}).get("logging interval", 10),
            log_on_iteration=settings.get("sampler", {}).get("log on iteration", True),
            seed=settings.get("sampler", {}).get("seed", 1234),
            flow_class="GWFlowProposal",
            signal_handling=True,
        )

        fp.run()


@click.command
@click.option("--settings")
def inference(settings):
    heron_inference(settings)
