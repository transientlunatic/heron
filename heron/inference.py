"""
Logic to allow heron to complete signal inference
"""

import logging
import os

import click

from .datatypes import TimeSeries
import astropy.units as u

from nessai.flowsampler import FlowSampler

from heron.detector import KNOWN_IFOS
from heron.models.lalnoise import KNOWN_PSDS
from heron.likelihood import TimeDomainLikelihood, MultiDetector, TimeDomainLikelihoodModelUncertainty
import heron.priors

from heron.sampling import NessaiSampler, AspireSampler

from heron.injection import make_injection, injection_parameters_add_units
from heron.models.lalsimulation import (
    SEOBNRv3,
    IMRPhenomPv2,
    IMRPhenomPv2_FakeUncertainty,
)
from heron.models.gpytorch import HeronNonSpinningApproximantMatern
from heron.utils import load_yaml

import otter

logger = logging.getLogger("heron.inference")

KNOWN_LIKELIHOODS = {
    "TimeDomainLikelihood": TimeDomainLikelihood,
    "TimeDomainUncertaintyLikelihood": TimeDomainLikelihoodModelUncertainty,
}
KNOWN_WAVEFORMS = {
    "IMRPhenomPv2": IMRPhenomPv2,
    "IMRPhenomPv2_FakeUncertainty": IMRPhenomPv2_FakeUncertainty,
    "HeronGPR": HeronNonSpinningApproximantMatern,
}


def _build_multidetector_likelihood(settings, data):
    """Build a MultiDetector likelihood from processed settings and loaded data.

    Parameters
    ----------
    settings : dict
        Parsed inference settings (after ``parse_dict``).
    data : dict
        Mapping of IFO name -> loaded TimeSeries.

    Returns
    -------
    MultiDetector
    """
    waveform_name = settings["waveform"]["model"]
    waveform_cls = KNOWN_WAVEFORMS[waveform_name]
    if "checkpoint" in settings.get("waveform", {}):
        waveform_model = waveform_cls.from_checkpoint(settings["waveform"]["checkpoint"])
    else:
        waveform_model = waveform_cls()
    likelihoods = []
    for ifo in settings["interferometers"]:
        likelihoods.append(
            KNOWN_LIKELIHOODS[settings.get("likelihood").get("function")](
                data[ifo],
                psd=settings["psds"][ifo](),
                waveform=waveform_model,
                detector=settings["interferometers"][ifo](),
                fixed_parameters=settings.get("fixed_parameters", {}),
                timing_basis=settings["likelihood"].get("timing basis", ["H1", "L1"]),
            )
        )
    return MultiDetector(*likelihoods)


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
    webdir = settings['pages directory']
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
    # Disable LaTeX rendering to avoid missing font issues
    matplotlib.rcParams['text.usetex'] = False

    data = {}

    report = otter.Otter(os.path.join(
        webdir,
        "inference.html"),
        author="Heron",
        title="Heron Inference"
    )

    if "data files" in settings.get("data", {}):
        # Load frame files from disk
        with report:
            report += "# Data"
        start = settings['event time'] - \
            settings['segment length'] + settings['after merger']
        end = settings['event time'] + settings['after merger']

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
                start=start,
                end=end,
            )
            with report:
                report += f"## {ifo}"
                f = data[ifo].plot()
                f.savefig(os.path.join(
                    other_settings.get('pages directory', 'pages'),
                    f"{ifo}_data.png"
                ))
                report += f

            if data[ifo].sample_rate != settings['likelihood']['sampling rate']:
                logger.info(
                    "Resampling the data to the likelihood sampling rate")
                data[ifo] = data[ifo].resample(
                    settings['likelihood']['sampling rate'])
    # elif "injection" in other_settings:
    #    pass

    # Make Likelihood
    if len(settings["interferometers"]) > 1:
        print("Creating likelihoods")
        likelihood = _build_multidetector_likelihood(settings, data)

    priors = heron.priors.PriorDict()
    priors.from_dictionary(settings["priors"])

    if settings["sampler"]["sampler"] == "naive":
        import numpy as np
        import matplotlib.pyplot as plt
        # Just draw 100 points across mass ratio space
        #prior_points = np.linspace(50, 70, 100)
        #prior_points = np.linspace(50, 150, 100)
        posterior = []
        prior_points = np.linspace(settings["sampler"]["naive range"][0],
            settings["sampler"]["naive range"][1],
            100)
        for mass in prior_points:
            parameters = {
            "total_mass": 60,#mass * u.solMass,
            "mass_ratio": 1.0,
            "luminosity_distance": 100,
            "gpstime": 4000,
            "ra": 1.0,
            "dec": 0.3}

            parameters[settings["sampler"]["naive parameter"]] = mass
            parameters = injection_parameters_add_units(parameters)
            posterior.append(likelihood(parameters))

        posterior = np.array(posterior)
        print(posterior)
        f, ax = plt.subplots(1,1, dpi=300)
        ax.plot(prior_points, posterior)
        f.savefig(os.path.join(webdir, "posterior.png"))

    elif settings["sampler"]["sampler"] == "nessai":
        nessai_model = NessaiSampler(
            likelihood,
            priors,
            injection_parameters_add_units(
                other_settings["injection"]["parameters"]),
        )

        fp = FlowSampler(
            nessai_model,
            nlive=settings.get("sampler", {}).get("live points", 1000),
            maximum_uninformed=settings.get("sampler", {}).get(
                "maximum uninformed", 2000
            ),
            output=settings["name"],
            resume=settings.get("sampler", {}).get("resume", True),
            checkpointing=settings.get(
                "sampler", {}).get("checkpointing", True),
            checkpoint_interval=settings.get("sampler", {}).get(
                "checkpointing interval", 3600
            ),
            logging_interval=settings.get(
                "sampler", {}).get("logging interval", 10),
            log_on_iteration=settings.get(
                "sampler", {}).get("log on iteration", True),
            seed=settings.get("sampler", {}).get("seed", 1234),
            flow_class=settings.get("sampler", {}).get(
                "flow class", "GWFlowProposal"),
            signal_handling=True,
        )

        fp.run()


@click.command
@click.option("--settings")
def inference(settings):
    heron_inference(settings)


def heron_aspire_inference(settings):
    """Run multi-stage Heron inference using the aspire SMC sampler.

    Optionally seeds the SMC from an upstream bilby (frequency-domain)
    result via the ``upstream_bilby`` key in the config.
    """
    import numpy as np

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

    import matplotlib
    matplotlib.use("agg")
    matplotlib.rcParams['text.usetex'] = False

    # Load data
    data = {}
    if "data files" in settings.get("data", {}):
        start = settings['event time'] - settings['segment length'] + settings['after merger']
        end = settings['event time'] + settings['after merger']
        for ifo in settings["interferometers"]:
            logger.info(f"Loading {ifo} data")
            data[ifo] = TimeSeries.read(
                source=settings["data"]["data files"][ifo],
                channel=settings["data"]["channels"][ifo],
                format="gwf",
                start=start,
                end=end,
            )
            if data[ifo].sample_rate != settings['likelihood']['sampling rate']:
                data[ifo] = data[ifo].resample(settings['likelihood']['sampling rate'])

    if not data:
        raise RuntimeError(
            "No data was loaded for heron aspire inference. "
            "Ensure 'data files' is set under the 'data' key in the config."
        )
    likelihood = _build_multidetector_likelihood(settings, data)

    priors = heron.priors.PriorDict()
    priors.from_dictionary(settings["priors"])

    aspire_cfg = settings.get("sampler", {}).get("aspire", {})
    sampler = AspireSampler(
        likelihood=likelihood,
        priors=priors,
        base_p=injection_parameters_add_units(
            other_settings.get("injection", {}).get("parameters", {})
        ),
        initial_result=other_settings.get("upstream_bilby"),
    )

    posterior, history = sampler.sample(
        n_samples=aspire_cfg.get("n_samples", 500),
        n_epochs=aspire_cfg.get("n_epochs", 30),
        sampler=aspire_cfg.get("sampler", "smc"),
        sampler_kwargs=aspire_cfg.get("sampler_kwargs", {}),
    )

    out_dir = settings["name"]
    os.makedirs(out_dir, exist_ok=True)
    posterior.save(os.path.join(out_dir, "aspire_result.h5"), path="posterior")
    logger.info(f"Aspire posterior saved to {out_dir}/aspire_result.h5")


@click.command
@click.option("--settings")
def aspire(settings):
    heron_aspire_inference(settings)
