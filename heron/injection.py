"""
Code to create injections into data.
"""
import logging
import os
import yaml

import scipy.signal

import click
import otter
import otter.bootstrap as bt

from heron.types import PSD
from heron.utils import noise_psd

import lalsimulation
import torch

import matplotlib.pyplot as plt
import numpy as np

from asimov.utils import update
from elk.waveform import Timeseries
from gwpy.timeseries import TimeSeries as gwpyTimeSeries

from heron.models.lalinference import IMRPhenomPv2
from heron.models.torchbased import HeronCUDA

models = {
    "heron": HeronCUDA(
        datafile="training_data.h5",
        datalabel="IMR training linear",
        name="Heron IMR Non-spinning",
        device=torch.device("cuda"),
    ),
    "IMRPhenomPv2": IMRPhenomPv2(torch.device("cuda")),
}


def psd_from_lalinference(name: str, frequencies, lower_frequency: float = 20):
    """Create a PSD object from a LALInference PSD class

    Parameters
    ----------
    name : str
       The name of the PSD from LALInference.
    frequencies : array-like
       The frequencies at which the PSD should be evaluated.
    lower_frequency : float
       The lowest frequency at which the PSD should be evaluated.

    Notes
    -----
    Frequently the PSD will either be invalid below a given frequency,
    or will be unusable. You should set a lower_frequency in order to
    account for this.
    """
    if name in dir(lalsimulation):
        psd_func = getattr(lalsimulation, name)
    else:
        raise NotImplementedError(
            "This noise source is not implemented in LALSimulation"
        )

    masked_psd = (
        lambda f: psd_func(f) if f >= lower_frequency else psd_func(lower_frequency)
    )
    psd = torch.tensor([masked_psd(float(f)) for f in frequencies], dtype=torch.float64)
    psd = PSD(data=psd, frequencies=frequencies)
    return psd


def frequencies_from_times(times):
    """Get the frequency axies which corresponds to a given time axis."""
    times = torch.linspace(
        -times["before"],
        times["after"],
        int(times["sample rate"] * (times["before"] + times["after"])),
    )
    dt = times[1] - times[0]
    frequencies = torch.arange((len(times) + 1) // 2) / (dt * len(times))
    return frequencies


def create_noise_series(psd: PSD, times, device="cuda"):
    """Create a timeseries filled with noise from a specific PSD.
    Parameters
    ----------
    psd : `heron:PSD`
       The power spectral density of the noise to be generated in the time
       series.
    times : array-like
       The time axis of the timeseries.
    device : str or torch.device
       The device which the noise series should be stored in.
    """
    frequencies = frequencies_from_times(times)
    noise = torch.tensor(
        noise_psd(
            int(times["sample rate"] * (times["before"] + times["after"])),
            frequencies=frequencies,
            psd=psd.data,
        ),
        device=device,
    )
    return noise


@click.command()
@click.option("--settings")
def injection(settings):

    logger = logging.getLogger("heron.injection")

    with open(settings, "r") as settings_file:
        settings = yaml.safe_load(settings_file)

    defaults = {
        "interferometers": ["H1", "L1"],
        "injection": {
            "noise model": {
                "H1": "SimNoisePSDaLIGOZeroDetHighPower",
                "L1": "SimNoisePSDaLIGOZeroDetHighPower",
            }
        },
    }
    settings = update(defaults, settings)

    report = otter.Otter(
        os.path.join(settings["report"]["location"], f"{settings['name']}-injection.html"),
        author="Heron",
        title=f"Heron Injection Report | {settings['name']}",
        author_email="daniel.williams@ligo.org",
    )

    with report:
        navbar = bt.Navbar("Heron", background="navbar-dark bg-primary")
        report + navbar

    srate = settings["likelihood"]["sample rate"]
    duration = settings["data"]["segment length"]

    times = {
        "duration": duration,
        "sample rate": srate,
        "before": 0.05,
        "after": 0.05,
    }

    settings["injection"]["parameters"].update(times)

    injection = {}
    for ifo in settings["interferometers"]:
        print("Making injections")
        logger.info(f"Generating injection for {ifo}")

        settings["injection"]["parameters"]["detector"] = ifo

        with report:
            report += f"## {ifo} Injection"
            report += settings["injection"]

        signal = models[settings["injection"]["injection model"]].time_domain_waveform(
            p=settings["injection"]["parameters"],
        )

        if "noise model" in settings["injection"]:
            logger.info(f"Using noise model {settings['injection']['noise model'][ifo]}")
            psd = psd_from_lalinference(
                settings["injection"]["noise model"][ifo],
                frequencies=frequencies_from_times(times),
            )
            noise = create_noise_series(psd, times)

            f, ax = plt.subplots(1, 1, dpi=300)
            ax.plot(psd.frequencies, psd.data)

            with report:
                report += "### Noise PSD"
                report += f

            directory = os.path.join(settings.get('rundir', os.getcwd()), "psds")
            os.makedirs(directory, exist_ok=True)
            output_psd = np.vstack([psd.frequencies, psd.data])
            np.savetxt(os.path.join(directory, f"{ifo}.dat"), output_psd.T)


            data = signal.data + noise
            data.detector = ifo
            snr = torch.sum(signal.data / noise)

            print("SNR", snr)

        else:
            logger.info("No noise model set so creating noise-free injections")
            data = signal.data
            data.detector = ifo
            noise = torch.zeros(len(data))
            snr = 0

        detection = Timeseries(data=data, times=signal.times)

        sos = scipy.signal.butter(
            10,
            20,
            "hp",
            fs=float(1 / (detection.times[1] - detection.times[0])),
            output="sos",
        )

        logger.info("Applying butterworth filter to the data")
        detection.data = torch.tensor(
            scipy.signal.sosfilt(sos, detection.data.cpu()),
            device=detection.data.device,
        )
        detection.detector = ifo

        f, ax = plt.subplots(1, 1, dpi=300)
        ax.plot(detection.times.cpu(), noise.cpu())
        ax.plot(detection.times.cpu(), detection.data.cpu())
        ax.plot(detection.times.cpu(), signal.data.cpu())
        with report:
            report += "Injected waveform"
            report += f

            report += f"SNR: {snr}"
        injection[ifo] = detection

        gwpy_ts = gwpyTimeSeries(data=detection.data.cpu(),
                                 times=detection.times.cpu(),
                                 channel=f"{ifo}:Injection",
                                 name=f"{ifo}:Injection",
                                 dtype=np.float64)

        directory = os.path.join(settings.get('rundir', os.getcwd()), "frames")
        os.makedirs(directory, exist_ok=True)
        gwpy_ts.write(os.path.join(directory, f"{ifo}-injection.gwf"), format="gwf")

        # directory = os.path.join(settings.get('rundir', os.getcwd()), "cache")
        # os.makedirs(directory, exist_ok=True)
        # with open(os.path.join(directory, f"{ifo}.cache"), "w") as f:
        #    absolute = os.path.abspath(f"{ifo}-injection.gwf")
        #    f.write(f"{ifo[0]}\t{ifo}_Injection\t{int(epoch-segment_length/2)}\t{segment_length}\tfile://localhost{absolute}\n")
