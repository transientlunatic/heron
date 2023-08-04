import os
from math import floor
import click
import torch
import heron.injection

import yaml

from heron.models.lalinference import IMRPhenomPv2
from heron.models.torchbased import HeronCUDA
from heron.likelihood import CUDALikelihood, InnerProduct, CUDATimedomainLikelihood
from heron.sampling import HeronSampler

import otter
import otter.bootstrap as bt
import matplotlib.pyplot as plt

from elk.waveform import Timeseries
import scipy.signal

from nessai.flowsampler import FlowSampler

models = {
    "heron": HeronCUDA(
        datafile="training_data.h5",
        datalabel="IMR training linear",
        name="Heron IMR Non-spinning",
        device=torch.device("cuda"),
    ),
    "IMRPhenomPv2": IMRPhenomPv2(torch.device("cuda")),
}


@click.command
@click.option("--settings")
def inference(settings):
    """
    Create an inference pipeline using heron.
    """
    with open(settings, "r") as settings_file:
        settings = yaml.safe_load(settings_file)

    report = otter.Otter(
        os.path.join(settings["report"]["location"], f"{settings['name']}.html"),
        author="Heron",
        title=f"Heron PE Report | {settings['name']}",
        author_email="daniel.williams@ligo.org",
    )

    with report:
        navbar = bt.Navbar("Heron", background="navbar-dark bg-primary")
        report + navbar

    srate = settings["data"]["sample rate"]
    duration = settings["data"]["duration"]
    #times = torch.linspace(
    #    settings["data"]["start time"],
    #    settings["data"]["start time"] + duration,
    #    floor(duration * srate),
    #)

    times = {"duration": duration,
             "sample rate": srate,
             "before": 0.05,
             "after": 0.01,
             }
    
    settings["injection"].update(times)
    
    if "injection" in settings:
        click.echo("Generating injection")

        with report:
            report += "## Injection"
            report += settings["injection"]

        psd = heron.injection.psd_from_lalinference(
            settings["noise model"]["name"],
            frequencies=heron.injection.frequencies_from_times(times),
        )
        noise = heron.injection.create_noise_series(psd, times)
        signal = models[settings["injection model"]].time_domain_waveform(
            p=settings["injection"]
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

        f, ax = plt.subplots(1, 1, dpi=300)
        ax.plot(detection.times.cpu(), noise.cpu())
        ax.plot(detection.times.cpu(), detection.data.cpu())
        ax.plot(detection.times.cpu(), signal.data.cpu())
        with report:
            report += "Injected waveform"
            report += f

    heron_likelihood = CUDATimedomainLikelihood(
        models[settings["waveform"]["model"]],
        data=detection,
        detector_prefix="L1",
        generator_args=times,
        psd=psd,
    )

    click.echo(f"Created likelihood on {heron_likelihood.device}")

    nessai_model = HeronSampler(
        heron_likelihood,
        settings["priors"],
        settings["injection"],
        uncertainty=settings["waveform"]["variance"],
    )

    fp = FlowSampler(
        nessai_model,
        nlive=2000,
        maximum_uninformed=4000,
        output=settings["name"],
        resume=False,
        seed=1234,
        flow_class="FlowProposal",
        signal_handling=False,
    )

    fp.run()
