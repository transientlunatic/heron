import os

import click
import torch

import yaml
import numpy as np

from heron.models.lalinference import IMRPhenomPv2
from heron.models.torchbased import HeronCUDA, device
from heron.likelihood import CUDALikelihood, InnerProduct, CUDATimedomainLikelihood
from heron.sampling import HeronSampler
from heron.injection import models
from heron.types import PSD

from heron.gw import StrainData

import otter
import otter.bootstrap as bt
import matplotlib.pyplot as plt

from gwpy.timeseries import TimeSeries as gwpyTimeSeries
from elk.waveform import Timeseries

from nessai.flowsampler import FlowSampler

import logging
logger = logging.getLogger("heron.inference")

@click.command
@click.option("--settings")
def inference(settings):
    """
    Create an inference pipeline using heron.
    """
    with open(settings, "r") as settings_file:
        settings = yaml.safe_load(settings_file)


    report = otter.Otter(
        os.path.join(settings["report"]["location"], f"{settings['name']}-inference.html"),
        author="Heron",
        title=f"Heron Inference Report | {settings['name']}",
        author_email="daniel.williams@ligo.org",
    )

    with report:
        navbar = bt.Navbar("Heron", background="navbar-dark bg-primary")
        report + navbar

    data = {}
    for ifo in settings['interferometers']:
        frame_data = gwpyTimeSeries.read(settings['data']['data files'][ifo],
                                     settings['data']['channels'][ifo])
        data[ifo] = Timeseries(data=torch.tensor(frame_data.data, device=device, dtype=torch.float64),
                               times=torch.tensor(frame_data.times, device=device))

    psds = {}
    for ifo in settings['interferometers']:
        psd = np.genfromtxt(settings['psds'][ifo])
        psds[ifo] = PSD(data=psd[:,1], frequencies=psd[:,0])

    srate = settings["likelihood"]["sample rate"]
    duration = settings["data"]["segment length"]

    times = {"duration": duration,
             "sample rate": srate,
             "before": 0.05,
             "after": 0.05,
             }

    likelihood = {}
    for ifo in settings['interferometers']:
        likelihood[ifo] = CUDATimedomainLikelihood(
            models[settings["waveform"]["model"]],
            data=data[ifo],
            detector_prefix=ifo,
            generator_args=times,
            psd=psds[ifo],
        )
        
    def joint_likelihood(p, model_var):
        likes = [l(p, model_var) for l in likelihood.values()]
        return sum(likes)

    logger.info(f"Created likelihood on {likelihood[settings['interferometers'][0]].device}")

    nessai_model = HeronSampler(
        joint_likelihood,
        settings["priors"],
        settings["injection"]["parameters"],
        uncertainty=settings["waveform"]["variance"],
    )

    fp = FlowSampler(
        nessai_model,
        nlive=1000,
        maximum_uninformed=2000,
        output=settings["name"],
        resume=True,
        checkpointing=True,
        checkpoint_interval=30*60,
        logging_interval=10,
        log_on_iteration=True,
        seed=1234,
        flow_class="GWFlowProposal",
        signal_handling=True,
    )

    fp.run()
