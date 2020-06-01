"""
Tools for testing and assessing heron-based models.
"""

import copy

import numpy as np
import heron
import heron.models.georgebased
import elk

import pandas as pd
from elk.catalogue import NRCatalogue
import matplotlib.pyplot as plt

import pycbc

def match(a, b, psd=None):
    
    data_a = a.pycbc()
    data_b = b.pycbc()
    
    if psd == "aligo":
        f_low = 5
        f_delta = 1./16
        flen = int(2048/ f_delta) + 1
        psd = pycbc.psd.aLIGOZeroDetHighPower(flen, f_delta, f_low)
    
        return pycbc.filter.match(data_a, data_b, psd=psd)
    
    else:
        return pycbc.filter.match(data_a, data_b)


def sample_match(generator, times, p, comparison, psd=None):
    """
    Calculate the match between the output of a model and a canonical waveform.

    Parameters
    ----------
    generator : `heron.model`
       The heron model to be tested.
    times : ndarray
       An array of times at which the model should be evaluated.
    p : dict
       A dictionary of parameters for the waveform.
    comparison : `elk.waveform`
       The waveform which should be compared to the model output
    psd : `pycbc.psd`, optional
       The PSD which should be used to evaluate the waveform match.
    """
    
    ts_data = generator.mean(p=p.copy(), times=times.copy())[0]

    return match(ts_data, comparison, psd)[0]
    

def nrcat_match(generator, catalogue):
    """
    Calculate the matches between each waveform in a given waveform catalogue 
    and the generator model.
    """
    matches = {}
    for waveform in catalogue.waveforms:
        spins = ["spin 1x", "spin 1y", "spin 1z", "spin 2x", "spin 2y", "spin 2z"]
        pars = dict(zip(spins, waveform.spins))
        pars['mass ratio'] = waveform.mass_ratio

        nr_data = waveform.timeseries(total_mass=60, f_low=70, t_max=0.02, t_min=-0.015)
        
        matches[waveform.tag] = heron.testing.sample_match(generator,
                                                           nr_data[0].times,
                                                           pars,
                                                           nr_data[0])
    return matches

