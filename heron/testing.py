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

def outsample_retrain(generator, catalogue = NRCatalogue('GeorgiaTech')):
    """
    Calculate the out-sample matches between a given heron model 
    and an NR catalogue.

    Parameters
    ----------
    generator : `heron.model`
       The heron model which is used to generate the waveforms.
    catalogue : `elk.catalogue`
       The waveform catalogue to prepare matches against.
    """

    results = {}
    for waveform in catalogue.waveforms:
        
        try:

            spins = ["spin 1x", "spin 1y", "spin 1z", "spin 2x", "spin 2y", "spin 2z"]
            pars = dict(zip(spins, waveform.spins))
            pars['mass ratio'] = waveform.mass_ratio

            waveform_nr = waveform.timeseries(total_mass=60, f_low=70, t_max=0.02, t_min=-0.015) 
            times = waveform_nr[0].times

            new_catalogue = copy.copy(catalogue)
            new_catalogue.waveforms = new_catalogue.waveforms[new_catalogue.waveforms != waveform]
            training_data = new_catalogue.create_training_data(total_mass = 60, f_min=70, tmax=0.02, tmin=-0.015, sample_rate=1024)

            training_data[:,generator.c_ind['time']] *= generator.time_factor
            training_data[:,generator.c_ind['mass ratio']] = np.log(training_data[:,generator.c_ind['mass ratio']])
            training_data[:,generator.c_ind['h+']] *= generator.strain_input_factor
            training_data[:,generator.c_ind['hx']] *= generator.strain_input_factor

            new_model = copy.deepcopy(generator)
            new_model.training_data = training_data

            new_model.build(0.0, 0.0, 1e-6)

            heron.models.georgebased.train(new_model, max_iter=10000)

            waveform_gp = new_model.mean(pars.copy(), times=times.copy())


            time_range = [times[0], times[-1], len(times)]
            #
            # waveform_imr = imr_cat.waveform(time_range=time_range, p=pars.copy())
            # time_range[0] /=6
            # time_range[1] /=6
            # waveform_seo = seo3_cat.waveform(time_range=time_range, p=pars.copy())

            # waveform_seo[0].times *= 6
            # waveform_seo[0].data *=6
            # waveform_seo[0].dt = waveform_seo[0].times[1] - waveform_seo[0].times[0]

            #waveform_nrsur = nrsur_cat.waveform(time_range=time_range, p=pars.copy())
            #waveform_nrsur[0].times *= 6
            #waveform_nrsur[0].data *=6
            #waveform_nrsur[0].dt = waveform_nrsur[0].times[1] - waveform_nrsur[0].times[0]

            #td_pad = np.pad(waveform_gp[0].data, int((len(waveform_imr[0].data) - len(waveform_gp[0].data))/2),
            #                "constant", constant_values=0)
            #waveform_gp[0].data = td_pad
            #waveform_gp[0].times = waveform_imr[0].times

            #nr_pad = np.pad(waveform_nr[0].data, int((len(waveform_imr[0].data) - len(waveform_nr[0].data))/2), 
            #                "constant", constant_values=0)
            #waveform_nr[0].data = nr_pad
            #waveform_nr[0].times = waveform_imr[0].times



            nr_gp_match = match(waveform_nr[0], waveform_gp[0])
            # imr_match = match(waveform_imr[0], waveform_nr[0])
            # seo_match = match(waveform_seo[0], waveform_nr[0])
            # #seo_match = match(waveform_nrsur[0], nr_timeseries[0])
            # #print("\tNR match with NRSUR: {}".format(nrsur_match[0]))
            # imr_gp_match = match(waveform_imr[0], waveform_gp[0])
            # seo_gp_match = match(waveform_seo[0], waveform_gp[0])
            #seo_gp_match = match(waveform_nrsur[0], waveform_gp[0])
            #print("\tGP match with NRSUR: {}".format(nrsur_gp_match[0]))
            results[waveform.tag] ={
                #"waveform": row['tag'], 
                #"parameters": p, 
                "nr gp match": nr_gp_match[0], 
                # "imr gp match": imr_gp_match[0], 
                # "seo gp match": seo_gp_match[0],
                # #"nrsur gp match": nrsur_gp_match[0],
                # "imr nr match": imr_match[0], 
                # "seo nr match": seo_match[0],
                #"nrsur nr match": nrsur_match[0],
                           }


            #results[waveform.tag] = match(waveform_gp[0], waveform_nr[0])[0]

            data = pd.DataFrame.from_dict(results, orient='index')
            data.to_json("outsample-tests-retrain.json")
        except:
            results[waveform.tag] = np.nan

    return results
