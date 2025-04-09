"""
Interfaces with the bilby library
---------------------------------

This file contains code to allow heron to interface with bilby,
including reading bilby data files.
"""

import pickle
import bilby_pipe

def read_pickle(filename):
    """
    Read a data pickle created by bilby_pipe_generation.

    Parameters
    ----------
    filename : str
       The path to the file to be read to provide strain data.

    Notes
    -----

    At present this function only returns the strain data.
    Other information is also stored in the pickle, and this could
    also be extracted, but in order to use bilby to e.g. perform
    injections we currently only need this.
    """
    output = {}
    with open(filename, "rb") as pickle_file:
        data = pickle.load(pickle_file)

    output['strain'] = {}

    for ifo in data.interferometers:
        output['strain'][ifo.name] = ifo.time_domain_strain

    return output
