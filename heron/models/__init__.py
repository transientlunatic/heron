import torch
from lal import antenna


class Model(object):
    """
    This is the factory class for statistical models used for waveform generation.

    A model class must expose the following methods:

    - `distribution` : produce a distribution of waveforms at a given point in the parameter space
    - `mean` : produce a mean waveform at a given point in the parameter space
    - `train` : provide an interface for training the model
    """

    def __init__(self):
        """
        Make some default initialisations.
        These don't really mean anything, but it's important that the variables exist.
        This __init__ function ought to be usable as a template for other functions, or run from the superclass.
        """

        # Need to explicitly set the number of dimensions of the training data.

        # Need to create a lookup table between the columns in the model
        # and their location in the training data
        self.columns = {0: "time", 1: "second quantity"}
        self.c_ind = {j: i for i, j in self.columns.items()}

        # The training data contains both the x and the y data, so we need
        # to specify which are parameters.
        # It's worth noting that right now, time isn't a parameter...
        self.parameters = ("second quantity",)

    def _process_inputs(self, times, p):
        """
        Apply regularisations and normalisations to any input point dictionary.

        Parameters
        ----------
        times: list, array-like
           An array of time stamps.
        p : dict
           A dictionary of the input locations
        """

        # The default implementation of this method just passes the data straight through.

        return times, p

    def _generate_eval_matrix(self, p, times):
        """
        Create the matrix of parameter points at which to evaluate the model.
        """

        times, p = self._process_inputs(times, p)
        nt = len(times)
        points = torch.ones((nt, self.x_dimensions))
        for parameter in self.parameters:
            if parameter == b"time":
                points[:, self.c_ind[b"time"]] = times
            else:
                if parameter.decode("ascii") in p.keys():
                    value = p[parameter.decode("ascii")]
                else:
                    value = 0.0
                points[:, self.c_ind[parameter]] *= value

        return points

    def _get_antenna_response(self, detector, ra, dec, psi, time):
        """
        Get the antenna responses for a given detector.

        Parameters
        ----------
        detectors : str
           The detector abbreviation, for example ``H1`` for the 4-km
           detector at LIGO Hanford Observatory.
        ra : float
           The right-ascension of the source, in radians.
        dec : float
           The declination of the source, in radians.
        psi : float
           The polarisation angle, in radians.

        time : float, or array of floats
           The GPS time, or an array of GPS times, at which
           the response should be evaluated.

        Returns
        -------
        plus : float, or array
           The 'plus' component of the response function.
        cross : float, or array
           The 'cross' component of the response function.
        """
        responses = antenna.AntennaResponse(detector, ra, dec, psi=psi, times=time)
        return responses
