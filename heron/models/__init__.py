import json
import numpy as np

class Model(object):
    """
    This is the factory class for statistical models used for waveform generation.

    A model class must expose the followi*** TODO Email about Away Dayc_ind = {j:i for i,j in columns.items()}c_ind = {j:i for i,j in columns.items()}ng methods:
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
        # TODO: Decide if this is actually required.
        self.x_dimensions = 2

        # Need to create a lookup table between the columns in the model
        # and their location in the training data
        self.columns = {0: "time",
                        1: "second quantity"}
        self.c_ind = {j:i for i,j in self.columns.items()}

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
        points = np.ones((nt, self.x_dimensions))
        points[:,self.c_ind['time']] = times

        for parameter in self.parameters:
            if parameter in p.keys():
                value = p[parameter]
            else:
                value = 0.0
                
            points[:, self.c_ind[parameter]] *= value

        return points
    
    pass


class ReducedModel(Model):
    """Construct a model using a reduced basis."""

    def __init__(self, data):
        with open(data, "r") as fp:
            data = json.load(fp)

        self.basis = np.array(data['vectors'])
        self.abscissa = np.array(data['abscissa'])
        self.coeffs = np.array(data['coefficients'])
        self.locs = np.array(data['locations'])
