
import numpy as np

class Model(object):
    """
    This is the factory class for statistical models used for waveform generation.

    A model class must expose the following methods:
    - `distribution` : produce a distribution of waveforms at a given point in the parameter space
    - `mean` : produce a mean waveform at a given point in the parameter space
    - `train` : provide an interface for training the model
    """

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
