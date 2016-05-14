import numpy as np
class Kernel():
    """
    A generic factory for Kernel classes.
    """
    name = "Generic kernel"
    ndim = 1
    def set_hyperparameters(self, hypers):
        self.hyper = hypers
    
    def distance(self, data1, data2, hypers = None):
        """
        Calculate the squared distance to the point in parameter space.
        """
        if not hypers:
            hypers = np.ones(self.ndim)
        elif not hasattr(hypers, '__getitem__'):
            hypers = np.ones(self.ndim)*hypers
        hypers = np.squeeze(hypers)
        hypers = np.exp(hypers)
        if data1.ndim >= 2:
            d = np.zeros((data1.shape[0], data2.shape[0]))
            for i in xrange(data1.shape[-1]):
                d += hypers[i]*(np.atleast_2d(data1[...,i]).T - np.atleast_2d(data2[...,i]))**2
        else:
            data1, data2 = np.atleast_2d(data1), np.atleast_2d(data2)
            d = hypers*(data1.T - data2)**2
            
        return d
    
    def matrix(self, data1, data2):
        """
        Produce a gram matrix based off this kernel.
        
        Parameters
        ----------
        data : ndarray
            An array of data (x)
        
        Returns
        -------
        covar : ndarray 
            A covariance matrix.
        """
        #data1 = np.atleast_2d(data1)
        #data2 = np.atleast_2d(data2)
        #data1, data2 = np.expand_dims(data1,0), np.expand_dims(data2,0)
        return self.function(data1, data2)

class SquaredExponential(Kernel):
    """
    An implementation of the squared-exponential kernel.
    """
    name = "Squared exponential kernel"
    hyper = [1.0]
    def __init__(self,ndim=1, amplitude=100, width=15):
        self.ndim = ndim
        self.hyper = [amplitude, width]
        
    def set_hyperparameters(self, hypers):
        self.hyper = [hypers[0], [hypers[1:]]]
        
    def function(self, data1, data2):
        """
        The functional form of the kernel.
        """
        d = self.distance(data1, data2, self.hyper[1])
        return self.hyper[0]**2*np.exp(-np.abs(d))

    def gradient(self, data1, data2):
        """
        Calculate the graient of the kernel.
        """
        gradients = np.zeros(len(self.hyper[1])+1)
        # First calculate the gradient wrt to the scaling term
        gradients[0] = np.exp(-np.abs(d))
        # Now calculate the gradient wrt all of the width factors
        for i in xrange(len(self.hyper[1])):
            # Set the ith hyperparameter to equal 1
            th = np.copy(self.hyper[1])
            th[i] = 1
            d = self.distance(data1, data2, th )
            gradients[i+1] = self.hyper[0] * np.exp(-np.abs(d))
        return gradients
        

from scipy.special import kv
class Matern(Kernel):
    """
    An implementation of the Matern Kernel.
    """
    name = "Matern"
    order = 1.5
    def __init__(self,  order=1.5, amplitude=100, width=15):
        self.hyper = [amplitude, width]
    def function(self, data1, data2):
        d = np.abs(self.distance(data1, data2))
        K = self.hyper[0]*d**self.order * kv(self.order, self.hyper[1]*d)
        K[np.isnan(K)]=0
        return K
