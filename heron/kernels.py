"""
Kernel functions for GPs.
"""

import numpy as np
from scipy.spatial.distance import cdist

class Kernel():
    """
    A generic factory for Kernel classes.
    """
    name = "Generic kernel"
    ndim = 1
    def set_hyperparameters(self, hypers):
        if not len(hypers) == ndim + 1:
            raise ValueError("Wrong number of hyper-parameters passed to the kernel.")
        self.hyper = hypers
    
    def distance(self, data1, data2, hypers = None):
        """
        Calculate the squared distance to the point in parameter space.
        """
        
        if hypers == None:
            hypers = np.ones(self.ndim)
        hypers = np.atleast_2d(hypers)
        if len(hypers) == 1:
            # Assume that we've given a single hyper 
            # which can be used for all of the dimensions.
            hypers = np.ones(self.ndim)*hypers
        elif len(hypers) != self.ndim:
            raise ValueError("There are not enough dimensions in the hyperparameter vector.")
        hypers = np.squeeze(hypers)
        #hypers = np.exp(hypers)
        
        d = cdist(data1, data2, 'wminkowski', p=2, w = hypers)
        return d**2
    
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
        #print data1.shape, data2.shape
        return self.function(data1, data2)

class ExponentialSineSq(Kernel):
    """
    An implementation of the exponential sine-squared kernel.
    """
    name = "Exponential sine-squared kernel"
    hyper = [1, 1]
    
    def __init__(self, period = 1, width = 15, ax=0):
        
        # The number of axes dictates the dimensionality of the 
        # Kernel.
        # -
        # This kernel is only easily expressed as a single 
        # dimensional expression, so we take the product 
        # of multiple kernels to make it multidimensional.
        self.ax = ax
        if isinstance(ax, int):
            self.ndim = 1
        else:
            self.ndim = len(ax)
        # The kernel possesses one parameter which applies to
        # all of the axes, and one which applies to each
        # axis separately, so
        self.nparam = 1 + self.ndim
        if isinstance(period, list):
            self.hyper = [width, period]
        else:
            self.hyper = [width, [period]*self.ndim]


    def function(self, data1, data2, period):
        """
        The functional form of the kernel inside the exponential.
        """
        # The distance is not weighted, and we perform the 
        # weighting later.
        d = self.distance(data1, data2)
        return np.sin(np.pi * (d/period) )**2

    def matrix(self, data1, data2):
        front = 2 / self.hyper[0]**2
        if isinstance(self.ax, int):
            return np.exp(front * self.function(data1, data2, period = self.hyper[1]))
        else:
            matrix = np.zeros(data1, data2)
            for axis in ax:
                matrix += self.function(data1[:,axis], data2[:,axis], period = self.hyper[1][axis])
            return np.exp(front * matrix)

    def gradient(self, data1, data2):
        gradients = np.zeros(self.nparam)
        # Precalculate the gram matrix
        k = self.matrix(data1,data2)
        # The first component is the fixed width-factor
        gradient[0] = - 1. / self.hyper[0]**3 * k
        
        gradient[1:]  = 2*np.pi

class SquaredExponential(Kernel):
    """
    An implementation of the squared-exponential kernel.
    """
    name = "Squared exponential kernel"
    hyper = [1.0]
    def __init__(self,ndim=1, amplitude=100, width=15):
        self.ndim = ndim
        self.nparam = 1 + self.ndim
        self.hyper = [amplitude, width]
        
    def set_hyperparameters(self, hypers):
        self.hyper = [hypers[0], hypers[1:]]
        
    @property
    def flat_hyper(self):
        return np.append(self.hyper[0], self.hyper[1:])

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
        #gradients = np.zeros(self.nparam)
        gradients = []
        d = self.distance(data1, data2)
        # First calculate the gradient wrt to the scaling term
        gradients.append( np.exp(-np.abs(d)) )
        # Now calculate the gradient wrt all of the width factors
        for i in xrange(self.ndim):
            # Set the ith hyperparameter to equal 1
            th = np.copy(self.hyper[1])
            th[i] = 1
            d = self.distance(data1, data2, hypers = th )
            gradients.append( self.hyper[0] * np.exp(-np.abs(d)) )
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
