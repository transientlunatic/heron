import numpy as np
import emcee
class Regressor():
    """
    An implementation of a Gaussian Process Regressor
    """
    
    km = None
    
    def __init__(self, training_data, training_y, kernel, kernel_args):
        self.training_data = training_data
        self.training_y = training_y
        #np.atleast_2d(training_data)
        self.kernel = kernel(training_data.ndim, *kernel_args)
    
    def optimise(self, nwalkers=100, nsamples=1000, burn=1000):
        # This is an ugly kludge, FIX ME by moving to the kernel class
        ndim = len(self.kernel.hyper[1]) + 1
        # Make a random initial point
        p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
        # Set up the MCMC sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.set_hyperparameters, args=[])
        # Run the burn-in
        pos, prob, state = sampler.run_mcmc(p0, burn)
        sampler.reset()
        # Make the production samples
        pos, prob, state = sampler.run_mcmc(p0, nsamples)
        return sampler, pos, prob, state
    
    def set_hyperparameters(self, hypers):
        """
        Set the hyperparameters of the kernel function.
        """
        self.kernel.set_hyperparameters(hypers)
        return self.loglikelihood()
    
    
    def K_matrix(self):
        """
        Produce the Kx,x matrix (the covariance matrix of the training
        inputs)
        """
        km = self.kernel.matrix(self.training_data, self.training_data)
        return km
    
    def Kstar_matrix(self, data):
        """
        Produce the Nx1 matrix which describes the locations of the prediction
        data.
        """
        return self.kernel.matrix(self.training_data, data)
    
    def Kstar_scalar(self, data):
        return self.kernel.matrix(data, data)
    
    def Kplus_matrix(self, data):
        data = np.expand_dims(data,0)
        new_size = self.training_data.shape[-1]+data.shape[-1]
        new_matrix = np.zeros((new_size, new_size))
        a = len(gp.K_matrix())
        new_matrix[:a, :a] = self.K_matrix()
        new_matrix[a:,:a] = self.Kstar_matrix(data)
        new_matrix[:a,a:] = self.Kstar_matrix(data).T
        new_matrix[a:,a:] = self.Kstar_scalar(data)
        return new_matrix
    
    def loglikelihood(self):
        training_y = self.training_y
        try:
            KI = np.linalg.inv(self.K_matrix())
        except LinAlgError:
            return -np.inf
        LD = np.linalg.slogdet(self.K_matrix())
        return -0.5 * np.dot(np.dot(training_y.T, KI),training_y) - 0.5 * LD[0]*LD[1]  - 0.5*np.log(2*np.pi)
    
    def prediction(self, new_datum):
        training_y = self.training_y
        KI = np.linalg.inv(self.K_matrix())
        new_datum = np.array(new_datum)
        KS = self.Kstar_matrix(new_datum)
        KK = np.dot(KS.T, KI)
        mean = np.dot(KK, training_y)
        variance = self.Kstar_scalar(new_datum) - np.dot(np.dot(KS.T, KI), KS)
        return mean, variance
