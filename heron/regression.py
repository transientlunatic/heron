"""
Functions and classes for contructing regression surrogate models.
"""

import math as m
import numpy as np
import emcee
import scipy.linalg
from scipy.optimize import minimize
import george
import scipy
from .training import *
import copy

def load(filename):
    """
    Load a pickled heron Gaussian Process.
    """
    import pickle
    with open(filename, "rb") as gp_file:
        return pickle.load(gp_file)

class SingleTaskGP(object):
    """
    This is an implementaion of a Single task Gaussian process 
    regressor. That is, a GPR which is capable of acting as a 
    surrogate to a many-to-one function. The Single Task GPR is
    the fundamental building block of the MultiTask GPR, which 
    consists of multiple Single Tasks which are trained in tandem 
    (but which do NOT share correlation information).
    ---
    Ahem... There /are/ components of this code in here, but 
    things need a little bit more thought before this will work
    efficiently...
    An implementation of a Gaussian Process Regressor with 
    multiple response outputs and multiple inputs.
    """
    
    km = None
    
    def __repr__(self):
        """
        The printable representation of this object.
        """
        return "<Heron Gaussian Process instance with {} training points>".format(len(self.training_data))

    def _html_repr_(self):
        """
        The HTML representation of the object for use with IPython /
        Jupyter notebooks.

        """

        output = "<table>"
        output += "<tr>"
        output += "<th>Heron Gaussian Process</th>"
        output += "<th></th>"
        output += "</tr>"
        output += "<tr>"
        output += "<td>Training Points</th>"
        output += "<td>{}</td>".format(len(self.training_data))
        output += "</tr>"
        output += "<tr>"
        output += "<td>Test Points</th>"
        output += "<td>{}</td>".format(len(self.training_object.test_labels))
        output += "</tr>"
        output += "<tr>"
        output += "<td>Model Correlation</th>"
        output += "<td>{}</td>".format(self.correlation())
        output += "</tr>"
        output += "<tr>"
        output += "<td>Model RMSE</th>"
        output += "<td>{}</td>".format(self.rmse())
        output += "</tr>"
        output += "</table>"
        return output

    def __init__(self, training_data, kernel, tikh=1e-6, solver=george.HODLRSolver, hyperpriors = None, **kwargs):
        """
        Set up the Gaussian process regression.

        Parameters
        ----------
        training data : heron data object
           The training data, consisiting of labels and targets.
        kernel : heron kernel
           The kernel used to calculate the covariance matrix.
        tikh : float
           The Tikhonov regularization factor to be applied to the diagonal
           to avoid the attempt to invert an ill-posed matrix problem. Defaults to 1e-6.
        """

        self.tikh = tikh

        self.training_object = training_data
        self.training_data = self.training_object.targets
        self.training_y = self.training_object.labels
        self.yerror = self.training_object.label_sigma
        self.input_dim = self.training_data.ndim
        self.output_dim = self.training_y.ndim
        self.kernel = kernel #kernel(self.training_data.ndim, *kernel_args)
        #kwargs = {}
        if solver == george.HODLRSolver:
            kwargs['tol'] = self.tikh
        self.gp = george.GP(kernel,
                            solver=solver,
                            mean = np.mean(self.training_y),
                            fit_mean=False,
                            fit_white_noise=False,
                            **kwargs
        )
        self.kernel = self.gp.kernel
        self.hyperpriordistributions = hyperpriors
        self.update()
    
    def active_learn(self, afunction, x,y, iters=1, afunc_args={}):
        """
        Actively train the Gaussian process from a set of provided
        labels and targets using some acquisition function.

        afunction : function
           The acquisition function.
        x : array-like
           The input labels of the data. This can be a multi-dimensional array.
        y : array-like
           The input targets of the data. This can only be a single-dimensional 
           array at present.
        iters : int
           The number of times to iterate the learning process: equivalently, 
           the number of training points to digest.
        afunc_args : dict
           A dictionary of arguments for the acquisition function. Optional.

        """
        i=0
        while i < iters:
            # Choose the new sample from the area with the greatest uncertainty
            mean, var =  self.prediction(x)
            err = np.sqrt(np.diag(np.abs(var)))
            LB = afunction(mean, err, **afunc_args)
            new_sample = np.argmax(LB)
            self.add_data(np.atleast_1d(x[new_sample]), y[new_sample])
            i += 1

    def add_data(self, target, label, label_error=None):
        """
        Add data to the Gaussian process.
        """
        self.training_object.add_data(target, label, label_sigma=label_error, target_sigma=None)
        self.training_data = self.training_object.targets
        self.training_y = self.training_object.labels[0]
        self.yerror = self.training_object.label_sigma
        self.update()

    def set_bmatrix(self, values):
        """
        Set the values of the B matrix from a vector.
        """
        bm = values.reshape(self.output_dim, self.output_dim)
        bm += self.tikh * np.eye(bm.shape[0], bm.shape[1])
        if not np.all(np.linalg.eigvals(bm) > 0): return -1e25
        self.B_matrix = bm
        return self.loglikelihood()
        
    def set_hyperparameters(self, hypers):
        """
        Set the hyperparameters of the kernel function.
        """
        self.gp.set_parameter_vector(hypers)
        self.update()
        #return self.loglikelihood()

    def get_hyperparameters(self):
        """
        Return the kernel hyperparameters.
        """
        return self.gp.get_parameter_vector()
        
    
    def update(self):
        """
        Update the stored matrices.
        """
        self.gp.compute(self.training_data, self.training_object.label_sigma)
        #self.test_predict()


    def prediction(self, new_datum, normalised=False):
        """
        Produce a prediction at a new point, or set of points.

        Parameters
        ----------
        new_datum : array
           The coordinates of the new point(s) at which the GPR model should be evaluated.
        normalised : bool
           A flag to indicate if the input is already normalised 
           (this might be the case if you're trying to efficiently sample to parameter 
           space). If False the input will be normalised to the same range as the 
           training data.

        Returns
        -------
        prediction mean : array
           The mean values of the function drawn from the Gaussian Process.
        prediction variance : array
           The variance values for the function drawn from the GP.
        """
        
        training_y = self.training_y
        if training_y.ndim > 1:
            training_y = training_y[0,:]
        #new_datum = np.atleast_2d(new_datum)
        if not normalised:
            new_datum = self.training_object.normalise(new_datum, "target")
        mean, variance = self.gp.predict(self.training_y, new_datum, return_var=True)
        #return mean, variance
        return self.training_object.denormalise(mean, "label"), self.training_object.denormalise(variance, "label")


    def test_predict(self):
        """
        Calculate the value of the GP at the test targets.   
        """
        self.test_predictions = self.prediction(self.training_object.denormalise(self.training_object.test_targets, "target"))[0]

    def correlation(self):
        """
        Calculate the correlation between the model and the test data.
        
        Returns
        -------
        corr : float
           The correlation squared.
        """
        a = self.training_object.denormalise(self.training_object.test_labels, "label")
        b = self.test_predictions
        return np.linalg.det((np.cov(a,b) / np.sqrt(np.var(a) * np.var(b)))**2)

    def rmse(self):
        """
        Calculate the root mean squared error of the whole model.
        
        Returns
        -------
        rmse : float
           The root mean squared error.
        """

        a = self.training_object.denormalise(self.training_object.test_labels, "label")
        b = self.test_predictions
        return np.sqrt(np.mean((a - b)**2) )

    def expected_improvement(self, x):
        '''
        Returns the expected improvement at the design vector X in the model
        
        Parameters
        ==========
        x : array-like
           A real world coordinates design vector
        
        Returns
        =======
        EI: float 
           The expected improvement value at the point x in the model
        '''

        x = np.atleast_2d(x)
        p, S = self.prediction(x)
        p = np.abs(p)
        y_min = np.min(self.training_y)
        EI_one = ((y_min - p) * (0.5 + 0.5*scipy.special.erf((
            1./np.sqrt(2.))*((y_min - p) /
                             S))))
        EI_two = ((S * (1. / np.sqrt(2. * np.pi))) * (np.exp(-(1./2.) *
                                                             ((y_min - p)**2. / S**2.))))
        EI = EI_one + EI_two
        return EI

    def nei(self, x):
        """
        Calculate the negative of the expected improvement at a point x.
        """
        return -self.expected_improvement(x)

    def ln_likelihood(self, p):
        """
        Provides a convenient wrapper to the ln likelihood function.

        Notes
        -----
        This is implemented in a separate function because of the mild 
        peculiarities of how the pickle module needs to serialise 
        functions, which means that instancemethods (which this would 
        become) can't be serialised. 
        """
        return ln_likelihood(p, self)

    def _lnlikelihood(self, p):
        """
        Calculates the lnlikelihood for the GP.

        Parameters
        ----------
        p : list
           The vector of hyperparameters at which the lnlikelihood should be evaluated.

        Returns
        -------
        float 
           The lnlikelihood for the system.

        """
        self.set_hyperparameters(p)
        if self.training_y.ndim > 1:
            return  self.gp.lnlikelihood(self.training_y[0])
        else:
            return self.gp.lnlikelihood(self.training_y)
    
    def neg_ln_likelihood(self, p):
        """

        Returns the negative of the log-likelihood; designed for use with
        minimisation algorithms.

        Parameters
        ----------
        gp : heron `Regressor` object
           The gaussian process to be evaluated.
        p : array-like
           An array of the hyper-parameters at which the model is to be evaluated.

        Returns
        -------
        neg_ln_likelihood : float
           The negative of the log-likelihood for the Gaussian process
        """
        return -self.ln_likelihood(p)

    def entropy(self):
        """Return the entropy of the Gaussian Process distribution. This can
        be calculated directly from the covariance matrix, making this
        a nice, quick calculation to perform.

        Returns
        -------
        entropy : float
           The differential entropy of the GP.
        """
        
        return 0.5 * ( np.log((2*np.pi*np.e)**self.training_data.shape[1]) + self.gp.solver.log_determinant)

    def hyperpriortransform(self, p):
        """Return the true value in the desired hyperprior space, given an
        input of a unit-hypercube prior space.

        Parameters
        ----------
        p : array-like
           The point in the unit hypercube space

        Returns
        -------
        x : The position in the desired hyperparameter space of the point.
        """

        hypers = self.hyperpriordistributions
        x = []
        for hyper, pv in zip(hypers, p):
            x.append(hyper.transform(p))
        return np.array(x)
    
    def loghyperpriors(self, p):

        """
        Calculate the log of the hyperprior distributions at a given 
        point.
        
        Parameters
        ----------
        p : ndarray
            The location to be tested.
        """
        hypers = self.hyperpriordistributions
        probs = 1
        for hyper, pv in zip(hypers, p):
            probs += hyper.logp(pv)
        return probs
        

    def grad_neg_ln_likelihood(self, p):
        """
        Return the negative of the gradient of the log likelihood for the
        GP when its hyperparameters have some specified value.

        Parameters
        ----------
        gp : heron `Regressor` object
           The gaussian process to be evaluated
        p : array-like
           An array of the hyper-parameters at which the model is to be evaluated.

        Returns
        -------
        grad_ln_likelihood : float
           The gradient of log-likelihood for the Gaussian process
        """

        self.gp.set_parameter_vector(p)
        return self.gp.grad_lnlikelihood(self.training_y)

    def train(self, method="MCMC", metric="loglikelihood", sampler="ensemble", **kwargs):
        """
        Train the Gaussian process by finding the optimal 
        values for the kernel hyperparameters.
        
        Parameters
        ----------
        method : str {"MCMC", "MAP", "nested"}
           The method to be employed to calculate the hyperparameters.
        metric : str
           The metric which should be used to assess the model.
        hyperpriors : list
           The hyperprior distributions for the hyperparameters. Defaults to None, in which 
           case the prior is uniform over all real numbers.
        """

        if method=="MCMC":
            gp, samples, burn = run_training_mcmc(self, metric = metric, samplertype=sampler, **kwargs)
            self.gp = gp
            return samples, burn
        elif method=="nested":
            # Use nested sampling to train the model
            # NB this is experimental
            results = run_nested(gp, metric=metric, **kwargs)
            return results
        elif method == "MAP":
            MAP = run_training_map(self, metric = metric, **kwargs)
            
            return MAP
        
    def save(self, filename):
        """
        Save the Gaussian Process to a file which can be reloaded later.

        Parameters
        ----------
        filename : str
           The location at which the Gaussian Process should be written.

        Notes
        -----
        In the current implementation the serialisation of the GP is performed
        by the python `pickle` library, which isn't guaranteed to be binary-compatible 
        with all machines.
        """

        import pickle
        with open(filename, "wb") as filedump:
            pickle.dump(self, filedump)
        
class MultiTaskGP(SingleTaskGP):
    """
    An implementation of a co-trained set of Gaussian processes which
    share the same hyperparameters, but which model differing
    data. The training of these models is described in RW pp115--116.

    A multi-task GPR is capable of acting as a surrogate to a many-to-many function, 
    and is trained by making the assumption that all of the outputs from the function 
    share a common correlation structure.
    
    The principle difference compared to a single task GP is the
    presence of multiple Gaussian Processes, with one to model each
    dimension of the output data.

    Notes
    -----
    The MultiTask GPR implementation is very much a work in progress at the
    moment, and not all methods implemented in the SingleTask GPR are implemented
    correctly yet.

    """

    def __init__(self, training_data, kernel, tikh=1e-6, solver=george.HODLRSolver, hyperpriors = None):
        """
        Set up the multi-task Gaussian process regression.

        Parameters
        ----------
        training data : heron data object
           The training data, consisiting of labels and targets.
        kernel : heron kernel
           The kernel used to calculate the covariance matrix.
        tikh : float
           The Tikhonov regularization factor to be applied to the diagonal
           to avoid the attempt to invert an ill-posed matrix problem. Defaults to 1e-6.
        """

        self.tikh = tikh

        self.training_object = training_data
        self.training_data = self.training_object.targets
        self.training_y = self.training_object.labels
        self.yerror = self.training_object.label_sigma
        self.input_dim = self.training_data.ndim
        self.output_dim = self.training_y.ndim
        #self.kernel = kernel #kernel(self.training_data.ndim, *kernel_args)
        self.gps = []
        for i in xrange(self.output_dim):
            sub_training_data = training_data.copy()
            sub_training_data.labels = sub_training_data.labels[:,i]
            sub_training_data.label_sigma = sub_training_data.label_sigma[:,i]
            self.gps.append(SingleTaskGP(sub_training_data, kernel, tikh, solver, hyperpriors))
        self.kernel = self.gps[0].kernel
        self.hyperpriordistributions = hyperpriors
        self.update()

    def update(self):
        """
        Update the stored matrices.
        """
        for gp in self.gps:
            gp.update()

    def get_hyperparameters(self):
        """
        Return the kernel hyperparameters. Returns the hyperparameters of
        only the first GP in the network; the others /should/ all be
        the same, but there might be something to be said for checking
        this.

        Returns
        -------
        hypers : list
           A list of the kernel hyperparameters
        """
        return self.gps[0].get_hyperparameters()
            
    def set_hyperparameters(self, hypers):
        """
        Set the hyperparameters of the kernel function on each Gaussian process.
        """
        for gp in self.gps:
            gp.set_hyperparameters(hypers)

    def train(self, method="MCMC", metric="loglikelihood", sampler="ensemble", **kwargs):
        """
        Train the Gaussian process by finding the optimal 
        values for the kernel hyperparameters.
        
        Parameters
        ----------
        method : str {"MCMC", "MAP"}
           The method to be employed to calculate the hyperparameters.
        metric : str
           The metric which should be used to assess the model.
        hyperpriors : list
           The hyperprior distributions for the hyperparameters. Defaults to None, in which 
           case the prior is uniform over all real numbers.
        """

        if method=="MCMC":
            samples, burn = run_training_mcmc(self, metric = metric, samplertype=sampler, **kwargs)
            return samples, burn
        elif method == "MAP":
            MAP = run_training_map(self, metric = metric, **kwargs)
            return MAP

    def _lnlikelihood(self, p):
        """
        Calculates the lnlikelihood for the entire system of GPs in the multitask setup.

        Parameters
        ----------
        p : list
           The vector of hyperparameters at which the lnlikelihood should be evaluated.

        Returns
        -------
        float 
           The lnlikelihood for the system.

        """
        self.set_hyperparameters(p)
        lnlike = [gp._lnlikelihood(p) for gp in self.gps]
        return np.sum(lnlike)
        
    def ln_likelihood(self, p):
        """Provides a wrapper to the ln_likelihood functions for each
        component Gaussian process in the multi-task system.

        Notes
        -----
        This is implemented in a separate function because of the mild 
        peculiarities of how the pickle module needs to serialise 
        functions, which means that instancemethods (which this would 
        become) can't be serialised.

        """
        return ln_likelihood(p, self)

    def prediction(self, new_datum):
        """
        Produce a prediction at a new point, or set of points.

        Parameters
        ----------
        new_datum : array
           The coordinates of the new point(s) at which the GPR model should be evaluated.

        Returns
        -------
        prediction means : array
           The mean values of the function drawn from the Gaussian Process.
        prediction variances : array
           The variance values for the function drawn from the GP.
        """
        means, variances = [], []
        #new_datum = np.atleast_2d(new_datum)#.T
        new_datum = self.training_object.normalise(new_datum, "target")

        print(new_datum)
        
        for ix, gp in enumerate(self.gps):
            training_y = self.training_y[:,ix]
            mean, variance = gp.gp.predict(training_y, new_datum, return_var=True)
            means.append(mean)#gp.training_object.denormalise(mean, "label"))
            variances.append(variance) #gp.training_object.denormalise(variance, "label"))
        return means, variances

# For backwards compatibility...
class Regressor(SingleTaskGP):
    pass
