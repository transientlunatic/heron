import math as m
import numpy as np
import emcee
import scipy.linalg
from scipy.optimize import minimize
import george
import scipy
from .training import *

class Regressor():
    """
    An implementation of a Gaussian Process Regressor with multiple response outputs and multiple inputs.
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

    def __init__(self, training_data, kernel, tikh=1e-6, solver=george.HODLRSolver):
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
        self.gp = george.GP(kernel, solver=solver, tol=self.tikh, mean = 0, fit_mean=False, fit_white_noise=False)
        self.kernel = self.gp.kernel
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
        self.gp.set_vector(hypers)
        self.update()
        #return self.loglikelihood()
    
    def update(self):
        """
        Update the stored matrices.
        """
        self.gp.compute(self.training_data)
        #self.test_predict()


    def prediction(self, new_datum):
        """
        Produce a prediction at a new point, or set of points.

        Parameters
        ----------
        new_datum : array
           The coordinates of the new point(s) at which the GPR model should be evaluated.

        Returns
        -------
        prediction mean : array
           The mean values of the function drawn from the Gaussian Process.
        prediction variance : array
           The variance values for the function drawn from the GP.
        """
        
        training_y = self.training_y
        new_datum = np.atleast_2d(new_datum).T
        new_datum = self.training_object.normalise(new_datum, "target")

        mean, variance = self.gp.predict(self.training_y, new_datum, return_var=True)
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
        return training.ln_likelihood(p, self)
    
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

        self.gp.set_vector(p)
        return self.gp.grad_lnlikelihood(self.training_y)

    def train(self, method="MCMC", metric="loglikelihood", sampler="ensemble"):
        """
        Train the Gaussian process by finding the optimal 
        values for the kernel hyperparameters.
        
        Parameters
        ----------
        method : str {"MCMC", "CV"}
           The method to be employed to calculate the hyperparameters.
        metric : str
           The metric which should be used to assess the model.
        """

        if method=="MCMC":
            samples, burn = run_training_mcmc(self, metric = metric, samplertype=sampler)
            return samples, burn
        elif method=="CV":
            run_training_map(self, metric, samplertype=sampler)


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
        with open(filename, "wb"):
            pickle.dump(self, filename)
        
        
