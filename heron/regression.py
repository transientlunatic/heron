import math as m
import numpy as np
import emcee
import scipy.linalg
from scipy.optimize import minimize
import george
import scipy

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

    def __init__(self, training_data, kernel, yerror = 0, tikh=1e-6, solver=george.HODLRSolver):
        """
        Set up the Gaussian process regression.

        Parameters
        ----------
        training data : heron data object
           The training data, consisiting of labels and targets.
        kernel : heron kernel
           The kernel used to calculate the covariance matrix.
        yerror : 
           The variance of the labels
        tikh : float
           The Tikhonov regularization factor to be applied to the diagonal
           to avoid the attempt to invert an ill-posed matrix problem. Defaults to 1e-6.
        """

        self.tikh = tikh

        self.training_object = training_data
        self.training_data = self.training_object.targets
<<<<<<< HEAD
        self.training_y = self.training_object.labels
        self.yerror = self.training_object.label_sigma
        self.input_dim = self.training_data.ndim
        self.output_dim = self.training_y.ndim
=======
        self.training_y = np.atleast_2d(self.training_object.labels.T.reshape(-1)).T
        self.yerror = self.training_object.label_sigma.reshape(-1)
        self.input_instances = self.training_object.targets.shape[0]
        self.input_dim = self.training_object.targets.shape[1]
        self.output_dim = self.training_object.labels.shape[1]

        self.B_matrix = np.identity(self.output_dim)

        #np.atleast_2d(training_data)
>>>>>>> f02bf7fa5ed4df3149e9245a8268c710f611e0f9
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
<<<<<<< HEAD
        self.gp.compute(self.training_data)
        #self.test_predict()


    def loglikelihood(self):
        """
        Return the log-likelihood function for the Gaussian process.
        """
        return self.gp.lnlikelihood(self.training_y)
=======
        km = self.kernel.matrix(self.training_data, self.training_data) 

        km += self.tikh * np.eye(km.shape[0], km.shape[1])
        self.L = scipy.linalg.cholesky(km)
        km = np.kron(self.B_matrix, km)
        if isinstance(self.yerror , float):
            km += self.yerror * np.eye(km.shape[0], km.shape[1])
        elif isinstance(self.yerror, np.ndarray):
            km += np.diag(self.yerror)
        self.km = km 
        #self.test_predict()

    def K_matrix(self):
        """
        Produce the Kx,x matrix (the covariance matrix of the training
        inputs)
        """
        return self.km 
        #km = self.kernel.matrix(self.training_data, self.training_data)
        #return km
    
    def Kstar_matrix(self, data):
        """
        Produce the Nx1 matrix which describes the locations of the prediction
        data.
        """
        return np.kron(self.B_matrix, self.kernel.matrix(self.training_data, data))
    
    def Kstar_scalar(self, data):
        return np.kron(self.B_matrix, self.kernel.matrix(data, data))
    
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
        training_y = self.training_y.T[0]
        LD = np.linalg.slogdet(self.K_matrix())
        return -0.5 * np.dot(self.apply_inverse(training_y),training_y) - 0.5 * LD[0]*LD[1]  - self.output_dim*0.5*np.log(2*np.pi)
>>>>>>> f02bf7fa5ed4df3149e9245a8268c710f611e0f9
    
    def grad_loglikelihood(self):
        """
        Calculate the gradient of the log(likelihood) function
        """
<<<<<<< HEAD
        return self.gp.grad_lnlikelihood(self.training_y)
        
    
=======
        dK = self.kernel.gradient(self.training_data, self.training_data)
        dK = np.kron(self.B_matrix, dK)
        KI = self.apply_inverse(np.eye(self.km.shape[0]))
        A = self.apply_inverse(self.training_y.T[0])
        B = np.outer(A, A) - KI
        g = 0.5 * np.einsum('ijk,jk', dK, B)
        return g
        

    def apply_inverse(self, matrix):
        """
        Apply the inverse of the K matrix to another object using Colesky
        decomposition.
        """
        #KK = np.copy(self.K_matrix())
        #KK += self.tikh * np.eye(KK.shape[0], KK.shape[1])
        L = self.L
        B = scipy.linalg.cholesky(self.B_matrix)
        LI = np.linalg.inv(L)
        BI = np.linalg.inv(B)
        LI = np.dot(LI.T, LI)
        BI = np.dot(BI.T, BI)
        A = np.kron(BI, LI)
        return np.dot(A, matrix)
        #return scipy.linalg.cho_solve(L, matrix, overwrite_b=False)
    
    def mean(self, newdata):
        KS = self.Kstar_matrix(newdata)
        means = np.dot(KS.T, self.apply_inverse(self.training_y))
        return means
        
    def covariance(self, newdata):
        KS = self.Kstar_matrix(newdata)
        KST = np.ascontiguousarray(KS.T, dtype=np.float64)
        b =self.apply_inverse(KS)
        cov = self.Kstar_scalar(newdata) - np.dot(KST, self.apply_inverse(KS))
        return cov

>>>>>>> f02bf7fa5ed4df3149e9245a8268c710f611e0f9
    def prediction(self, new_datum):
        training_y = self.training_y
        new_datum = np.atleast_2d(new_datum).T
        new_datum = self.training_object.normalise(new_datum, "target")
<<<<<<< HEAD
        mean, variance = self.gp.predict(self.training_y, new_datum, return_var=True)
        return self.training_object.denormalise(mean, "label"), self.training_object.denormalise(variance, "label")
=======
        mean = self.mean(new_datum)
        variance = self.covariance(new_datum)
        return mean.reshape(self.output_dim,-1).T, variance
>>>>>>> f02bf7fa5ed4df3149e9245a8268c710f611e0f9

    def optimise(self):
        """
        Find the optimal values for the kernel hyper-parameters by maximising the 
        log-likelihood of the entire Gaussian Process. It's also possible to do
        this via cross-validation.
        """
        def neg_ln_like(p):
            self.gp.set_vector(p)
            return -self.gp.lnlikelihood(self.training_y)

        def grad_neg_ln_like(p):
            self.gp.set_vector(p)
            return -self.gp.grad_lnlikelihood(self.training_y)

<<<<<<< HEAD
        result = minimize(neg_ln_like, self.gp.get_vector(), method="BFGS",  jac=grad_neg_ln_like)

        #x0 = self.gp.get_vector()
        #res = minimize(nll, x0, method='BFGS', jac=grad_nll ,options={'disp': False})
        return result
=======
        def grad_nll(p):
            self.set_hyperparameters(p)
            return -self.grad_loglikelihood()
        x0 = self.kernel.flat_hyper
        res = minimize(nll, x0, method='BFGS', jac=grad_nll ,options={'disp': False})
        
        return res
>>>>>>> f02bf7fa5ed4df3149e9245a8268c710f611e0f9


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
