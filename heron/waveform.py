"""
This module is designed to allow for Gaussian Process modelling of 
gravitational waveforms.
"""


import george
from george import HODLRSolver
import elk
from george import kernels, GP
from elk.waveform import Waveform, Timeseries
from elk.catalogue import Catalogue, PPCatalogue
import numpy as np
import matplotlib.pyplot as plt


class Generator(Catalogue):
    """
    This class is a waveform generator which uses some underlying statistical model
    (for example, a Gaussian Process regressor) to produce waveforms.
    """

    def __init__(self, model):
        """
        Produce a waveform generator given a specific model.
        """
        self.model = model

    def _generate(self, theta):
        """
        Produce a single waveform at a location `theta` in the intrinsic parameter space.

        Parameters
        ----------
        theta : array-like
           A location in the intrinsic parameter space.
        """
        return self.model(theta)

    def __call__(self, theta):
        """
        Produce a single waveform at a location `theta` in the /extrinsic/ parameter space.

        TODO: Make this actually return the /extrinsic/ waveform.
        """
        intrinsic =  self._generate(theta)
        return intrinsic






    

class GPCatalogue(Catalogue):
    """
    This class represents a 'catalogue' of waveform built out of
    a numerical relativity catalogue.

    The Gaussian process allows the parameter space of the catalogue
    to be represented continuously, as opposed to in the discrete manner
    of the underlying NR catalogue.
    """

    def __init__(self, nrcat, kernel, total_mass=100, f_min=None, solver="hodlr",
                 tmax=0.01,
                 tmin=-0.015,
                 mean=0.0,
                 tol=1e-6,
                 white_noise=0,
                 ma = None,
                 fsample=4096):
        """
        Build the GP Catalogue from a numerical relativity catalogue.

        Parameters
        ----------
        nrcat : `elk.catalogue.NRCatalogue`
           The catalogue of numerical relativity waveforms.
        kernel : `george.kernel`
           The covariance function to be used for the Gaussian process.
        total_mass : float
           The total mass of the system to be simulated.
        f_min : float
           The minimum frequency to be included in the waveform.
        """

        self.kernel = kernel
        
        self.nr_data = nrcat
        self.total_mass = total_mass
        self.f_min = f_min
        self.sample_rate = fsample
        self.ma = ma
        self.tmax = tmax
        self.tmin = tmin
        self.training_data = self.nr_data.create_training_data(total_mass,
                                                               f_min = f_min,
                                                               sample_rate=fsample,
                                                               ma=ma,
                                                               tmax = tmax,
                                                               tmin=tmin,
        )
        
        self.columns = {0: "time",
                        1: "mass ratio",
                        2: "spin 1x",
                        3: "spin 1y",
                        4: "spin 1z",
                        5: "spin 2x",
                        6: "spin 2y",
                        7: "spin 2z",
                        8: "h+",
                        9: "hx"
        }
        self.c_ind = {j:i for i,j in self.columns.items()}

        self.training_data[:,self.c_ind['time']] *= 10000
        self.training_data[:,self.c_ind['mass ratio']] = np.log( self.training_data[:,self.c_ind['mass ratio']])
        #self.training_data[:,self.c_ind['mass ratio']] = np.log(self.training_data[:,self.c_ind['mass ratio']])
        #self.training_data[:,self.c_ind['time']] -= self.training_data[np.argmax(self.training_data[:,self.c_ind['time']]),self.c_ind['time']]
        self.training_data[:,self.c_ind['h+']] *= 1e19
        self.training_data[:,self.c_ind['hx']] *= 1e19

        self.x_dimensions = self.kernel.ndim
        self.solver = solver
        self.mean_f = mean
        self.white_noise = white_noise
        self.tol = tol
        
        self.build(solver, mean, white_noise, tol)

    def add_waveform(self, p):
        """
        Insert a new waveform into the Gaussian Process training set.
        This currently only works for the PPCatalogue.

        Parameters
        ----------
        p : dict
           A dictionary of waveform parameters.
        """
        if type(self.nr_data) == PPCatalogue:
            catalogue = self.nr_data
            catalogue.waveforms.append(p)
        else:
            raise ValueError("This method currently works only with PP catalogues.")
        self.training_data = self.nr_data.create_training_data(self.total_mass,
                                                               f_min = self.f_min,
                                                               sample_rate=self.sample_rate,
                                                               ma=self.ma,
                                                               tmax = self.tmax,
                                                               tmin=self.tmin)
        self.training_data[:,self.c_ind['time']] *= 10000
        self.training_data[:,self.c_ind['h+']] *= 1e19
        self.training_data[:,self.c_ind['hx']] *= 1e19
        
        self.build(self.solver, self.mean_f, self.white_noise, self.tol)

    def optimise(self, algorithm="adam", max_iter = 100, **kwargs):
        
        p0 = self.gp.get_parameter_vector()
        
        # # Define the objective function (negative log-likelihood in this case).
        def nll(p):
            self.gp.set_parameter_vector(p)
            #print(np.exp(p))
            ll = self.gp.log_likelihood(self.training_data[:,self.c_ind['h+']], quiet=True)
            #print(-ll)
            return -ll if np.isfinite(ll) else 1e25

        # # And the gradient of the objective function.
        def grad_nll(p):
            #print(np.exp(p))
            self.gp.set_parameter_vector(p)
            #print(gp.log_likelihood(data[c_ind['h+']]*1e19, quiet=True))
            return -self.gp.grad_log_likelihood(self.training_data[:,self.c_ind['h+']], quiet=True)

        if algorithm in ["adam", "adadelta"]:
        
            if algorithm == "adam":
                """
                Optimise using the adam algorithm.
                """
                import climin
                opt = climin.Adam(p0, grad_nll, **kwargs)

            elif algorithm == "adadelta":
                """
                Optimise using the adam algorithm.
                """
                import climin
                opt = climin.Adadelta(p0, grad_nll, **kwargs)

            for info in opt:
                    if info['n_iter']%10 == 0:
                        print("{} - {} - {}".format(info['n_iter'],
                                                    self.gp.log_likelihood(
                                                        self.training_data[:,self.c_ind['h+']],
                                                        quiet=True),
                                                    np.exp(self.gp.get_parameter_vector())
                        ))
                    if info['n_iter'] > max_iter: break
        else:
            def callback(xk):
                status_string = "Location: {}\nlog(evidence): {}"
                # Stop the optimisation if the maximum iterations
                # have been made.
                #if state['nit'] > max_iter: return True
                #if state['nit'] % 10 == 0:
                print(status_string.format(np.exp(xk), -nll(xk)))
                return False
                
            
            if algorithm == "bfgs":
                """
                Optimise using the Broyden-Fletcher-Goldfarb-Shanno quasi-Newtonian optimiser.
                """
                from scipy.optimize import minimize, Bounds
                
                bounds = [(-10, 10) for _ in range(len(p0))]
                

                result = minimize(nll, p0,
                         jac=grad_nll,
                         method="L-BFGS-B",
                         bounds=bounds,
                         callback=callback, **kwargs)
                self.gp.set_parameter_vector(result['x'])
        
    def build(self, solver="hodlr", mean=0.0, white_noise=0, tol=1e-6):
        """
        Construct the GP
        """

        if not solver:
            self.gp = GP(self.kernel, mean=mean, white_noise=white_noise)
        else:
            self.gp = GP(self.kernel, solver=HODLRSolver,
                         tol=tol,
                         min_size=100,
                         mean=mean, white_noise=white_noise)

        self.yerr = np.ones(len(self.training_data)) * 0 #1e-8
        self.gp.compute(self.training_data[:, :self.x_dimensions], self.yerr)

    def waveform(self, p, time_range, polarisation="h+"):
        """
        Return the mean waveform at a given location in the 
        BBH parameter space.
        """

        nt = time_range[2]
        points = np.ones((nt, self.x_dimensions))
        points[:,self.c_ind['time']] = np.linspace(time_range[0], time_range[1], nt)
        p['mass ratio'] = np.log(p['mass ratio'])
        for column, value in p.items():
            points[:, self.c_ind[column]] *= value

            
        mean, var = self.gp.predict(self.training_data[:,self.c_ind[polarisation]],
                                    points,
                                    return_var=True,
        )
        return Timeseries(data=mean/1e19, times=points[:,self.c_ind['time']]/1e4), Timeseries(data=var/1e19, times=points[:,self.c_ind['time']]/1e4)

    def waveform_samples(self, p, time_range, samples=100):
        """
       Return the mean waveform at a given location in the 
        BBH parameter space.
        """

        nt = time_range[2]
        points = np.ones((nt, self.x_dimensions))
        times = np.linspace(time_range[0], time_range[1], nt)
        points[:,self.c_ind['time']] = times

        for column, value in p.items():
            points[:, self.c_ind[column]] *= value
        
        samples = self.gp.sample_conditional(self.training_data[:,self.c_ind['h+']],
                                             points,
                                             size=samples
        )

        return_samples = [Timeseries(data=sample/1e19, times=times/1e4) for sample in samples]
        
        return np.array(return_samples)
        
    def mean(self, ranges, fixed):
        """
        Calculate the mean waveform from the Gaussian process.

        Parameters
        ----------
        ranges : dict
           A dictionary in which the keys are the name of the parameter
           and the values are a list in the format [start, end, npoints]
           at which the GP should be evaluated for the plane.
        fixed : dict
           A dictionary in which the keys are the name of the parameter
           which should be fixed, and the value is the fixed value of 
           that parameter.
        """

        if len(ranges.items()) > 2:
            raise ValueError("""The number of predicted dimensions must be 
            no greater than 2.""")

        elif len(ranges.items()) == 1:
            # This is a one-dimensional query of the catalogue
            range1, range1_label = list(ranges.values())[0], list(ranges.keys())[0]
            nx = range1[2]
            if range1_label == "mass ratio":                
                x = np.linspace(np.log(range1[0]), np.log(range1[1]), nx)
            else:
                x = np.linspace(range1[0], range1[1], nx)
            points = np.ones((nx, self.x_dimensions))
            points[:, self.c_ind[range1_label]] = x
            for column, value in fixed.items():
                points[:, self.c_ind[column]] *= value

            #points[:, self.c_ind['mass ratio']] = np.log(points[:, self.c_ind['mass ratio']])
                
            mean, var = self.gp.predict(self.training_data[:,self.c_ind['h+']],
                                    points,
                                    return_var=True,
            )

            return mean, var

        else:
            
            range1, range1_label = list(ranges.values())[0], list(ranges.keys())[0]
            range2, range2_label = list(ranges.values())[1], list(ranges.keys())[1]

            nx, ny = range1[2], range2[2]

            x = np.linspace(range1[0], range1[1], nx)
            y = np.linspace(range2[0], range2[1], ny)

            gridpoints = np.dstack(np.meshgrid(x, y)).reshape(-1, 2)

            points = np.zeros((nx*ny, self.x_dimensions))

            points[:, [self.c_ind[range1_label], self.c_ind[range2_label]]] = gridpoints

            mean, var = self.gp.predict(self.training_data[:,self.c_ind['h+']],
                                        points,
                                        return_var=True,
            )

            return mean.reshape(ny, nx), var.reshape(ny, nx)
        

    def plot_planes(self, ranges, fixed):
        """
        Produce a plot of the waveform predictions from
        the GP.

        Parameters
        ----------
        ranges : dict
           A dictionary in which the keys are the name of the parameter
           and the values are a list in the format [start, end, npoints]
           at which the GP should be evaluated for the plane.
        fixed : dict
           A dictionary in which the keys are the name of the parameter
           which should be fixed, and the value is the fixed value of 
           that parameter.
        """

        mean, var = self.mean(ranges, fixed)

        ranges_x = list(ranges.items())[0][1]
        ranges_y = list(ranges.items())[1][1]
        
        f, ax = plt.subplots(1,1)
        im = ax.imshow(mean, origin="lower", cmap = "magma", vmin=-3, vmax=3,
                       extent = (ranges_x[0], ranges_x[1], ranges_y[0], ranges_y[1]),
                       aspect = ( (ranges_x[1] - ranges_x[0])
                                  / (ranges_y[1] - ranges_y[0] )))
        cax = f.add_axes([0.9, 0.1, 0.02, 0.8])
        f.colorbar(im, cax=cax, orientation='vertical')

        ax.set_xlabel("Time [s * 1e4]")
        ax.set_ylabel(list(ranges.keys())[0])

        g, ax = plt.subplots(1,1)
        im = ax.imshow(var, origin="lower",
                       cmap = "magma",
                       extent = (ranges_x[0], ranges_x[1], ranges_y[0], ranges_y[1]),
                       aspect = ( (ranges_x[1] - ranges_x[0])
                                  / (ranges_y[1] - ranges_y[0] )))

        cax = f.add_axes([0.9, 0.1, 0.02, 0.8])
        f.colorbar(im, cax=cax, orientation='vertical')

        ax.set_xlabel("Time [s * 1e4]")
        ax.set_ylabel(list(ranges.keys())[1])

        return f, g
