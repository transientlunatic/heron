"""
Models which use the GPyTorch GPR package as their backbone.
"""

from functools import reduce
import operator

import logging

import pkg_resources

import os

import torch
import gpytorch
from gpytorch.kernels import RBFKernel
from lal import cached_detector_by_prefix, TimeDelayFromEarthCenter, LIGOTimeGPS

from elk.waveform import Timeseries, FrequencySeries

from . import Model
from ..data import DataWrapper
from .gw import BBHSurrogate, HofTSurrogate

DATA_PATH = pkg_resources.resource_filename("heron", "models/data/")
disable_cuda = False
if not disable_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

disable_cuda = False
if not disable_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def train(model, iterations=1000, lr=0.1):
    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.model_plus.parameters(), lr=lr
    )  # Includes GaussianLikelihood parameters
    model.model_plus.train()
    model.model_cross.train()
    model.likelihood.train()
    model.likelihood_cross.train()
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model.model_plus)
    for i in range(iterations):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model.model_plus(model.training_x)
        # Calc loss and backprop gradients
        loss = -mll(output, model.training_y)
        loss.backward()
        optimizer.step()
        model.model_cross.load_state_dict(model.model_plus.state_dict())

    # Update the hyperparameters in the training file
    model.training_data.h5file.close()
    training_data = DataWrapper(model.datafile, write=True)
    try:
        del training_data["model states"][model.model_name]
    except KeyError:
        pass
    training_data.add_state(
        name=model.model_name, group=model.datalabel, data=model.model_plus.state_dict()
    )

    model.training_data = DataWrapper(model.datafile)
    #
    model.model_plus.eval()
    model.model_cross.eval()
    model.likelihood.eval()
    model.likelihood.eval()


class CUDAModel(Model):
    """
    The factory class for all CUDA-based models.
    """

    def __init__(self, device=None):
        if device:
            self.device = device
        else:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

    def eval(self):
        """
        Prepare the model to be evaluated.
        """
        if hasattr(self, "model_plus"):
            self.model_plus.eval()
        if hasattr(self, "model_cross"):
            self.model_cross.eval()
        self.likelihood.eval()
        self.likelihood_cross.eval()

    def _process_inputs(self, times, p):
        times *= self.time_factor
        times += 0.40
        times[times < 0] = times[times < 0] / 4.0
        times -= 0.40
        times -= 0.0017
        p = {k: self.time_factor * v for k, v in p.items()}
        return times, p

    def _predict(self, times, p, polarisation="plus"):
        """
        Query the model for the mean and covariance tensors.
        Optionally include the covariance.

        Parameters
        -------------
        times : ndarray
           The times at which the model should be evaluated.
        p : dict
           A dictionary of locations in parameter space where the model
           should be evaluated.

        Returns
        -------
        mean : torch.tensor
           The mean waveform
        var : torch.tensor
            The variance of the waveform
        cov : torch.tensor, optional
            The covariance matrix of the waveform.
            Only returned if covariance was True.
        """

        if polarisation == "plus":
            model = self.model_plus
            likelihood = self.likelihood
        elif polarisation == "cross":
            model = self.model_cross
            likelihood = self.likelihood_cross

        if not isinstance(times, torch.Tensor):
            times = torch.tensor(times)

        points = self._generate_eval_matrix(p, times)
        points = points.to(device=self.device, dtype=torch.float)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_preds = likelihood(model(points))

        mean = f_preds.mean / self.strain_input_factor
        var = f_preds.variance.double().abs() / self.strain_input_factor
        covariance = f_preds.covariance_matrix.double()
        covariance /= self.strain_input_factor**2

        return mean, var, covariance

    def mean(self, times, p, covariance=False, polarisation=None):
        """
        Provide the mean waveform and its variance.
        Optionally include the covariance.

        Parameters
        ----------
        times : ndarray
           The times at which the model should be evaluated.
        p : dict
           A dictionary of locations in parameter space where the
           model should be evaluated.
        covariance : bool, optional
           A flag to determine if the whole covariance matrix
           should be returned.

        Returns
        -------
        mean : torch.tensor
           The mean waveform
        var : torch.tensor
            The variance of the waveform
        cov : torch.tensor, optional
            The covariance matrix of the waveform.
            Only returned if covariance was True.

        Examples
        --------

        To plot a waveform and its variance:
        >>> preds = model.mean(times=1*np.linspace(-.04, 0.01, 1000),
        >>>                    p={"mass ratio": q})
        >>> plt.plot(preds['plus'].times.cpu(), preds['plus'].data.cpu())
        >>> plt.fill_between(preds['plus'].times.cpu(),
        >>>                  preds['plus'].data.cpu()+(preds['plus'].variance.cpu().detach()),
        >>>                  preds['plus'].data.cpu()-(preds['plus'].variance.cpu().detach()), alpha=0.2)
        """
        timeseries = {}
        if not isinstance(times, torch.Tensor):
            times = times - 0.0017
            times = torch.tensor(times)

        if hasattr(self, "model_cross"):
            polarisations = ["plus", "cross"]
        else:
            polarisations = ["plus"]
        for polarisation in polarisations:

            mean, var, covariance = self._predict(
                times.clone(), p, polarisation=polarisation
            )

            timeseries[polarisation] = Timeseries(
                data=mean, times=times + 0.0017, covariance=covariance, variance=var
            )
        return timeseries


class HeronCUDA(CUDAModel, BBHSurrogate, HofTSurrogate):
    """
    A GPR BBH waveform model which is capable of using CUDA resources.
    """

    time_factor = 100
    other_input_factor = 100
    strain_input_factor = 1e22
    x_dimensions = 8
    decimation = 1

    def __init__(
        self,
        datafile: str = None,
        datalabel: str = None,
        device: str = None,
        size: int = None,
        noise: list = [0.001, 0.1],
        name: str = "HeronCUDA",
        lengths: dict = {"mass ratio": [0.001, 1], "time": [0.001, 0.1]},
    ):
        """
        Construct a CUDA-based waveform model with pyTorch

        Parameters
        ----------
        datafile : str
           The name or path to the file containing the training data.
        datalabel : str
           The label for the training set to be used for this model.
        name : str
           The name of the model; used to find the saved model state information.
        size : int, optional
           The size of the training data set to be used.
           Defaults to None in which case all of the training data is used.
        noise : list, optional
           A list containing the lower and upper bounds of the noise to be added to
           the likelihood function.
        lengths : dict, optional
           A dictionary of lengthscales for the kernels.

        Examples
        --------
        Loading a non-spinning model trained with IMRPhenomPv2 waveforms:
        >>> import torch
        >>> from heron.models.torchbased import HeronCUDA,  train
        >>> model = HeronCUDA(datafile="training_data.h5",
        >>>          datalabel="IMR training linear",
        >>>          name="Heron IMR Non-spinning",
        >>>          device=torch.device("cuda"),
        >>>         )
        """
        super().__init__(device=device)
        #
        self.model_name = name if name else "Heron IMR Non-spinning"
        self.logger = logging.getLogger("heron.models.HeronCUDA")
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Name:{self.model_name}")
        self.logger.info(f"Device:{self.device}")
        #
        self.datafile = datafile if datafile else "training_data.h5"
        self.logger.info(f"Data file: {self.datafile}")
        self.datalabel = datalabel if datalabel else "IMR training linear"
        self.logger.info(f"Data label: {self.datalabel}")
        self.data_size = size
        #

        # Load the training data
        if not os.path.exists(self.datafile):
            if os.path.exists(os.path.join(DATA_PATH, self.datafile)):
                self.datafile = os.path.join(DATA_PATH, self.datafile)
            else:
                raise FileNotFoundError
        self.training_data = DataWrapper(self.datafile)

        #

        if len(self.training_data[self.datalabel]["meta"]["reference mass"]) > 0:
            self.reference_mass = self.training_data[self.datalabel]["meta"][
                "reference mass"
            ]
        else:
            self.reference_mass = 20.0

        # Kernel and likelihood settings
        self.noise = noise
        self.lengths = lengths
        #
        self.x_dimensions = len(
            self.training_data[self.datalabel]["meta"]["parameters"]
        )
        self.parameters = list(self.training_data[self.datalabel]["meta"]["parameters"])
        #
        self.columns = dict(
            enumerate(self.training_data[self.datalabel]["meta"]["parameters"])
        )
        self.c_ind = {j: i for i, j in self.columns.items()}
        #
        self.time_factor = 100
        self.strain_input_factor = 1e22
        #
        (self.model_plus, self.model_cross), (
            self.likelihood,
            self.likelihood_cross,
        ) = self.build()
        #
        # Load the trained hyperparameters if they're available
        #
        try:
            hypers = self.training_data.get_states(
                name=self.model_name, device=self.device
            )["hyperparameters"]
            self.model_plus.load_state_dict(hypers)
            self.model_cross.load_state_dict(hypers)
        except KeyError as e:
            self.logger.warning(
                "This model needs to be trained as training states could not be found!"
            )
            self.logger.exception(e)
        self.eval()

    # def _process_inputs(self, times, p):
    #     times *= self.time_factor
    #     p = {k: self.other_input_factor*v for k, v in p.items()}
    #     return times, p

    def build(self):
        """
        Right now this isn't need by this method
        """

        def prod(iterable):
            return reduce(operator.mul, iterable)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_cross = gpytorch.likelihoods.GaussianLikelihood()
        # This needs to be changed when the model has more than one dimension, but I'm trying to
        # get anything to work right now.
        time_kernel = RBFKernel(active_dims=self.c_ind[b"time"])
        mass_kernel = RBFKernel(active_dims=self.c_ind[b"mass ratio"])
        # TODO: Add mass ratio support to the model
        # TODO: Add spin support to the model

        class ExactGPModel(gpytorch.models.ExactGP):
            """
            Use the GpyTorch Exact GP
            """

            def __init__(self, train_x, train_y, likelihood):
                """Initialise the model"""
                super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

                self.mean_module = gpytorch.means.ZeroMean()
                inner_kernels = time_kernel * mass_kernel
                self.covar_module = gpytorch.kernels.ScaleKernel(inner_kernels)

            def forward(self, x):
                """Run the forward method of the model"""
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        # Prepare the training data by rescaling it
        x, y = self.training_data.get_training_data(
            label=self.datalabel, polarisation=b"+"
        )
        self.training_x = torch.tensor(
            x * self.other_input_factor, device=self.device, dtype=torch.float
        ).T[
            ::2
        ]  # [1, :398]
        training_y = self.training_y = torch.tensor(
            y * self.strain_input_factor, device=self.device, dtype=torch.float
        )[
            ::2
        ]  # [:398]

        _, y2 = self.training_data.get_training_data(
            label=self.datalabel, polarisation=b"x"
        )

        y_mean = training_y.mean()
        training_y -= y_mean
        self.training_y = training_y

        self.training_x[:, self.c_ind[b"time"]] += 0.40
        self.training_x[
            self.training_x[:, self.c_ind[b"time"]] < 0, self.c_ind[b"time"]
        ] = (
            self.training_x[
                self.training_x[:, self.c_ind[b"time"]] < 0, self.c_ind[b"time"]
            ]
            / 4
        )
        self.training_x[:, self.c_ind[b"time"]] -= 0.40

        self.training_y_cross = torch.tensor(
            y2 * self.strain_input_factor, device=self.device, dtype=torch.float
        )[
            ::2
        ]  # [:398]
        y_cross_mean = self.training_y_cross.mean()
        self.training_y_cross -= y_cross_mean

        model = ExactGPModel(self.training_x, self.training_y, likelihood)

        model_cross = ExactGPModel(
            self.training_x, self.training_y_cross, likelihood_cross
        )
        if self.device.type == "cuda":
            model = model.cuda()
            model_cross = model_cross.cuda()
            likelihood = likelihood.cuda()
            likelihood_cross = likelihood_cross.cuda()
            # Annoyingly this can't be passed-in as a device keyword

        # Right now we're not returning a different model for the cross-polarisation
        # that's clearly not the correct way to do things
        # TODO: Add the cross polarisation correctly.
        return [model, model_cross], [likelihood, likelihood_cross]

    def distribution(self, times, p, samples=100):
        """
        Return a number of sample waveforms from the GPR distribution.
        """
        times_b = times.clone()
        points = self._generate_eval_matrix(p, times_b)
        points = torch.tensor(points, device=self.device)  # .float().cuda()
        return_samples = []
        for polarisation in [self.model_plus, self.model_cross]:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                f_preds = polarisation(points)
                y_preds = self.likelihood(f_preds)

                return_samples.append(
                    [
                        Timeseries(
                            data=sample / self.strain_input_factor, times=times_b
                        )
                        for sample in y_preds.sample_n(samples)
                    ]
                )
        return return_samples

    def frequency_domain_waveform(self, p, times, window=None, polarisation=None):
        """
        Return the frequency domain waveform.

        Parameters
        ----------
        p : dict
           A dictionary of the parameter values for the waveform.
        window : array
           An array containing the window which should be applied
           to the data before performing the FFT.
        times : array, optional
           The times at which the model should be evaluated.
        polarisation : str
           The polarisation to return. Default is `None` in which case all available polarisations are returned.
        """

        window = window if window else torch.hamming_window

        timeseries = self.time_domain_waveform(times=times.clone(), p=p)
        frequencyseries = {
            pol: ts.to_frequencyseries(window=window) for pol, ts in timeseries.items()
        }

        polarisations = frequencyseries
        if "ra" in p.keys():
            ra, dec, psi, gpstime = p["ra"], p["dec"], p["psi"], p["gpstime"]
            detector = cached_detector_by_prefix[p["detector"]]
            response = self._get_antenna_response(detector, ra, dec, psi, gpstime)
            dt = TimeDelayFromEarthCenter(
                detector.location, ra, dec, LIGOTimeGPS(gpstime)
            )
            tshiftvec = torch.exp(
                1j
                * 2
                * torch.pi
                * dt
                * torch.tensor(frequencyseries["plus"].frequencies, device=self.device)
            )
            plus_r = torch.tensor(response.plus, device=self.device)
            cross_r = torch.tensor(response.cross, device=self.device)

            waveform_mean = (
                polarisations["plus"].data * plus_r
                + polarisations["cross"].data * cross_r
            ) * tshiftvec
            waveform_variance = (
                polarisations["plus"].variance * response.plus**2
                + polarisations["cross"].variance * response.cross**2
            )
            waveform = FrequencySeries(
                data=waveform_mean,
                variance=waveform_variance,
                frequencies=torch.tensor(
                    frequencyseries["plus"].frequencies, device=self.device
                ),
            )
        else:
            waveform = polarisations

        return waveform

    def time_domain_waveform(self, p, times=None):
        """
        Return the timedomain waveform.
        """
        defaults = {
            "before": 0.05,
            "after": 0.01,
            "pad before": 0.2,
            "pad after": 0.05
        }
        evals = defaults.copy()
        evals.update(p)
        p = evals
        
        if "distance" in p:
            # The distance in megaparsec
            distance = p["distance"]
        else:
            distance = 1

        if "total mass" in p:
            total_mass = p["total mass"]
        elif "mass 1" in p:
            mass_1 = p["mass 1"]
            mass_2 = p["mass 2"]
            total_mass = mass_1 + mass_2
            mass_ratio = mass_2 / mass_1
            p["mass ratio"] = mass_ratio
        else:
            total_mass = 20

        mass_factor = total_mass / self.reference_mass
        epoch = p["gpstime"]
        if isinstance(times, type(None)):
            times = torch.linspace(
                epoch - p["before"],
                epoch + p["after"],
                int(p["sample rate"] * (p["before"] + p["after"])),
                device=self.device,
                dtype=torch.float64,
            )

            diff = torch.min(torch.abs(times - epoch))
            #times += diff

            eval_times = torch.linspace(
                p['before'] / mass_factor, p['after'] / mass_factor, int((p['after']+p['before'])*p['sample rate']),
                device=self.device
            )

        else:
            eval_times = times / mass_factor
            
        if "ra" in p.keys():
            ra, dec, psi, gpstime = (
                p["ra"],
                p["dec"],
                torch.tensor(p["psi"]),
                p["gpstime"],
            )
            detector = cached_detector_by_prefix[p["detector"]]
            response = self._get_antenna_response(
                detector, ra, dec, float(psi), gpstime
            )
            dt = TimeDelayFromEarthCenter(
                detector.location, ra, dec, LIGOTimeGPS(gpstime)
            )
            polarisations = self.mean(eval_times, p)
            waveform_mean = polarisations["plus"].data * response.plus * torch.cos(
                psi
            ) + polarisations["cross"].data * response.cross * torch.sin(psi)

            shift = int(torch.round(dt / torch.diff(times-times[0])[0]))
            
            pre_pad = int(torch.round(p['pad before'] / torch.diff(times)[0]))
            post_pad = int(torch.round(p['pad after'] / torch.diff(times)[0]))
            waveform_mean = torch.nn.functional.pad(waveform_mean, (pre_pad, post_pad))
            waveform_mean = torch.roll(waveform_mean, shift)
            waveform_mean = waveform_mean[pre_pad:-post_pad]
            waveform_variance = polarisations["plus"].variance * torch.cos(
                psi
            ) * response.plus**2 + polarisations[
                "cross"
            ].variance * response.cross**2 * torch.sin(
                psi
            )
            waveform_variance = torch.nn.functional.pad(waveform_variance, (pre_pad, post_pad))
            waveform_variance = torch.roll(waveform_variance, shift)
            waveform_variance = waveform_variance[pre_pad:-post_pad]

            waveform_covariance = polarisations[
                "plus"
            ].covariance * response.plus**2 * torch.cos(psi) + polarisations[
                "cross"
            ].covariance * response.cross**2 * torch.sin(
                psi
            )

            waveform = Timeseries(
                data=mass_factor * waveform_mean / distance,
                variance=(mass_factor**2) * waveform_variance / distance**2,
                covariance=(mass_factor**2) * waveform_covariance / distance**2,
                times=times,
                detector=p["detector"],
            )
        else:
            polarisations = self.mean(eval_times, p)
            waveform = polarisations
        return waveform
