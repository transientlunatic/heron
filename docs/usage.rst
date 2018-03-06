=====
Usage
=====

Training a GP Regressor
-----------------------

A simple regression task involving just two dimensions can be handled
by all of `heron`'s built-in infrastructure at present.

First, import the parts of `heron` which we'll need:
::
   from heron import data, regression, priors, training

we'll also need to import the kernel module from George (but in the
near future `heron` will wrap this up in a sensible way too):
::
   from george import kernels

We now need to load in the training data; we can do that from a text
file. A sample training set of waveforms produced by the IMRPhenomPv2
approximant is supplied in the package repository (in the `data`
directory).
::
   training_data = np.loadtxt("IMRPhenomPv2_nonspinning_q1to10.dat")

   data = data.Data(training_data[:2,::5].T, training_data[2,::5],
                 label_sigma = [0.1],
                 target_names = ["t", "q"],
                 label_names = ["hx"],)
		
Heron's data class handles the preparation of the training data, the
selection of a test set, and the normalisation of the training data,
so we're now ready to pass the data to a GPR regression model.

First, however, it's necessary to define the covariance model.
::
   p0 = data.get_starting()
   hyper_priors = [priors.Normal(hyper, 1) for hyper in p0]
   kernel = np.std(data.labels) * kernels.Matern52Kernel(sep, ndim=len(sep))

Here we've set up a Matern-5/2 covariance function, and we've assigned
normal distribution priors to the value of each hyperparameter, which
are centred at a the value returned by the `get_starting()` method of
the data object.

In order to provide the best possible model, the Gaussian process must
now be trained. This can be done with the `heron.training` module.
::
   samples, burn = gp.train("MCMC")

Here we've trained the model using an MCMC process, and the samples
and the burn-in samples are returned. The training process sets the
hyperparameters of the kernel to the point in hyperparameter space
which maximises the GP's evidence. Finally, we can save the Gaussian
process, so that it can be called later.
::
   gp.save("IMRPhenomPv2_nonspinning.gp")
