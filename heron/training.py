"""

These are functions designed to be used for training a Gaussian
process made using heron.


"""


from scipy.optimize import minimize
import emcee
import numpy as np



def ln_likelihood(p, gp):
    """
    Returns to log-likelihood of the Gaussian process, 
    which can be used to learn the hyperparameters of the GP.

    Parameters
    ----------
    gp : heron `Regressor` object
       The gaussian process to be evaluated
    p : array-like
       An array of the hyper-parameters at which the model is to be evaluated.

    Returns
    -------
    ln_likelihood : float
       The log-likelihood for the Gaussian process

    Notes
    -----
    * TODO Add the ability to specify the priors on each hyperparameter.
    """
    gp.gp.set_vector(p)
    return logp(p) * gp.gp.lnlikelihood(gp.training_y)



# MCMC Stuff using emcee

try:
    from IPython.core.display import clear_output
    ipython_available = True
except:
    ipython_available = False
    
def run_sampler(sampler, initial, iterations):
    """
    Run the MCMC sampler for some number of iterations, 
    but output a progress bar so you can keep track of what's going on
    """
    #sampler.run_mcmc(initial, 1)
    #for iteration in xrange(iterations/10-1):
    #    sampler.run_mcmc(None, 10)
    import progressbar

    with progressbar.ProgressBar(max_value=iterations) as bar:
        for iteration, sample in enumerate(sampler.sample(initial, iterations=iterations)):
            position = sample[0]
            with open("chain.dat", "a") as f:
                np.save(f, position)
            #    for k in range(position.shape[0]):
            #        f.write("{0:4d} {1:s}\n".format(str(k), " ".join(position[k])))
            if ipython_available: clear_output()
            bar.update(iteration)
    return sampler

def run_training_map(gp, metric = "loglikelihood"):
    """
    Find the maximum a posteriori training values for the Gaussian Process.

    Parameters
    ----------
    gp : heron.Regressor
       The Gaussian process object.
    metric : {"loglikelihood", "cv"}
       The metric to be used to train the MCMC. Defaults to log likelihood 
       (loglikelihood), which is the more traditionally Bayesian manner, but 
       cross-validation (cv) is also available.
    
    Notes
    -----
    The current implementation has no way of specifying the optimisation algorithm.

    * TODO Add an option to change the optimisation algorithm.

    """
    if metric=="loglikelihood":
        minfunc = gp.neg_ln_likelihood
    elif metric=="cv":
        minfunc = gp.neg_cross_validation
    
    MAP = minimize(minfunc, gp.gp.get_vector(),)
    gp.gp.set_vector(MAP.x)
    gp.update()


def run_training_mcmc(gp, walkers = 200, burn = 500, samples = 1000, metric = "loglikelihood", samplertype="ensemble"):
    """
    Train a Gaussian process using an MCMC process to find the maximum evidence.

    Parameters
    ----------
    gp : heron.Regressor
       The Gaussian process object.
    walkers : int
       The number of MCMC walkers.
    burn : int
       The number of samples to be used to evaluate the burn-in for the MCMC.
    samples : int
       The number of samples to be used for the production sampling.
    metric : {"loglikelihood", "cv"}
       The metric to be used to train the MCMC. Defaults to log likelihood 
       (loglikelihood), which is the more traditionally Bayesian manner, but 
       cross-validation (cv) is also available.
    samplertype : str {"ensemble", "pt"}
       The sampler to be used on the model.

    Returns
    -------
    samples : array
       The array of samples from the sampling chains.
    burn : array
       The array of samples from the burn-in chains.

    Notes
    -----
    At present the algorithm assigns the median of the samples to the 
    value of the kernel vector; this may not ultimately be the best 
    way to do this, and so it should be possible to specify the desired
    value to be used from the distribution.

    * TODO Add ability to change median to other statistics for training
    """
    start = gp.gp.get_vector()
    ndim, nwalkers, ntemps = len(start), walkers, 20
    
    #

    if metric=="loglikelihood":
        minfunc = ln_likelihood
    elif metric=="cv":
        minfunc = cross_validation
        
    if samplertype == "ensemble":
        p0 = [start for i in range(nwalkers)]
        p0 = np.random.uniform(low=-1.0, high=1.0, size=(nwalkers, ndim))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, minfunc, args=[gp], threads=4)
    elif samplertype == "pt":
        p0 = np.random.uniform(low=-1.0, high=1.0, size=(ntemps, nwalkers, ndim))
        sampler = emcee.PTSampler(ntemps, nwalkers, ndim, minfunc, logp, loglargs=[gp], threads=4)
    burn = run_sampler(sampler, p0, burn)
    sampler.reset()
    sampler = run_sampler(sampler, p0, samples)
    samples = sampler.chain[:, :, :].reshape((-1, ndim))
    gp.gp.set_vector(np.median(samples,axis=0))
    return samples, burn

def cross_validation(p, gp):
    """
    Calculate the cross-validation factor between the training set and the test set.

    Parameters
    ----------
    gp : heron.Regressor
       The Gaussian process object.
    p : array, optional
       The hyperparameters for the Gaussian process kernel. Defaults to None, which causes
       the current values for the hyperparameters to be used.

    Returns
    -------
    cv : float
       The cross validation of the test data and the model.
    """
    if p:
        old_p = gp.get_vector()
        gp.set_vector(p)
    prediction = gp.predict(gp.training_y, gp.training_oject.test_targets, return_var=True)
    
    return (gp.training_object.test_labels-prediction[0]).max()



def train_cv(gp):
    MAP = minimize(cross_validation, gp.gp.get_vector())
    gp.gp.set_vector(MAP.x)
    #gp.compute(training_x_batch, yerr=1e-6, seed=1234)
    return MAP


def logp(x):
    if np.any(np.abs(x) > 20): return -np.inf
    return 0.0
