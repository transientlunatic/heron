import math
import torch
import gpytorch
from gpytorch.kernels import RBFKernel
from matplotlib import pyplot as plt
import numpy as np
import tqdm


from functools import reduce
import operator
def prod(iterable):
    return reduce(operator.mul, iterable)

data = np.genfromtxt("../../heron/models/data/gt-M60-F1024.dat")

training_x = torch.tensor(data[:,:-2]*100).float().cuda()
training_y = torch.tensor(data[:,-2]*1e21).float().cuda()

mass_kernel = gpytorch.kernels.RBFKernel(active_dims=1, lengthscale_constraint=gpytorch.constraints.GreaterThan(10.))
time_kernel = gpytorch.kernels.RBFKernel(active_dims=0, lengthscale_constraint=gpytorch.constraints.GreaterThan(0.1))
spin_kernels = [gpytorch.kernels.RBFKernel(active_dims=dimension, lengthscale_constraint=gpytorch.constraints.GreaterThan(3.9)) for dimension in range(2,8)]


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            time_kernel*mass_kernel*prod(spin_kernels),
            lengthscale_constraint=gpytorch.constraints.LessThan(2.0) 
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.LessThan(0.8))
model = ExactGPModel(training_x, training_y, likelihood)

# Offload onto a CUDA device
model = model.cuda()
likelihood = likelihood.cuda()



training_iterations = 50000
model.train()
likelihood.train()

# We use SGD here, rather than Adam. Emperically, we find that SGD is better for variational regression
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.1)

# Our loss object. We're using the VariationalELBO
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


epochs_iter = tqdm.tqdm_notebook(range(training_iterations), desc="Epoch")
for i in epochs_iter:
    optimizer.zero_grad()
    # Output from model
    output = model(training_x)
    # Calc loss and backprop gradients
    loss = -mll(output, training_y).cuda()
    loss.backward()
    optimizer.step()
    if i%100==0:
        torch.save(model.state_dict(), 'model_state.pth')
