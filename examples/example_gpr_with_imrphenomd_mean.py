"""
Example: Training a GPR model with IMRPhenomD as mean function

This script demonstrates how to train a Gaussian Process Regression model
that uses IMRPhenomD (via ripple) as the mean function. The GP then learns
corrections to IMRPhenomD to match a more accurate target approximant.

The key advantage is that the GP only needs to learn small corrections
(~5% residuals) rather than the full waveform (100%), leading to:
- Fewer training points needed
- Better extrapolation
- Faster convergence
- More accurate predictions
"""

import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import heron modules
from heron.models.gpytorch import ExactGPModelKeOpsWithMean
from heron.models.mean_functions import IMRPhenomDMeanFunction


def create_gpr_with_phenomd_mean(
    total_mass=20.0,
    distance=100.0,
    train_x=None,
    train_y=None,
    polarization='plus',
    use_mean_function=True
):
    """
    Create a GPR model with optional IMRPhenomD mean function.

    Parameters
    ----------
    total_mass : float
        Total mass in solar masses
    distance : float
        Luminosity distance in Mpc
    train_x : torch.Tensor
        Training inputs [n_points, 2] with [mass_ratio, time]
    train_y : torch.Tensor
        Training targets (waveform values or residuals)
    polarization : str
        Polarization ('plus' or 'cross')
    use_mean_function : bool
        If True, use IMRPhenomD mean; if False, use zero mean

    Returns
    -------
    model : ExactGPModelKeOpsWithMean
        The GP model
    likelihood : gpytorch.likelihoods.GaussianLikelihood
        The likelihood function
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create mean function if requested
    mean_function = None
    if use_mean_function:
        print("Creating IMRPhenomD mean function...")
        mean_function = IMRPhenomDMeanFunction(
            total_mass=total_mass,
            distance=distance,
            delta_t=1.0/4096,
            f_lower=20.0,
            f_ref=20.0,
            device=device
        )
        print("Mean function created successfully")

    # Create likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # Create model
    model = ExactGPModelKeOpsWithMean(
        train_x=train_x.to(device),
        train_y=train_y.to(device),
        likelihood=likelihood,
        mean_function=mean_function,
        polarization=polarization
    )

    # Move to device
    model = model.to(device)
    likelihood = likelihood.to(device)

    return model, likelihood


def train_model(model, likelihood, train_x, train_y, iterations=100, learning_rate=0.05):
    """
    Train the GP model.

    Parameters
    ----------
    model : gpytorch.models.ExactGP
        The GP model to train
    likelihood : gpytorch.likelihoods.Likelihood
        The likelihood function
    train_x : torch.Tensor
        Training inputs
    train_y : torch.Tensor
        Training targets
    iterations : int
        Number of training iterations
    learning_rate : float
        Learning rate for optimizer

    Returns
    -------
    losses : list
        Training losses at each iteration
    """
    # Set to training mode
    model.train()
    likelihood.train()

    # Use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Use exact marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []

    print(f"\nTraining for {iterations} iterations...")
    for i in range(iterations):
        optimizer.zero_grad()

        # Forward pass
        output = model(train_x)
        loss = -mll(output, train_y)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{iterations} - Loss: {loss.item():.4f}")

    print("Training complete!")

    return losses


def evaluate_model(model, likelihood, test_x):
    """
    Evaluate the trained model on test points.

    Parameters
    ----------
    model : gpytorch.models.ExactGP
        The trained GP model
    likelihood : gpytorch.likelihoods.Likelihood
        The likelihood function
    test_x : torch.Tensor
        Test inputs

    Returns
    -------
    mean : torch.Tensor
        Predicted mean values
    variance : torch.Tensor
        Predicted variance values
    lower : torch.Tensor
        Lower confidence bound (mean - 2*std)
    upper : torch.Tensor
        Upper confidence bound (mean + 2*std)
    """
    # Set to eval mode
    model.eval()
    likelihood.eval()

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        mean = observed_pred.mean
        variance = observed_pred.variance
        lower = mean - 2 * variance.sqrt()
        upper = mean + 2 * variance.sqrt()

    return mean, variance, lower, upper


def example_simple_training():
    """
    Simple example: Create synthetic data and train a model.
    """
    print("="*60)
    print("Simple Training Example with IMRPhenomD Mean Function")
    print("="*60)

    # Parameters
    total_mass = 20.0  # Solar masses
    distance = 100.0   # Mpc
    mass_ratio = 0.8   # q = m2/m1

    # Create training data (simple synthetic example)
    # In a real scenario, this would come from actual waveform simulations
    n_train = 50
    time_range = torch.linspace(-0.1, 0.05, n_train)  # Geometric time
    mass_ratios = torch.ones(n_train) * mass_ratio

    train_x = torch.stack([mass_ratios, time_range], dim=1)

    # For this example, create synthetic training targets
    # In practice, these would be high-accuracy waveforms (e.g., from SEOBNRv4)
    train_y = torch.sin(time_range * 100) * torch.exp(time_range * 10)

    # Create model WITH mean function
    print("\n--- Model WITH IMRPhenomD mean function ---")
    model_with_mean, likelihood_with_mean = create_gpr_with_phenomd_mean(
        total_mass=total_mass,
        distance=distance,
        train_x=train_x,
        train_y=train_y,
        polarization='plus',
        use_mean_function=True
    )

    # Train model
    losses_with_mean = train_model(
        model_with_mean,
        likelihood_with_mean,
        train_x,
        train_y,
        iterations=50,
        learning_rate=0.05
    )

    # Create model WITHOUT mean function (zero mean baseline)
    print("\n--- Model WITHOUT mean function (zero mean baseline) ---")
    model_no_mean, likelihood_no_mean = create_gpr_with_phenomd_mean(
        total_mass=total_mass,
        distance=distance,
        train_x=train_x,
        train_y=train_y,
        polarization='plus',
        use_mean_function=False
    )

    # Train baseline model
    losses_no_mean = train_model(
        model_no_mean,
        likelihood_no_mean,
        train_x,
        train_y,
        iterations=50,
        learning_rate=0.05
    )

    # Plot training comparison
    plt.figure(figsize=(10, 4))
    plt.plot(losses_with_mean, label='With IMRPhenomD mean', linewidth=2)
    plt.plot(losses_no_mean, label='Zero mean (baseline)', linewidth=2, linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Negative Log Marginal Likelihood')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "gpr_phenomd_training_comparison.png", dpi=150)
    print(f"\nPlot saved to {output_dir / 'gpr_phenomd_training_comparison.png'}")

    # Make predictions
    test_x = torch.stack([
        torch.ones(100) * mass_ratio,
        torch.linspace(-0.1, 0.05, 100)
    ], dim=1)

    mean_with, var_with, lower_with, upper_with = evaluate_model(
        model_with_mean, likelihood_with_mean, test_x
    )
    mean_no, var_no, lower_no, upper_no = evaluate_model(
        model_no_mean, likelihood_no_mean, test_x
    )

    # Plot predictions
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(test_x[:, 1].cpu(), mean_with.cpu(), 'b-', label='Mean prediction', linewidth=2)
    plt.fill_between(
        test_x[:, 1].cpu(),
        lower_with.cpu(),
        upper_with.cpu(),
        alpha=0.3,
        label='95% confidence'
    )
    plt.scatter(train_x[:, 1].cpu(), train_y.cpu(), c='r', s=20, label='Training data', zorder=10)
    plt.xlabel('Time (M)')
    plt.ylabel('Waveform amplitude')
    plt.title('With IMRPhenomD Mean Function')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(test_x[:, 1].cpu(), mean_no.cpu(), 'b-', label='Mean prediction', linewidth=2)
    plt.fill_between(
        test_x[:, 1].cpu(),
        lower_no.cpu(),
        upper_no.cpu(),
        alpha=0.3,
        label='95% confidence'
    )
    plt.scatter(train_x[:, 1].cpu(), train_y.cpu(), c='r', s=20, label='Training data', zorder=10)
    plt.xlabel('Time (M)')
    plt.ylabel('Waveform amplitude')
    plt.title('Zero Mean (Baseline)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "gpr_phenomd_predictions_comparison.png", dpi=150)
    print(f"Plot saved to {output_dir / 'gpr_phenomd_predictions_comparison.png'}")

    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)


if __name__ == "__main__":
    # Run the simple example
    example_simple_training()
