"""
Code to train GPR waveform models from training data.

This module provides functionality to:
1. Load training data from HDF5 files
2. Train GPR models with diagnostic output
3. Save trained model states
4. Generate diagnostic plots during training
"""

import os
import logging
import click
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import torch
import gpytorch

from heron.training.data import DataWrapper
from heron.models.gpytorch import ExactGPModelKeOps
from heron.utils import load_yaml

logger = logging.getLogger("heron.train_gpr")

# Device selection
disable_cuda = False
if not disable_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class TrainingMonitor:
    """Monitor training progress and generate diagnostic plots."""

    def __init__(self, plots_dir, save_frequency=100):
        self.plots_dir = plots_dir
        self.save_frequency = save_frequency
        self.losses = {'plus': [], 'cross': []}
        self.lengthscales = {'plus': [], 'cross': []}
        self.noise_levels = {'plus': [], 'cross': []}
        os.makedirs(plots_dir, exist_ok=True)

    def record(self, polarization, iteration, loss, model):
        """Record training metrics at this iteration."""
        self.losses[polarization].append((iteration, loss))

        # Extract hyperparameters
        try:
            # Get lengthscales from the kernel
            lengthscales = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
            self.lengthscales[polarization].append((iteration, lengthscales))

            # Get noise level
            noise = model.likelihood.noise.detach().cpu().item()
            self.noise_levels[polarization].append((iteration, noise))
        except Exception as e:
            logger.warning(f"Could not extract hyperparameters: {e}")

    def should_save(self, iteration):
        """Check if we should save plots at this iteration."""
        return iteration % self.save_frequency == 0

    def save_diagnostics(self, iteration=None):
        """Save diagnostic plots."""
        suffix = f"_iter{iteration}" if iteration is not None else ""

        # Loss curves
        self._plot_losses(suffix)

        # Hyperparameter evolution
        self._plot_hyperparameters(suffix)

    def _plot_losses(self, suffix=""):
        """Plot training loss curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for idx, pol in enumerate(['plus', 'cross']):
            if self.losses[pol]:
                iterations, losses = zip(*self.losses[pol])
                axes[idx].plot(iterations, losses, 'b-', linewidth=1)
                axes[idx].set_xlabel('Iteration', fontsize=10)
                axes[idx].set_ylabel('Negative Log Likelihood', fontsize=10)
                axes[idx].set_title(f'{pol.capitalize()} Polarization', fontsize=11)
                axes[idx].grid(True, alpha=0.3)
                axes[idx].set_yscale('log')

        plt.suptitle('Training Loss Evolution', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f"training_loss{suffix}.png"), dpi=150)
        plt.close()

    def _plot_hyperparameters(self, suffix=""):
        """Plot hyperparameter evolution."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        for pol_idx, pol in enumerate(['plus', 'cross']):
            # Lengthscales
            if self.lengthscales[pol]:
                iterations, lengthscales = zip(*self.lengthscales[pol])
                lengthscales = np.array(lengthscales)

                for dim in range(lengthscales.shape[1]):
                    axes[0, pol_idx].plot(iterations, lengthscales[:, dim],
                                         label=f'Dim {dim}', linewidth=1)

                axes[0, pol_idx].set_xlabel('Iteration', fontsize=10)
                axes[0, pol_idx].set_ylabel('Lengthscale', fontsize=10)
                axes[0, pol_idx].set_title(f'{pol.capitalize()} - Lengthscales', fontsize=11)
                axes[0, pol_idx].legend(fontsize=8)
                axes[0, pol_idx].grid(True, alpha=0.3)
                axes[0, pol_idx].set_yscale('log')

            # Noise levels
            if self.noise_levels[pol]:
                iterations, noise = zip(*self.noise_levels[pol])
                axes[1, pol_idx].plot(iterations, noise, 'r-', linewidth=1)
                axes[1, pol_idx].set_xlabel('Iteration', fontsize=10)
                axes[1, pol_idx].set_ylabel('Noise Level', fontsize=10)
                axes[1, pol_idx].set_title(f'{pol.capitalize()} - Noise', fontsize=11)
                axes[1, pol_idx].grid(True, alpha=0.3)
                axes[1, pol_idx].set_yscale('log')

        plt.suptitle('Hyperparameter Evolution', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f"hyperparameters{suffix}.png"), dpi=150)
        plt.close()


def train_gpr_model(
    training_data_file,
    group_name="training",
    model_name="gpr_model",
    iterations=1000,
    learning_rate=0.05,
    output_scale=1e27,
    warp_scale=2,
    plots_dir="plots",
    checkpoint_frequency=100,
    validation_samples=None
):
    """
    Train a GPR model from training data.

    Parameters
    ----------
    training_data_file : str
        Path to HDF5 training data file
    group_name : str
        Name of the training data group in HDF5
    model_name : str
        Name for the trained model
    iterations : int
        Number of training iterations
    learning_rate : float
        Learning rate for optimizer
    output_scale : float
        Scale factor for output data
    warp_scale : float
        Time warping factor for pre-merger region
    plots_dir : str
        Directory for diagnostic plots
    checkpoint_frequency : int
        How often to save checkpoints and diagnostics
    validation_samples : int, optional
        Number of samples for validation (if None, use all)

    Returns
    -------
    DataWrapper
        The data wrapper with trained model state stored
    str
        Path to the plots directory
    """
    logger.info("Starting GPR model training")
    logger.info(f"Training data file: {training_data_file}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Iterations: {iterations}")
    logger.info(f"Device: {device}")

    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)

    # Load training data
    logger.info("Loading training data")
    data = DataWrapper(training_data_file, write=True)

    # Initialize training monitor
    monitor = TrainingMonitor(plots_dir, save_frequency=checkpoint_frequency)

    # Train models for each polarization
    models = {}

    for polarization in ['plus', 'cross']:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {polarization} polarization model")
        logger.info(f"{'='*60}")

        # Get training data for this polarization
        pol_char = 'p' if polarization == 'plus' else 'c'

        try:
            xdata, ydata = data.get_training_data(
                label=group_name,
                polarisation=pol_char,
                size=validation_samples
            )
        except Exception as e:
            logger.error(f"Failed to load training data for {polarization}: {e}")
            raise

        logger.info(f"Training data shape: X={xdata.shape}, Y={ydata.shape}")

        # Convert to torch tensors
        train_x = torch.tensor(xdata.T, dtype=torch.float32).to(device)
        train_y = torch.tensor(ydata, dtype=torch.float32).to(device) * output_scale

        # Apply time warping
        time_idx = 1  # Assuming time is the second dimension
        train_x_warped = train_x.clone()
        train_x_warped[train_x[:, time_idx] < 0, time_idx] /= warp_scale

        logger.info(f"Warped time axis with factor {warp_scale}")

        # Initialize model
        logger.info("Initializing GPR model")
        model = ExactGPModelKeOps(train_x_warped, train_y).to(device)
        model.likelihood.to(device)

        # Set training mode
        model.train()
        model.likelihood.train()

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Loss function
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

        logger.info(f"Starting training loop for {iterations} iterations")

        # Training loop with diagnostics
        for i in range(iterations):
            optimizer.zero_grad()
            output = model(train_x_warped)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

            # Record metrics
            loss_val = loss.item()
            monitor.record(polarization, i, loss_val, model)

            # Periodic logging and checkpointing
            if i % checkpoint_frequency == 0:
                logger.info(f"Iteration {i}/{iterations}: Loss = {loss_val:.4f}")

                # Save diagnostics
                monitor.save_diagnostics(iteration=i)

                # Save model state to HDF5
                try:
                    data.add_state(
                        name=f"{model_name}_{polarization}",
                        group=group_name,
                        data=model.state_dict()
                    )
                    logger.info(f"Saved checkpoint at iteration {i}")
                except Exception as e:
                    logger.warning(f"Could not save checkpoint: {e}")

        # Final evaluation mode
        model.eval()
        model.likelihood.eval()

        # Save final model state
        logger.info(f"Saving final {polarization} model state")
        data.add_state(
            name=f"{model_name}_{polarization}",
            group=group_name,
            data=model.state_dict()
        )

        models[polarization] = model
        logger.info(f"Completed training for {polarization} polarization")

    # Generate final diagnostic plots
    logger.info("Generating final diagnostic plots")
    monitor.save_diagnostics()

    # Generate validation plots
    _plot_model_predictions(data, group_name, models, plots_dir, output_scale, warp_scale)

    logger.info(f"GPR model training complete")
    logger.info(f"Model states saved to: {training_data_file}")
    logger.info(f"Diagnostic plots saved to: {plots_dir}")

    return data, plots_dir


def _plot_model_predictions(data, group_name, models, plots_dir, output_scale, warp_scale):
    """Generate plots comparing model predictions to training data."""
    logger.info("Generating model prediction comparison plots")

    for polarization in ['plus', 'cross']:
        pol_char = 'p' if polarization == 'plus' else 'c'

        try:
            # Get a subset of training data for visualization
            xdata, ydata = data.get_training_data(
                label=group_name,
                polarisation=pol_char,
                size=5000  # Sample for faster plotting
            )

            # Convert to torch
            train_x = torch.tensor(xdata.T, dtype=torch.float32).to(device)
            train_y = torch.tensor(ydata, dtype=torch.float32).to(device) * output_scale

            # Warp
            train_x_warped = train_x.clone()
            time_idx = 1
            train_x_warped[train_x[:, time_idx] < 0, time_idx] /= warp_scale

            # Get predictions
            model = models[polarization]
            model.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                predictions = model.likelihood(model(train_x_warped))
                pred_mean = predictions.mean.cpu().numpy() / output_scale
                pred_std = predictions.stddev.cpu().numpy() / output_scale

            true_y = ydata

            # Plot predictions vs truth
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))

            # Scatter plot
            axes[0].scatter(true_y, pred_mean, alpha=0.3, s=1)
            axes[0].plot([true_y.min(), true_y.max()],
                        [true_y.min(), true_y.max()],
                        'r--', linewidth=2, label='Perfect fit')
            axes[0].set_xlabel('True Strain', fontsize=10)
            axes[0].set_ylabel('Predicted Strain', fontsize=10)
            axes[0].set_title(f'{polarization.capitalize()} - Predictions vs Truth', fontsize=11)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Residuals
            residuals = pred_mean - true_y
            axes[1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            axes[1].axvline(0, color='r', linestyle='--', linewidth=2)
            axes[1].set_xlabel('Residual (Predicted - True)', fontsize=10)
            axes[1].set_ylabel('Count', fontsize=10)
            axes[1].set_title(f'Residual Distribution', fontsize=11)
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"predictions_{polarization}.png"), dpi=150)
            plt.close()

            logger.info(f"Created prediction plot for {polarization}")

        except Exception as e:
            logger.warning(f"Could not create prediction plot for {polarization}: {e}")


@click.command(name='train-gpr')
@click.option("--config", required=True, help="Path to GPR training configuration YAML file")
@click.option("--training-data", default=None, help="Override training data HDF5 file path")
@click.option("--iterations", default=None, type=int, help="Override number of training iterations")
def train_gpr(config, training_data, iterations):
    """
    Train a GPR waveform model from training data.

    This command loads training data from an HDF5 file and trains a GPR model
    with diagnostic output and checkpointing.
    """
    click.echo("Training GPR waveform model")

    # Load configuration
    settings = load_yaml(config)

    # Setup logging
    if "logging" in settings:
        level = settings.get("logging", {}).get("level", "warning")
        LOGGER_LEVELS = {
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "warning": logging.WARNING,
        }
        logging.basicConfig(level=LOGGER_LEVELS[level])

    # Extract settings
    training_config = settings.get('training', {})
    training_data_file = training_data or training_config.get('data_file')

    if not training_data_file:
        raise ValueError("No training data file specified. Use --training-data or set training.data_file in config")

    if not os.path.exists(training_data_file):
        raise FileNotFoundError(f"Training data file not found: {training_data_file}")

    group_name = training_config.get('group_name', 'training')
    model_name = settings.get('model_name', 'gpr_model')

    # Training hyperparameters
    hyperparams = settings.get('hyperparameters', {})
    num_iterations = iterations or hyperparams.get('iterations', 1000)
    learning_rate = hyperparams.get('learning_rate', 0.05)
    output_scale = hyperparams.get('output_scale', 1e27)
    warp_scale = hyperparams.get('warp_scale', 2)
    checkpoint_frequency = hyperparams.get('checkpoint_frequency', 100)
    validation_samples = hyperparams.get('validation_samples', None)

    # Determine plots directory
    if 'pages directory' in settings:
        plots_dir = os.path.join(settings['pages directory'], 'plots')
    else:
        plots_dir = 'plots'

    # Train model
    try:
        data, plots_path = train_gpr_model(
            training_data_file=training_data_file,
            group_name=group_name,
            model_name=model_name,
            iterations=num_iterations,
            learning_rate=learning_rate,
            output_scale=output_scale,
            warp_scale=warp_scale,
            plots_dir=plots_dir,
            checkpoint_frequency=checkpoint_frequency,
            validation_samples=validation_samples
        )

        click.echo(f"✓ GPR model training complete")
        click.echo(f"✓ Model states saved to: {training_data_file}")
        click.echo(f"✓ Diagnostic plots saved to: {plots_path}")

    except Exception as e:
        logger.exception(e)
        click.echo(f"✗ Error training GPR model: {e}", err=True)
        raise
