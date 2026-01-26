"""
Plot actual waveform draws from trained GPR models.

This module provides functionality to:
1. Load trained GPR models from HDF5 files
2. Generate waveform samples from the posterior distribution
3. Visualize waveform draws with uncertainty quantification
4. Compare draws across different parameter values
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
from heron.models.gpytorch import ExactGPModelKeOps, ExactGPModelKeOpsWithMean
from heron.models.mean_functions import IMRPhenomDMeanFunction
from heron.utils import load_yaml

logger = logging.getLogger("heron.plot_waveforms")

# Device selection
disable_cuda = False
if not disable_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def load_trained_model(
    training_data_file,
    group_name="training",
    model_name="gpr_model",
    polarization="plus",
    use_mean_function=False,
    mean_function_params=None,
    warp_scale=2.0
):
    """
    Load a trained GPR model from an HDF5 file.

    Parameters
    ----------
    training_data_file : str
        Path to HDF5 file containing trained model state
    group_name : str
        Name of the training data group in HDF5
    model_name : str
        Name of the trained model
    polarization : str
        Which polarization to load ('plus' or 'cross')
    use_mean_function : bool
        Whether to use a mean function (e.g., IMRPhenomD)
    mean_function_params : dict, optional
    warp_scale : float
        Time warping factor used during training (default: 2.0)

    Returns
    -------
    model : gpytorch.models.ExactGP
        The loaded trained model
    train_x : torch.Tensor
        Training inputs
    train_y : torch.Tensor
        Training outputs
    """
    logger.info(f"Loading trained model: {model_name}_{polarization}")
    logger.info(f"From file: {training_data_file}")

    # Load data wrapper
    data = DataWrapper(training_data_file, write=False)

    # Get training data for this polarization
    pol_char = b'p' if polarization == 'plus' else b'c'

    try:
        xdata, ydata = data.get_training_data(
            label=group_name,
            polarisation=pol_char,
            size=None
        )
    except Exception as e:
        logger.error(f"Failed to load training data for {polarization}: {e}")
        raise

    logger.info(f"Loaded training data: X={xdata.shape}, Y={ydata.shape}")

    # Convert to torch tensors
    train_x = torch.from_numpy(np.ascontiguousarray(xdata.T, dtype=np.float32)).to(device)
    ydata_flat = ydata.flatten() if ydata.ndim > 1 else ydata
    train_y = torch.from_numpy(np.ascontiguousarray(ydata_flat, dtype=np.float32)).to(device)

    # Create the appropriate model
    mean_function = None
    if use_mean_function and mean_function_params:
        logger.info("Creating model with mean function")
        mean_function = IMRPhenomDMeanFunction(
            total_mass=mean_function_params.get('total_mass', 20.0),
            distance=mean_function_params.get('distance', 100.0),
            delta_t=mean_function_params.get('delta_t', 1.0/4096),
            f_lower=mean_function_params.get('f_lower', 20.0),
            f_ref=mean_function_params.get('f_ref', 20.0),
            device=device,
            polarization=polarization,
            warp_scale=warp_scale
        )
        model = ExactGPModelKeOpsWithMean(
            train_x,
            train_y,
            mean_function=mean_function,
            polarization=polarization
        ).to(device)
    else:
        logger.info("Creating model with zero mean")
        model = ExactGPModelKeOps(train_x, train_y).to(device)

    model.likelihood.to(device)

    # Load trained state
    state_name = f"{model_name}_{polarization}"
    try:
        state_data = data.get_states(name=state_name, device=device)
        logger.info(f"Loaded model state: {state_name}")

        # Extract the hyperparameters dict
        state_dict = state_data['hyperparameters']

        # Extract normalization parameters if they exist
        y_mean = None
        y_std = None
        if 'y_mean' in state_dict:
            y_mean = float(state_dict.pop('y_mean'))
            y_std = float(state_dict.pop('y_std'))
            logger.info(f"Found normalization parameters: mean={y_mean:.3e}, std={y_std:.3e}")

        # Remove mean function config from state_dict if present (not needed for loading)
        for key in list(state_dict.keys()):
            if key.startswith('mean_function_'):
                state_dict.pop(key)

        model.load_state_dict(state_dict)
        model.y_mean = y_mean
        model.y_std = y_std

        # Set normalization on mean function if it exists
        if use_mean_function and mean_function is not None and y_mean is not None and y_std is not None:
            mean_function.set_normalization(y_mean, y_std)
            logger.info(f"Set mean function normalization: y_mean={y_mean:.3e}, y_std={y_std:.3e}")
    except Exception as e:
        logger.error(f"Failed to load model state: {e}")
        raise

    # Set to evaluation mode
    model.eval()
    model.likelihood.eval()

    logger.info("Model loaded successfully")
    return model, train_x, train_y


def draw_waveform_samples(
    model,
    mass_ratio,
    times,
    num_samples=10,
    warp_scale=2,
    output_scale=1e27
):
    """
    Draw waveform samples from the trained GPR model.

    Parameters
    ----------
    model : gpytorch.models.ExactGP
        Trained GP model
    mass_ratio : float
        Mass ratio q = m2/m1
    times : array-like
        Time points at which to evaluate waveform
    num_samples : int
        Number of samples to draw from the posterior
    warp_scale : float
        Time warping factor for pre-merger region
    output_scale : float
        Scale factor for output data

    Returns
    -------
    samples : torch.Tensor
        Waveform samples [num_samples, len(times)]
    mean : torch.Tensor
        Mean prediction [len(times)]
    std : torch.Tensor
        Standard deviation [len(times)]
    test_x : torch.Tensor
        Input points [len(times), 2]
    """
    logger.info(f"Drawing {num_samples} waveform samples")
    logger.info(f"Mass ratio: {mass_ratio}, Time points: {len(times)}")

    # Create test points
    times_tensor = torch.tensor(times, dtype=torch.float32).to(device)
    mass_ratios = torch.ones_like(times_tensor) * mass_ratio
    test_x = torch.stack([mass_ratios, times_tensor], dim=1)

    # Apply time warping
    test_x_warped = test_x.clone()
    test_x_warped[test_x[:, 1] < 0, 1] /= warp_scale

    # Get predictions
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Get the posterior distribution
        posterior = model.likelihood(model(test_x_warped))

        # Draw samples
        samples = posterior.sample(sample_shape=torch.Size([num_samples]))

        # Get mean and std
        mean = posterior.mean
        std = posterior.stddev

    # Apply denormalization if available
    if hasattr(model, 'y_mean') and model.y_mean is not None:
        samples = samples * model.y_std + model.y_mean
        mean = mean * model.y_std + model.y_mean
        std = std * model.y_std

    logger.info("Waveform samples drawn successfully")
    return samples.cpu(), mean.cpu(), std.cpu(), test_x.cpu()


def plot_waveform_draws(
    times,
    samples,
    mean,
    std,
    mass_ratio,
    polarization,
    output_file,
    title=None,
    show_training_data=False,
    train_x=None,
    train_y=None
):
    """
    Create a plot of waveform draws from the posterior.

    Parameters
    ----------
    times : array-like
        Time points
    samples : torch.Tensor
        Waveform samples [num_samples, len(times)]
    mean : torch.Tensor
        Mean prediction
    std : torch.Tensor
        Standard deviation
    mass_ratio : float
        Mass ratio
    polarization : str
        Polarization ('plus' or 'cross')
    output_file : str
        Path to save the plot
    title : str, optional
        Custom title for the plot
    show_training_data : bool
        Whether to show training data points
    train_x : torch.Tensor, optional
        Training inputs (if show_training_data=True)
    train_y : torch.Tensor, optional
        Training outputs (if show_training_data=True)
    """
    logger.info(f"Creating waveform draws plot: {output_file}")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot individual samples
    for i in range(samples.shape[0]):
        ax.plot(
            times,
            samples[i].numpy(),
            alpha=0.3,
            linewidth=0.8,
            color='steelblue',
            label='Posterior samples' if i == 0 else None
        )

    # Plot mean
    ax.plot(
        times,
        mean.numpy(),
        'r-',
        linewidth=2,
        label='Mean prediction',
        zorder=10
    )

    # Plot confidence interval
    ax.fill_between(
        times,
        (mean - 2*std).numpy(),
        (mean + 2*std).numpy(),
        alpha=0.2,
        color='red',
        label='95% confidence',
        zorder=5
    )

    # Optionally show training data
    if show_training_data and train_x is not None and train_y is not None:
        # Filter training data for this mass ratio
        mask = torch.abs(train_x[:, 0] - mass_ratio) < 0.01
        if mask.sum() > 0:
            train_times = train_x[mask, 1].numpy()
            train_vals = train_y[mask].numpy()

            # Apply denormalization if available
            if hasattr(train_y, 'mean') and hasattr(train_y, 'std'):
                train_vals = train_vals * train_y.std + train_y.mean

            ax.scatter(
                train_times,
                train_vals,
                c='black',
                s=10,
                alpha=0.5,
                label='Training data',
                zorder=15
            )

    # Labels and formatting
    ax.set_xlabel('Time (geometric units)', fontsize=12)
    ax.set_ylabel('Strain', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(
            f'{polarization.capitalize()} Polarization Waveform Draws (q={mass_ratio:.2f})',
            fontsize=14
        )

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Plot saved to {output_file}")


def plot_multiple_mass_ratios(
    model,
    mass_ratios,
    times,
    polarization,
    output_file,
    num_samples=5,
    warp_scale=2,
    output_scale=1e27
):
    """
    Create a comparison plot of waveform draws at different mass ratios.

    Parameters
    ----------
    model : gpytorch.models.ExactGP
        Trained GP model
    mass_ratios : list of float
        Mass ratios to compare
    times : array-like
        Time points
    polarization : str
        Polarization
    output_file : str
        Path to save the plot
    num_samples : int
        Number of samples per mass ratio
    warp_scale : float
        Time warping factor
    output_scale : float
        Output scale factor
    """
    logger.info(f"Creating comparison plot for {len(mass_ratios)} mass ratios")

    n_rows = len(mass_ratios)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4*n_rows), sharex=True)

    if n_rows == 1:
        axes = [axes]

    for idx, mass_ratio in enumerate(mass_ratios):
        ax = axes[idx]

        # Draw samples
        samples, mean, std, _ = draw_waveform_samples(
            model,
            mass_ratio,
            times,
            num_samples=num_samples,
            warp_scale=warp_scale,
            output_scale=output_scale
        )

        # Plot samples
        for i in range(samples.shape[0]):
            ax.plot(
                times,
                samples[i].numpy(),
                alpha=0.4,
                linewidth=0.8,
                color='steelblue',
                label='Posterior samples' if i == 0 else None
            )

        # Plot mean
        ax.plot(
            times,
            mean.numpy(),
            'r-',
            linewidth=2,
            label='Mean',
            zorder=10
        )

        # Plot confidence
        ax.fill_between(
            times,
            (mean - 2*std).numpy(),
            (mean + 2*std).numpy(),
            alpha=0.2,
            color='red',
            label='95% confidence',
            zorder=5
        )

        ax.set_ylabel('Strain', fontsize=11)
        ax.set_title(f'q = {mass_ratio:.2f}', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (geometric units)', fontsize=12)

    plt.suptitle(
        f'{polarization.capitalize()} Polarization Waveforms at Different Mass Ratios',
        fontsize=14,
        y=0.995
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Comparison plot saved to {output_file}")


@click.command(name='plot-waveforms')
@click.option("--config", required=True, help="Path to configuration YAML file")
@click.option("--model-file", default=None, help="Override model HDF5 file path")
@click.option("--mass-ratio", default=None, type=float, help="Mass ratio for single waveform plot")
@click.option("--num-samples", default=10, type=int, help="Number of waveform samples to draw")
@click.option("--polarization", default="plus", type=click.Choice(['plus', 'cross', 'both']),
              help="Which polarization(s) to plot")
@click.option("--compare-mass-ratios", is_flag=True,
              help="Create comparison plot across multiple mass ratios")
def plot_waveforms(config, model_file, mass_ratio, num_samples, polarization, compare_mass_ratios):
    """
    Plot waveform draws from trained GPR models.

    This command loads a trained GPR model and generates plots showing
    posterior samples of waveforms, illustrating the uncertainty in the predictions.
    """
    click.echo("Plotting waveform draws from trained GPR model")

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
    model_data_file = model_file or training_config.get('data_file')

    if not model_data_file:
        raise ValueError("No model file specified. Use --model-file or set training.data_file in config")

    if not os.path.exists(model_data_file):
        raise FileNotFoundError(f"Model file not found: {model_data_file}")

    group_name = training_config.get('group_name', 'training')
    model_name = settings.get('model_name', 'gpr_model')

    # Hyperparameters
    hyperparams = settings.get('hyperparameters', {})
    warp_scale = hyperparams.get('warp_scale', 2)
    output_scale = hyperparams.get('output_scale', 1e27)

    # Mean function settings
    use_mean_function = settings.get('mean_function', {}).get('enabled', False)
    mean_function_params = settings.get('mean_function', {}).get('parameters', {})

    # Time range for plotting
    plot_config = settings.get('plotting', {})
    time_min = plot_config.get('time_min', -0.1)
    time_max = plot_config.get('time_max', 0.05)
    time_points = plot_config.get('time_points', 500)
    times = np.linspace(time_min, time_max, time_points)

    # Mass ratio(s) for plotting
    if compare_mass_ratios:
        mass_ratios = plot_config.get('mass_ratios', [0.5, 0.7, 0.9, 1.0])
    else:
        if mass_ratio is None:
            mass_ratio = plot_config.get('mass_ratio', 0.8)
        mass_ratios = [mass_ratio]

    # Output directory
    if 'pages directory' in settings:
        output_dir = os.path.join(settings['pages directory'], 'plots')
    else:
        output_dir = plot_config.get('output_dir', 'plots')

    os.makedirs(output_dir, exist_ok=True)

    # Determine which polarizations to plot
    polarizations = []
    if polarization == 'both':
        polarizations = ['plus', 'cross']
    else:
        polarizations = [polarization]

    # Generate plots for each polarization
    for pol in polarizations:
        try:
            click.echo(f"\nProcessing {pol} polarization...")

            # Load trained model
            model, train_x, train_y = load_trained_model(
                training_data_file=model_data_file,
                group_name=group_name,
                model_name=model_name,
                polarization=pol,
                use_mean_function=use_mean_function,
                mean_function_params=mean_function_params,
                warp_scale=warp_scale
            )

            if compare_mass_ratios:
                # Create comparison plot
                output_file = os.path.join(
                    output_dir,
                    f"waveform_draws_comparison_{pol}.png"
                )
                plot_multiple_mass_ratios(
                    model=model,
                    mass_ratios=mass_ratios,
                    times=times,
                    polarization=pol,
                    output_file=output_file,
                    num_samples=num_samples,
                    warp_scale=warp_scale,
                    output_scale=output_scale
                )
                click.echo(f"✓ Comparison plot saved to: {output_file}")
            else:
                # Create single mass ratio plot
                for q in mass_ratios:
                    samples, mean, std, test_x = draw_waveform_samples(
                        model=model,
                        mass_ratio=q,
                        times=times,
                        num_samples=num_samples,
                        warp_scale=warp_scale,
                        output_scale=output_scale
                    )

                    output_file = os.path.join(
                        output_dir,
                        f"waveform_draws_{pol}_q{q:.2f}.png"
                    )

                    plot_waveform_draws(
                        times=times,
                        samples=samples,
                        mean=mean,
                        std=std,
                        mass_ratio=q,
                        polarization=pol,
                        output_file=output_file,
                        show_training_data=plot_config.get('show_training_data', False),
                        train_x=train_x,
                        train_y=train_y
                    )

                    click.echo(f"✓ Waveform plot saved to: {output_file}")

        except Exception as e:
            logger.exception(e)
            click.echo(f"✗ Error plotting {pol} polarization: {e}", err=True)
            raise

    click.echo("\n✓ All waveform plots generated successfully")


# =============================================================================
# DIAGNOSTIC FUNCTIONS
# =============================================================================

def diagnose_training_data(training_data_file, group_name, polarization, output_dir):
    """
    Generate diagnostic plots for training data.
    """
    logger.info(f"Diagnosing training data: {training_data_file}")

    data = DataWrapper(training_data_file, write=False)
    pol_char = b'p' if polarization == 'plus' else b'c'

    xdata, ydata = data.get_training_data(
        label=group_name, polarisation=pol_char, size=None
    )

    mass_ratios = xdata[0, :]
    times = xdata[1, :]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Parameter space coverage
    ax = axes[0, 0]
    scatter = ax.scatter(times, mass_ratios, c=ydata, s=1, alpha=0.5, cmap='RdBu_r')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mass Ratio')
    ax.set_title('Training Data Coverage\n(color = strain)')
    plt.colorbar(scatter, ax=ax, label='Strain')

    # 2. Strain histogram
    ax = axes[0, 1]
    ax.hist(ydata, bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(ydata.mean(), color='r', linestyle='--', label=f'Mean: {ydata.mean():.2e}')
    ax.axvline(ydata.mean() + ydata.std(), color='orange', linestyle=':')
    ax.axvline(ydata.mean() - ydata.std(), color='orange', linestyle=':',
               label=f'Std: {ydata.std():.2e}')
    ax.set_xlabel('Strain')
    ax.set_ylabel('Count')
    ax.set_title('Strain Distribution')
    ax.legend(fontsize=8)

    # 3. Time distribution
    ax = axes[0, 2]
    ax.hist(times, bins=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Count')
    ax.set_title(f'Time Distribution\nRange: [{times.min():.4f}, {times.max():.4f}]')

    # 4. Mass ratio distribution
    ax = axes[1, 0]
    ax.hist(mass_ratios, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Mass Ratio')
    ax.set_ylabel('Count')
    ax.set_title(f'Mass Ratio Distribution\nRange: [{mass_ratios.min():.3f}, {mass_ratios.max():.3f}]')

    # 5. Waveform slices at different mass ratios
    ax = axes[1, 1]
    unique_q = np.unique(mass_ratios)
    sample_qs = unique_q[::max(1, len(unique_q)//5)][:5]

    for q in sample_qs:
        mask = np.abs(mass_ratios - q) < 0.01
        if mask.sum() > 0:
            t_slice = times[mask]
            y_slice = ydata[mask]
            sort_idx = np.argsort(t_slice)
            ax.plot(t_slice[sort_idx], y_slice[sort_idx], label=f'q={q:.2f}', alpha=0.7)

    ax.set_xlabel('Time')
    ax.set_ylabel('Strain')
    ax.set_title('Waveform Slices (Raw Training Data)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Data quality summary
    ax = axes[1, 2]
    ax.axis('off')

    n_samples = len(ydata)
    n_nan = np.isnan(ydata).sum()
    n_inf = np.isinf(ydata).sum()
    outlier_mask = np.abs(ydata - ydata.mean()) > 5 * ydata.std()
    n_outliers = outlier_mask.sum()

    quality_text = f"""DATA QUALITY SUMMARY
{'='*35}
Total samples:     {n_samples:,}
NaN values:        {n_nan} {'WARNING' if n_nan > 0 else 'OK'}
Inf values:        {n_inf} {'WARNING' if n_inf > 0 else 'OK'}
Outliers (>5s):    {n_outliers} ({100*n_outliers/n_samples:.2f}%)

STRAIN STATISTICS
{'='*35}
Min:               {ydata.min():.4e}
Max:               {ydata.max():.4e}
Mean:              {ydata.mean():.4e}
Std:               {ydata.std():.4e}

COORDINATE RANGES
{'='*35}
Time:              [{times.min():.4f}, {times.max():.4f}]
Mass ratio:        [{mass_ratios.min():.3f}, {mass_ratios.max():.3f}]
Unique mass ratios: {len(unique_q)}"""

    ax.text(0.1, 0.9, quality_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')

    plt.suptitle(f'Training Data Diagnostics: {polarization}', fontsize=14)
    plt.tight_layout()

    output_file = os.path.join(output_dir, f'diagnostics_data_{polarization}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Training data diagnostics saved to {output_file}")
    return {
        'n_samples': n_samples, 'n_nan': n_nan, 'n_inf': n_inf,
        'y_mean': ydata.mean(), 'y_std': ydata.std(),
        'time_range': (times.min(), times.max()),
        'q_range': (mass_ratios.min(), mass_ratios.max()),
        'unique_q': unique_q
    }


def diagnose_hyperparameters(model, polarization, output_dir):
    """
    Inspect trained model hyperparameters.
    """
    logger.info("Diagnosing model hyperparameters")

    hypers = {}

    # Noise
    hypers['noise_variance'] = model.likelihood.noise.item()
    hypers['noise_std'] = np.sqrt(hypers['noise_variance'])

    # Kernel parameters
    if hasattr(model.covar_module, 'outputscale'):
        hypers['output_scale'] = model.covar_module.outputscale.item()

    if hasattr(model.covar_module, 'base_kernel'):
        base = model.covar_module.base_kernel
        if hasattr(base, 'kernels'):
            for i, k in enumerate(base.kernels):
                if hasattr(k, 'lengthscale'):
                    ls = k.lengthscale.detach().cpu().numpy().flatten()[0]
                    dim = 'mass_ratio' if i == 0 else 'time'
                    hypers[f'lengthscale_{dim}'] = ls
        elif hasattr(base, 'lengthscale'):
            hypers['lengthscale'] = base.lengthscale.detach().cpu().item()

    # Normalization
    if hasattr(model, 'y_mean') and model.y_mean is not None:
        hypers['y_mean'] = model.y_mean
        hypers['y_std'] = model.y_std

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    ax = axes[0]
    names = [k for k, v in hypers.items() if isinstance(v, (int, float, np.floating))]
    values = [hypers[k] for k in names]
    colors = []
    for n, v in zip(names, values):
        if 'lengthscale' in n:
            colors.append('red' if v < 0.001 or v > 10 else 'green')
        elif 'noise' in n:
            colors.append('orange' if v > 1 else 'green')
        else:
            colors.append('steelblue')

    bars = ax.barh(names, values, color=colors)
    ax.set_xlabel('Value')
    ax.set_title('Trained Hyperparameters\n(Green=OK, Orange=Check, Red=Issue)')
    ax.set_xscale('symlog', linthresh=1e-6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f' {val:.4e}', va='center', fontsize=9)

    # Summary text
    ax = axes[1]
    ax.axis('off')

    txt = f"""HYPERPARAMETER ANALYSIS: {polarization}
{'='*50}

KERNEL PARAMETERS:
  Output scale:        {hypers.get('output_scale', 'N/A')}
"""
    if 'lengthscale_mass_ratio' in hypers:
        ls_q = hypers['lengthscale_mass_ratio']
        txt += f"  Lengthscale (q):     {ls_q:.4e}\n"
        txt += f"    -> Correlation ~{ls_q:.2f} in mass ratio\n"
    if 'lengthscale_time' in hypers:
        ls_t = hypers['lengthscale_time']
        txt += f"  Lengthscale (time):  {ls_t:.4e}\n"
        txt += f"    -> Correlation ~{ls_t:.4f} in time\n"

    txt += f"""
NOISE MODEL:
  Noise variance:      {hypers['noise_variance']:.4e}
  Noise std:           {hypers['noise_std']:.4e}
"""
    if 'y_mean' in hypers:
        txt += f"""
NORMALIZATION:
  y_mean:              {hypers['y_mean']:.4e}
  y_std:               {hypers['y_std']:.4e}
"""

    txt += "\nPOTENTIAL ISSUES:\n"
    issues = []
    if hypers.get('lengthscale_mass_ratio', 1) > 5:
        issues.append("- Mass ratio lengthscale very large")
    if hypers.get('lengthscale_time', 0.01) > 0.1:
        issues.append("- Time lengthscale large - may miss rapid changes")
    if hypers.get('lengthscale_time', 0.01) < 0.001:
        issues.append("- Time lengthscale very small - may overfit")
    if hypers['noise_variance'] > 1:
        issues.append("- High noise level")
    if not issues:
        issues.append("- No obvious issues")
    txt += '\n'.join(issues)

    ax.text(0.05, 0.95, txt, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    output_file = os.path.join(output_dir, f'diagnostics_hypers_{polarization}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Hyperparameter diagnostics saved to {output_file}")
    return hypers


def diagnose_predictions(model, train_x, train_y, polarization, output_dir, warp_scale=2):
    """
    Compare model predictions against training data.
    """
    logger.info("Diagnosing predictions vs training data")

    n_train = train_x.shape[0]
    n_test = min(2000, n_train)
    idx = np.random.choice(n_train, n_test, replace=False) if n_train > n_test else np.arange(n_train)

    test_x = train_x[idx]
    test_y = train_y[idx]

    # Warp
    test_x_warped = test_x.clone()
    test_x_warped[test_x[:, 1] < 0, 1] /= warp_scale

    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.likelihood(model(test_x_warped))
        pred_mean = posterior.mean.cpu().numpy()
        pred_std = posterior.stddev.cpu().numpy()

    test_x_np = test_x.cpu().numpy()
    test_y_np = test_y.cpu().numpy()

    # Denormalize predictions (training data is in original scale)
    if hasattr(model, 'y_mean') and model.y_mean is not None:
        pred_mean = pred_mean * model.y_std + model.y_mean
        pred_std = pred_std * model.y_std

    residuals = test_y_np - pred_mean
    norm_resid = residuals / pred_std

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Predicted vs Actual
    ax = axes[0, 0]
    ax.scatter(test_y_np, pred_mean, alpha=0.3, s=5)
    lims = [min(test_y_np.min(), pred_mean.min()), max(test_y_np.max(), pred_mean.max())]
    ax.plot(lims, lims, 'r--', label='Perfect')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Predicted vs Actual')

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((test_y_np - test_y_np.mean())**2)
    r2 = 1 - ss_res / ss_tot
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, fontsize=10, va='top')
    ax.legend()

    # 2. Residual histogram
    ax = axes[0, 1]
    ax.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Density')
    ax.set_title(f'Residuals (mean={residuals.mean():.2e})')

    # 3. Normalized residuals
    ax = axes[0, 2]
    ax.hist(norm_resid, bins=50, density=True, alpha=0.7, edgecolor='black')
    x_norm = np.linspace(-4, 4, 100)
    ax.plot(x_norm, np.exp(-x_norm**2/2)/np.sqrt(2*np.pi), 'r-', label='N(0,1)')
    ax.set_xlabel('Normalized Residual')
    ax.set_ylabel('Density')
    ax.set_title('Normalized Residuals\n(should match red)')
    ax.legend()

    # 4. Residuals vs Time
    ax = axes[1, 0]
    sc = ax.scatter(test_x_np[:, 1], residuals, c=test_x_np[:, 0], alpha=0.3, s=5, cmap='viridis')
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Time')
    plt.colorbar(sc, ax=ax, label='q')

    # 5. Residuals vs Mass Ratio
    ax = axes[1, 1]
    sc = ax.scatter(test_x_np[:, 0], residuals, c=test_x_np[:, 1], alpha=0.3, s=5, cmap='coolwarm')
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Mass Ratio')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Mass Ratio')
    plt.colorbar(sc, ax=ax, label='Time')

    # 6. Summary
    ax = axes[1, 2]
    ax.axis('off')

    txt = f"""PREDICTION DIAGNOSTICS: {polarization}
{'='*45}

OVERALL FIT:
  R² score:          {r2:.4f}
  RMSE:              {np.sqrt(np.mean(residuals**2)):.4e}
  Mean residual:     {residuals.mean():.4e}

NORMALIZED RESIDUALS:
  Mean:              {norm_resid.mean():.3f} (want ~0)
  Std:               {norm_resid.std():.3f} (want ~1)
  |z| > 2:           {100*np.mean(np.abs(norm_resid) > 2):.1f}%
  |z| > 3:           {100*np.mean(np.abs(norm_resid) > 3):.1f}%

INTERPRETATION:
"""
    issues = []
    if r2 < 0.5:
        issues.append("- CRITICAL: R² < 0.5, model is poor")
    elif r2 < 0.9:
        issues.append("- R² < 0.9, moderate fit")
    if abs(residuals.mean()) > 0.1 * residuals.std():
        issues.append("- Systematic bias in residuals")
    if norm_resid.std() > 1.5:
        issues.append("- Underconfident predictions")
    if norm_resid.std() < 0.5:
        issues.append("- Overconfident predictions")
    if not issues:
        issues.append("- Predictions look reasonable")
    txt += '\n'.join(issues)

    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10, va='top', fontfamily='monospace')

    plt.tight_layout()
    output_file = os.path.join(output_dir, f'diagnostics_predictions_{polarization}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Prediction diagnostics saved to {output_file}")
    return {'r2': r2, 'rmse': np.sqrt(np.mean(residuals**2))}


def diagnose_reconstruction(model, train_x, train_y, polarization, output_dir, warp_scale=2):
    """
    Compare reconstructed waveforms to training data slices.
    """
    logger.info("Diagnosing waveform reconstruction")

    train_x_np = train_x.cpu().numpy()
    train_y_np = train_y.cpu().numpy()

    unique_q = np.unique(train_x_np[:, 0])
    sample_qs = unique_q[::max(1, len(unique_q)//4)][:4]

    fig, axes = plt.subplots(len(sample_qs), 1, figsize=(12, 4*len(sample_qs)))
    if len(sample_qs) == 1:
        axes = [axes]

    for idx, q in enumerate(sample_qs):
        ax = axes[idx]

        mask = np.abs(train_x_np[:, 0] - q) < 0.01
        if mask.sum() == 0:
            ax.text(0.5, 0.5, f'No data for q={q:.2f}', transform=ax.transAxes, ha='center')
            continue

        t_train = train_x_np[mask, 1]
        y_train = train_y_np[mask]
        sort_idx = np.argsort(t_train)
        t_train, y_train = t_train[sort_idx], y_train[sort_idx]

        # Predict
        test_x = torch.tensor(np.stack([np.ones_like(t_train)*q, t_train], axis=1),
                              dtype=torch.float32).to(device)
        test_x_warped = test_x.clone()
        test_x_warped[test_x[:, 1] < 0, 1] /= warp_scale

        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post = model.likelihood(model(test_x_warped))
            pred_mean = post.mean.cpu().numpy()
            pred_std = post.stddev.cpu().numpy()

        if hasattr(model, 'y_mean') and model.y_mean is not None:
            pred_mean = pred_mean * model.y_std + model.y_mean
            pred_std = pred_std * model.y_std

        ax.plot(t_train, y_train, 'k-', lw=1.5, label='Training data', alpha=0.8)
        ax.plot(t_train, pred_mean, 'r--', lw=1.5, label='GP prediction')
        ax.fill_between(t_train, pred_mean - 2*pred_std, pred_mean + 2*pred_std,
                        alpha=0.2, color='red', label='95% CI')
        ax.set_xlabel('Time')
        ax.set_ylabel('Strain')
        ax.set_title(f'q = {q:.3f}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        rmse = np.sqrt(np.mean((y_train - pred_mean)**2))
        ax.text(0.02, 0.98, f'RMSE: {rmse:.2e}', transform=ax.transAxes, fontsize=9, va='top')

    plt.suptitle(f'Waveform Reconstruction: {polarization}', fontsize=14)
    plt.tight_layout()

    output_file = os.path.join(output_dir, f'diagnostics_reconstruction_{polarization}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Reconstruction diagnostics saved to {output_file}")


@click.command(name='diagnose-model')
@click.option("--config", required=True, help="Path to configuration YAML file")
@click.option("--model-file", default=None, help="Override model HDF5 file path")
@click.option("--polarization", default="plus", type=click.Choice(['plus', 'cross', 'both']),
              help="Which polarization(s) to diagnose")
@click.option("--skip-match", is_flag=True, help="Skip match computation against approximant")
@click.option("--approximant", default="IMRPhenomPv2", help="Reference approximant for match")
def diagnose_model(config, model_file, polarization, skip_match, approximant):
    """
    Run comprehensive diagnostics on a trained GPR model.

    Generates diagnostic plots to identify training issues:
    - Training data quality and coverage
    - Learned hyperparameters analysis
    - Prediction accuracy vs training data
    - Waveform reconstruction quality
    - Match against reference approximant (e.g., IMRPhenomPv2)
    """
    click.echo("Running model diagnostics...")

    settings = load_yaml(config)

    if "logging" in settings:
        level = settings.get("logging", {}).get("level", "warning")
        levels = {"info": logging.INFO, "debug": logging.DEBUG, "warning": logging.WARNING}
        logging.basicConfig(level=levels.get(level, logging.WARNING))

    training_config = settings.get('training', {})
    model_data_file = model_file or training_config.get('data_file')

    if not model_data_file or not os.path.exists(model_data_file):
        raise FileNotFoundError(f"Model file not found: {model_data_file}")

    group_name = training_config.get('group_name', 'training')
    model_name = settings.get('model_name', 'gpr_model')
    warp_scale = settings.get('hyperparameters', {}).get('warp_scale', 2)

    # Get reference waveform parameters (from mean_function config or defaults)
    mean_func_config = settings.get('mean_function', {})
    total_mass = mean_func_config.get('total_mass', 20.0)
    distance = mean_func_config.get('distance', 100.0)

    if 'pages directory' in settings:
        output_dir = os.path.join(settings['pages directory'], 'diagnostics')
    else:
        output_dir = settings.get('plotting', {}).get('output_dir', 'diagnostics')
    os.makedirs(output_dir, exist_ok=True)

    pols = ['plus', 'cross'] if polarization == 'both' else [polarization]

    for pol in pols:
        click.echo(f"\n{'='*50}")
        click.echo(f"Diagnosing {pol} polarization")
        click.echo('='*50)

        click.echo("\n1. Analyzing training data...")
        stats = diagnose_training_data(model_data_file, group_name, pol, output_dir)
        click.echo(f"   Found {stats['n_samples']:,} samples")

        click.echo("\n2. Loading trained model...")
        try:
            model, train_x, train_y = load_trained_model(
                model_data_file, group_name, model_name, pol,
                warp_scale=warp_scale
            )
            click.echo("   Model loaded")

            click.echo("\n3. Analyzing hyperparameters...")
            hypers = diagnose_hyperparameters(model, pol, output_dir)
            for k, v in hypers.items():
                if isinstance(v, (int, float, np.floating)):
                    click.echo(f"   {k}: {v:.4e}")

            click.echo("\n4. Analyzing predictions...")
            pred_stats = diagnose_predictions(model, train_x, train_y, pol, output_dir, warp_scale)
            click.echo(f"   R² = {pred_stats['r2']:.4f}")

            click.echo("\n5. Checking reconstruction...")
            diagnose_reconstruction(model, train_x, train_y, pol, output_dir, warp_scale)
            click.echo("   Done")

            if not skip_match:
                click.echo(f"\n6. Computing match against {approximant}...")
                try:
                    match_stats = diagnose_match_vs_approximant(
                        model=model,
                        train_x=train_x,
                        polarization=pol,
                        output_dir=output_dir,
                        warp_scale=warp_scale,
                        total_mass=total_mass,
                        distance=distance,
                        approximant_name=approximant
                    )
                    if match_stats:
                        click.echo(f"   Mean match: {match_stats['mean_match']:.4f}")
                        click.echo(f"   Min match:  {match_stats['min_match']:.4f}")
                        click.echo(f"   Max match:  {match_stats['max_match']:.4f}")
                except Exception as e:
                    click.echo(f"   Warning: Match computation failed: {e}", err=True)
                    logger.warning(f"Match computation failed: {e}")

        except Exception as e:
            click.echo(f"   Error: {e}", err=True)
            logger.exception(e)

    click.echo(f"\nDiagnostics complete! Check {output_dir}/")


# =============================================================================
# WAVEFORM MATCH DIAGNOSTICS
# =============================================================================

def compute_match(h1, h2, normalize=True):
    """
    Compute the match (inner product) between two waveforms.

    The match is defined as:
        match = <h1|h2> / sqrt(<h1|h1> * <h2|h2>)

    where <a|b> = sum(a * b) for real waveforms.

    For a proper GW match you'd weight by PSD, but for comparing
    GP predictions to training data this simple version is sufficient.

    Parameters
    ----------
    h1, h2 : array-like
        Waveform strain arrays (must be same length)
    normalize : bool
        If True, normalize to get match in [0, 1]

    Returns
    -------
    float
        Match value (1.0 = perfect match)
    """
    h1 = np.asarray(h1)
    h2 = np.asarray(h2)

    inner_12 = np.sum(h1 * h2)

    if normalize:
        inner_11 = np.sum(h1 * h1)
        inner_22 = np.sum(h2 * h2)
        if inner_11 == 0 or inner_22 == 0:
            return 0.0
        return inner_12 / np.sqrt(inner_11 * inner_22)
    else:
        return inner_12


def compute_match_optimized_phase(h1, h2):
    """
    Compute match optimized over phase.

    For complex waveforms or when phase alignment is unknown,
    this computes: max_phi |<h1|h2 * exp(i*phi)>|

    For real waveforms, we approximate this by computing both
    the match and the match with the Hilbert transform.

    Parameters
    ----------
    h1, h2 : array-like
        Waveform strain arrays

    Returns
    -------
    float
        Phase-optimized match value
    """
    from scipy.signal import hilbert

    h1 = np.asarray(h1)
    h2 = np.asarray(h2)

    # Normalize
    norm1 = np.sqrt(np.sum(h1 * h1))
    norm2 = np.sqrt(np.sum(h2 * h2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    h1_norm = h1 / norm1
    h2_norm = h2 / norm2

    # Match with original
    match_real = np.sum(h1_norm * h2_norm)

    # Match with Hilbert transform (90 degree phase shift)
    h2_hilbert = np.imag(hilbert(h2_norm))
    match_imag = np.sum(h1_norm * h2_hilbert)

    # Phase-optimized match
    return np.sqrt(match_real**2 + match_imag**2)


def generate_reference_waveform(
    mass_ratio,
    times,
    total_mass=20.0,
    distance=100.0,
    approximant_name="IMRPhenomPv2",
    polarization="plus"
):
    """
    Generate a reference waveform from a LAL approximant.

    Parameters
    ----------
    mass_ratio : float
        Mass ratio q = m2/m1 (<=1)
    times : array-like
        Time array (in geometric units, relative to merger at t=0)
    total_mass : float
        Total mass in solar masses
    distance : float
        Luminosity distance in Mpc
    approximant_name : str
        Name of LAL approximant
    polarization : str
        'plus' or 'cross'

    Returns
    -------
    array
        Waveform strain at the given times
    """
    from heron.models.lalsimulation import IMRPhenomPv2, SEOBNRv3
    import astropy.units as u

    # Get approximant
    approximants = {
        "IMRPhenomPv2": IMRPhenomPv2,
        "SEOBNRv3": SEOBNRv3,
    }

    if approximant_name not in approximants:
        raise ValueError(f"Unknown approximant: {approximant_name}")

    approx = approximants[approximant_name]()

    # Convert mass ratio to component masses
    # q = m2/m1 where m1 >= m2, so q <= 1
    m1 = total_mass / (1 + mass_ratio)
    m2 = total_mass - m1

    # Convert geometric time to seconds
    # t_geometric = t_seconds * c^3 / (G * M_total)
    # t_seconds = t_geometric * G * M_total / c^3
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    c = 299792458.0  # m/s
    M_sun = 1.98847e30  # kg
    M_total_kg = total_mass * M_sun
    time_conversion = G * M_total_kg / c**3

    times_seconds = np.asarray(times) * time_conversion

    # Generate waveform
    params = {
        "m1": m1 * u.solMass,
        "m2": m2 * u.solMass,
        "distance": distance * u.Mpc,
        "inclination": 0.0,
        "delta T": 1.0 / (4096.0 * u.Hertz),
        "f_min": 20.0 * u.Hertz,
        "f_ref": 20.0 * u.Hertz,
    }

    try:
        waveform_dict = approx.time_domain(params, times=times_seconds)
        return np.asarray(waveform_dict[polarization].data)
    except Exception as e:
        logger.warning(f"Failed to generate reference waveform: {e}")
        return np.zeros_like(times)


def diagnose_match_vs_approximant(
    model,
    train_x,
    polarization,
    output_dir,
    warp_scale=2,
    total_mass=20.0,
    distance=100.0,
    approximant_name="IMRPhenomPv2",
    n_mass_ratios=10
):
    """
    Compute and visualize match between GP predictions and reference approximant.

    Parameters
    ----------
    model : gpytorch model
        Trained GP model
    train_x : torch.Tensor
        Training input data
    polarization : str
        'plus' or 'cross'
    output_dir : str
        Directory for output plots
    warp_scale : float
        Time warping factor
    total_mass : float
        Total mass in solar masses
    distance : float
        Luminosity distance in Mpc
    approximant_name : str
        Reference approximant name
    n_mass_ratios : int
        Number of mass ratios to evaluate
    """
    logger.info(f"Computing match against {approximant_name}")

    train_x_np = train_x.cpu().numpy()

    # Get range of mass ratios and times from training data
    unique_q = np.unique(train_x_np[:, 0])
    time_min = train_x_np[:, 1].min()
    time_max = train_x_np[:, 1].max()

    # Sample mass ratios
    if len(unique_q) > n_mass_ratios:
        q_sample = unique_q[::len(unique_q)//n_mass_ratios][:n_mass_ratios]
    else:
        q_sample = unique_q

    # Create time array for comparison
    n_times = 500
    times = np.linspace(time_min, time_max, n_times)

    # Store results
    matches = []
    matches_phase_opt = []
    mass_ratios_tested = []

    for q in q_sample:
        try:
            # Generate GP prediction
            times_tensor = torch.tensor(times, dtype=torch.float32).to(device)
            test_x = torch.stack([
                torch.ones_like(times_tensor) * q,
                times_tensor
            ], dim=1)

            # Apply warping
            test_x_warped = test_x.clone()
            test_x_warped[test_x[:, 1] < 0, 1] /= warp_scale

            model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                posterior = model.likelihood(model(test_x_warped))
                gp_mean = posterior.mean.cpu().numpy()

            # Denormalize
            if hasattr(model, 'y_mean') and model.y_mean is not None:
                gp_mean = gp_mean * model.y_std + model.y_mean

            # Generate reference waveform
            ref_waveform = generate_reference_waveform(
                mass_ratio=q,
                times=times,
                total_mass=total_mass,
                distance=distance,
                approximant_name=approximant_name,
                polarization=polarization
            )

            # Skip if reference waveform generation failed
            if np.all(ref_waveform == 0):
                continue

            # Compute matches
            match = compute_match(gp_mean, ref_waveform)
            match_opt = compute_match_optimized_phase(gp_mean, ref_waveform)

            matches.append(match)
            matches_phase_opt.append(match_opt)
            mass_ratios_tested.append(q)

        except Exception as e:
            logger.warning(f"Failed to compute match for q={q}: {e}")
            continue

    if len(matches) == 0:
        logger.error("No matches computed successfully")
        return {}

    matches = np.array(matches)
    matches_phase_opt = np.array(matches_phase_opt)
    mass_ratios_tested = np.array(mass_ratios_tested)

    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Match vs mass ratio
    ax = axes[0, 0]
    ax.plot(mass_ratios_tested, matches, 'b-o', label='Direct match', markersize=6)
    ax.plot(mass_ratios_tested, matches_phase_opt, 'r-s', label='Phase-optimized', markersize=6)
    ax.axhline(0.97, color='g', linestyle='--', alpha=0.7, label='97% threshold')
    ax.axhline(0.99, color='orange', linestyle=':', alpha=0.7, label='99% threshold')
    ax.set_xlabel('Mass Ratio (q)')
    ax.set_ylabel('Match')
    ax.set_title(f'Match vs Mass Ratio\n(GP vs {approximant_name})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([min(0, matches.min() - 0.1), 1.05])

    # 2. Mismatch (1 - match) on log scale
    ax = axes[0, 1]
    mismatch = 1 - matches_phase_opt
    mismatch[mismatch <= 0] = 1e-10  # Avoid log(0)
    ax.semilogy(mass_ratios_tested, mismatch, 'r-o', markersize=6)
    ax.axhline(0.03, color='g', linestyle='--', alpha=0.7, label='3% mismatch')
    ax.axhline(0.01, color='orange', linestyle=':', alpha=0.7, label='1% mismatch')
    ax.set_xlabel('Mass Ratio (q)')
    ax.set_ylabel('Mismatch (1 - match)')
    ax.set_title('Mismatch (log scale)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # 3. Waveform comparison at best and worst mass ratios
    ax = axes[1, 0]
    best_idx = np.argmax(matches_phase_opt)
    worst_idx = np.argmin(matches_phase_opt)

    for idx, label, color in [(best_idx, 'Best', 'green'), (worst_idx, 'Worst', 'red')]:
        q = mass_ratios_tested[idx]

        # GP prediction
        times_tensor = torch.tensor(times, dtype=torch.float32).to(device)
        test_x = torch.stack([torch.ones_like(times_tensor) * q, times_tensor], dim=1)
        test_x_warped = test_x.clone()
        test_x_warped[test_x[:, 1] < 0, 1] /= warp_scale

        model.eval()
        with torch.no_grad():
            gp_mean = model.likelihood(model(test_x_warped)).mean.cpu().numpy()
        if hasattr(model, 'y_mean') and model.y_mean is not None:
            gp_mean = gp_mean * model.y_std + model.y_mean

        # Reference
        ref = generate_reference_waveform(q, times, total_mass, distance, approximant_name, polarization)

        # Normalize for comparison
        gp_norm = gp_mean / (np.max(np.abs(gp_mean)) + 1e-30)
        ref_norm = ref / (np.max(np.abs(ref)) + 1e-30)

        ax.plot(times, ref_norm, color=color, linestyle='-', alpha=0.7,
                label=f'{label}: q={q:.2f}, ref')
        ax.plot(times, gp_norm, color=color, linestyle='--', alpha=0.7,
                label=f'{label}: q={q:.2f}, GP')

    ax.set_xlabel('Time (geometric units)')
    ax.set_ylabel('Normalized Strain')
    ax.set_title('Waveform Comparison (best/worst)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    summary = f"""MATCH DIAGNOSTICS: {polarization}
{'='*50}

REFERENCE: {approximant_name}
Total mass: {total_mass} M_sun
Distance:   {distance} Mpc

MATCH STATISTICS (phase-optimized):
  Mean match:          {matches_phase_opt.mean():.4f}
  Min match:           {matches_phase_opt.min():.4f} (q={mass_ratios_tested[worst_idx]:.3f})
  Max match:           {matches_phase_opt.max():.4f} (q={mass_ratios_tested[best_idx]:.3f})
  Std match:           {matches_phase_opt.std():.4f}

MATCH THRESHOLDS:
  Match > 0.99:        {100*np.mean(matches_phase_opt > 0.99):.1f}%
  Match > 0.97:        {100*np.mean(matches_phase_opt > 0.97):.1f}%
  Match > 0.90:        {100*np.mean(matches_phase_opt > 0.90):.1f}%

INTERPRETATION:
"""
    issues = []
    if matches_phase_opt.mean() < 0.9:
        issues.append("- CRITICAL: Mean match < 90%")
    elif matches_phase_opt.mean() < 0.97:
        issues.append("- WARNING: Mean match < 97%")
    if matches_phase_opt.min() < 0.8:
        issues.append("- Some mass ratios have very poor match")
    if matches_phase_opt.std() > 0.1:
        issues.append("- High variance in match across q")
    if not issues:
        issues.append("- Match looks good across parameter space")
    summary += '\n'.join(issues)

    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.suptitle(f'Match Diagnostics: GP vs {approximant_name} ({polarization})', fontsize=14)
    plt.tight_layout()

    output_file = os.path.join(output_dir, f'diagnostics_match_{polarization}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Match diagnostics saved to {output_file}")

    # Create 2D match map if we have enough data
    if len(unique_q) >= 5:
        _plot_match_heatmap(
            model, train_x, polarization, output_dir, warp_scale,
            total_mass, distance, approximant_name
        )

    return {
        'mean_match': matches_phase_opt.mean(),
        'min_match': matches_phase_opt.min(),
        'max_match': matches_phase_opt.max(),
        'mass_ratios': mass_ratios_tested,
        'matches': matches_phase_opt
    }


def _plot_match_heatmap(
    model, train_x, polarization, output_dir, warp_scale,
    total_mass, distance, approximant_name
):
    """
    Create a 2D heatmap showing match quality across the parameter space.
    """
    logger.info("Creating match heatmap")

    train_x_np = train_x.cpu().numpy()

    # Define grid
    q_min, q_max = train_x_np[:, 0].min(), train_x_np[:, 0].max()
    t_min, t_max = train_x_np[:, 1].min(), train_x_np[:, 1].max()

    n_q = 15
    n_t = 50
    q_grid = np.linspace(q_min, q_max, n_q)
    t_grid = np.linspace(t_min, t_max, n_t)

    # Compute match at each mass ratio
    match_values = np.zeros(n_q)
    local_errors = np.zeros((n_q, n_t))

    for i, q in enumerate(q_grid):
        try:
            # GP prediction
            times_tensor = torch.tensor(t_grid, dtype=torch.float32).to(device)
            test_x = torch.stack([torch.ones_like(times_tensor) * q, times_tensor], dim=1)
            test_x_warped = test_x.clone()
            test_x_warped[test_x[:, 1] < 0, 1] /= warp_scale

            model.eval()
            with torch.no_grad():
                gp_mean = model.likelihood(model(test_x_warped)).mean.cpu().numpy()
            if hasattr(model, 'y_mean') and model.y_mean is not None:
                gp_mean = gp_mean * model.y_std + model.y_mean

            # Reference
            ref = generate_reference_waveform(
                q, t_grid, total_mass, distance, approximant_name, polarization
            )

            if not np.all(ref == 0):
                match_values[i] = compute_match_optimized_phase(gp_mean, ref)
                # Local error at each time point
                local_errors[i, :] = np.abs(gp_mean - ref) / (np.max(np.abs(ref)) + 1e-30)

        except Exception as e:
            logger.warning(f"Failed for q={q}: {e}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Match vs mass ratio (cleaner view)
    ax = axes[0]
    ax.fill_between(q_grid, 0.97, 1.0, alpha=0.2, color='green', label='Good (>97%)')
    ax.fill_between(q_grid, 0.90, 0.97, alpha=0.2, color='orange', label='Acceptable (90-97%)')
    ax.fill_between(q_grid, 0, 0.90, alpha=0.2, color='red', label='Poor (<90%)')
    ax.plot(q_grid, match_values, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Mass Ratio (q)')
    ax.set_ylabel('Match')
    ax.set_title(f'Match vs Mass Ratio\n(GP vs {approximant_name})')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # 2. Local error heatmap
    ax = axes[1]
    im = ax.imshow(
        local_errors.T,
        aspect='auto',
        origin='lower',
        extent=[q_min, q_max, t_min, t_max],
        cmap='hot_r',
        vmin=0,
        vmax=np.percentile(local_errors, 95)
    )
    ax.set_xlabel('Mass Ratio (q)')
    ax.set_ylabel('Time (geometric units)')
    ax.set_title('Local Error Map\n(darker = larger error)')
    plt.colorbar(im, ax=ax, label='Relative Error')

    plt.suptitle(f'Match Analysis: {polarization}', fontsize=14)
    plt.tight_layout()

    output_file = os.path.join(output_dir, f'diagnostics_match_map_{polarization}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Match heatmap saved to {output_file}")
