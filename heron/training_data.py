"""
Code to generate training data for GPR waveform models.

This module provides functionality to:
1. Generate waveform manifolds from approximants or NR catalogs
2. Store training data in HDF5 format
3. Create diagnostic plots for validation
"""

import os
import logging
import click
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import astropy.units as u

from heron.training.makedata import make_manifold, make_optimal_manifold
from heron.training.data import DataWrapper
from heron.models.lalsimulation import IMRPhenomPv2, SEOBNRv3
from heron.utils import load_yaml

logger = logging.getLogger("heron.training_data")

# Map of known approximant names to classes
KNOWN_APPROXIMANTS = {
    "IMRPhenomPv2": IMRPhenomPv2,
    "SEOBNRv3": SEOBNRv3,
}


def generate_training_data(
    waveform_source,
    parameter_space,
    polarizations=["plus", "cross"],
    output_file="training_data.h5",
    group_name="training",
    use_peak_sampling=False,
    plots_dir="plots"
):
    """
    Generate training data from waveforms.

    Parameters
    ----------
    waveform_source : dict
        Dictionary specifying waveform source:
        - type: 'approximant' or 'nr_catalog'
        - approximant: name of LAL approximant (if type='approximant')
        - catalog_path: path to NR catalog (if type='nr_catalog')
    parameter_space : dict
        Dictionary with 'fixed' and 'varied' parameters
    polarizations : list
        List of polarizations to include ('plus', 'cross')
    output_file : str
        Path to output HDF5 file
    group_name : str
        Name of the training data group in HDF5
    use_peak_sampling : bool
        Whether to use peak-based sampling (make_optimal_manifold)
    plots_dir : str
        Directory for diagnostic plots

    Returns
    -------
    DataWrapper
        The data wrapper with training data stored
    str
        Path to the plots directory
    """
    logger.info("Starting training data generation")

    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)

    # Get waveform approximant
    if waveform_source['type'] == 'approximant':
        approximant_name = waveform_source['approximant']
        if approximant_name not in KNOWN_APPROXIMANTS:
            raise ValueError(f"Unknown approximant: {approximant_name}. "
                           f"Known approximants: {list(KNOWN_APPROXIMANTS.keys())}")
        approximant = KNOWN_APPROXIMANTS[approximant_name]
        logger.info(f"Using approximant: {approximant_name}")
    elif waveform_source['type'] == 'nr_catalog':
        raise NotImplementedError("NR catalog support coming soon")
    else:
        raise ValueError(f"Unknown waveform source type: {waveform_source['type']}")

    # Build parameter dictionaries for make_manifold
    fixed_params = {}
    varied_params = {}

    # Process fixed parameters with units
    for param_name, param_value in parameter_space['fixed'].items():
        if param_name == 'total_mass':
            fixed_params['total_mass'] = param_value * u.solMass
        elif param_name == 'sample_rate':
            fixed_params['delta_T'] = 1.0 / (param_value * u.Hertz)
        elif param_name == 'duration':
            # Duration is handled via time array, not directly passed
            pass
        else:
            fixed_params[param_name] = param_value

    # Process varied parameters
    for param_name, param_spec in parameter_space['varied'].items():
        varied_params[param_name] = {
            'lower': param_spec['lower'],
            'upper': param_spec['upper'],
        }
        if 'step' in param_spec:
            varied_params[param_name]['step'] = param_spec['step']
        elif 'samples' in param_spec:
            varied_params[param_name]['samples'] = param_spec['samples']

    logger.info(f"Fixed parameters: {fixed_params}")
    logger.info(f"Varied parameters: {varied_params}")

    # Generate waveform manifold
    if use_peak_sampling:
        logger.info("Using peak-based sampling")
        # make_optimal_manifold returns (manifold_plus, manifold_cross)
        manifold_plus, manifold_cross = make_optimal_manifold(
            approximant=approximant,
            fixed=fixed_params,
            varied=varied_params
        )
        # For peak sampling, we have separate manifolds per polarization
        manifolds = {'plus': manifold_plus, 'cross': manifold_cross}
        n_waveforms = len(manifold_plus.data)
    else:
        logger.info("Using regular grid sampling")
        # make_manifold returns a single manifold with all polarizations
        manifold = make_manifold(
            approximant=approximant,
            fixed=fixed_params,
            varied=varied_params
        )
        # For regular sampling, use the same manifold for all polarizations
        manifolds = {'plus': manifold, 'cross': manifold}
        n_waveforms = len(manifold.data)

    logger.info(f"Generated {n_waveforms} waveforms")

    # Create diagnostic plots (use first manifold for diagnostics)
    logger.info("Creating diagnostic plots")
    diagnostic_manifold = manifolds['plus'] if use_peak_sampling else manifold
    _plot_parameter_space(diagnostic_manifold, plots_dir)
    _plot_waveform_samples(diagnostic_manifold, plots_dir, n_samples=9)
    _plot_manifold_heatmap(diagnostic_manifold, plots_dir)

    # Create or open HDF5 file
    if os.path.exists(output_file):
        logger.info(f"Opening existing training data file: {output_file}")
        data = DataWrapper(output_file)
    else:
        logger.info(f"Creating new training data file: {output_file}")
        data = DataWrapper.create(output_file)

    # Store waveforms in HDF5
    logger.info(f"Storing waveforms in HDF5 group: {group_name}")

    for pol in polarizations:
        logger.info(f"Processing {pol} polarization")
        pol_key = pol  # 'plus' or 'cross'

        # Get the appropriate manifold for this polarization
        pol_manifold = manifolds[pol_key]

        for idx, wf_dict in enumerate(pol_manifold.data):
            if idx % 10 == 0:
                logger.info(f"Storing waveform {idx+1}/{n_waveforms}")

            # Get the waveform for this polarization
            if pol_key not in wf_dict.waveforms:
                logger.warning(f"Polarization {pol_key} not found in waveform {idx}")
                continue

            waveform = wf_dict.waveforms[pol_key]

            # Extract parameters from the WaveformDict
            # The manifold stores parameters in manifold.locations[idx]
            locations = pol_manifold.locations[idx]

            # Store in HDF5
            data.add_waveform(
                group=group_name,
                polarisation=pol_key[0],  # 'p' or 'c'
                reference_mass=fixed_params.get('total_mass', 60*u.solMass).value,
                source=approximant_name,
                locations=locations,
                times=waveform.times.value,
                data=waveform.value
            )

    logger.info(f"Training data generation complete: {output_file}")

    return data, plots_dir


def _plot_parameter_space(manifold, plots_dir):
    """Plot parameter space coverage."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract parameter values from manifold.locations
    # For now, assume we're varying mass_ratio
    mass_ratios = []
    for params in manifold.locations:
        if 'mass_ratio' in params:
            mass_ratios.append(params['mass_ratio'])

    if mass_ratios:
        ax.scatter(mass_ratios, np.ones_like(mass_ratios), c='blue', alpha=0.6, s=50)
        ax.set_xlabel('Mass Ratio', fontsize=12)
        ax.set_ylabel('', fontsize=12)
        ax.set_ylim(0.5, 1.5)
        ax.set_yticks([])
        ax.set_title('Training Data Parameter Coverage', fontsize=14)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "parameter_space.png"), dpi=150)
    plt.close()
    logger.info("Created parameter_space.png")


def _plot_waveform_samples(manifold, plots_dir, n_samples=9):
    """Plot a grid of sample waveforms."""
    n_waveforms = len(manifold.data)

    # Select evenly spaced samples
    if n_waveforms < n_samples:
        n_samples = n_waveforms

    indices = np.linspace(0, n_waveforms-1, n_samples, dtype=int)

    # Create subplot grid
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_samples > 1 else [axes]

    for idx, ax in enumerate(axes):
        if idx < len(indices):
            wf_dict = manifold.data[indices[idx]]
            params = manifold.locations[indices[idx]]

            # Plot plus polarization
            if 'plus' in wf_dict.waveforms:
                wf = wf_dict.waveforms['plus']
                ax.plot(wf.times.value, wf.value, 'b-', linewidth=1)

                # Add title with parameters
                title = f"Waveform {indices[idx]}"
                if 'mass_ratio' in params:
                    title += f"\nq={params['mass_ratio']:.2f}"
                ax.set_title(title, fontsize=10)
                ax.set_xlabel('Time (s)', fontsize=9)
                ax.set_ylabel('Strain', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        else:
            ax.axis('off')

    plt.suptitle('Sample Waveforms from Manifold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "waveform_samples.png"), dpi=150)
    plt.close()
    logger.info("Created waveform_samples.png")


def _plot_manifold_heatmap(manifold, plots_dir):
    """Plot 2D heatmap of waveform manifold."""
    try:
        # Get manifold as array
        manifold_array = manifold.array()

        fig, ax = plt.subplots(figsize=(12, 6))

        im = ax.imshow(manifold_array.T, aspect='auto', cmap='RdBu_r',
                      origin='lower', interpolation='nearest')

        ax.set_xlabel('Waveform Index', fontsize=12)
        ax.set_ylabel('Time Sample', fontsize=12)
        ax.set_title('Waveform Manifold Heatmap', fontsize=14)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Strain', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "manifold_heatmap.png"), dpi=150)
        plt.close()
        logger.info("Created manifold_heatmap.png")
    except Exception as e:
        logger.warning(f"Could not create manifold heatmap: {e}")


@click.command(name='training-data')
@click.option("--config", required=True, help="Path to training data configuration YAML file")
@click.option("--output", default=None, help="Override output HDF5 file path")
def training_data(config, output):
    """
    Generate training data for GPR waveform models.

    This command generates waveform manifolds from approximants or NR catalogs
    and stores them in HDF5 format for training GPR models.
    """
    click.echo("Generating training data for GPR models")

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
    waveform_source = settings.get('waveform_source')
    parameter_space = settings.get('parameter_space')
    polarizations = settings.get('polarizations', ['plus', 'cross'])

    output_config = settings.get('output', {})
    output_file = output or output_config.get('file', 'training_data.h5')
    group_name = output_config.get('group_name', 'training')

    optimization = settings.get('optimization', {})
    use_peak_sampling = optimization.get('use_peak_sampling', False)

    # Determine plots directory
    if 'pages directory' in settings:
        plots_dir = os.path.join(settings['pages directory'], 'plots')
    else:
        plots_dir = 'plots'

    # Generate training data
    try:
        data, plots_path = generate_training_data(
            waveform_source=waveform_source,
            parameter_space=parameter_space,
            polarizations=polarizations,
            output_file=output_file,
            group_name=group_name,
            use_peak_sampling=use_peak_sampling,
            plots_dir=plots_dir
        )

        click.echo(f"✓ Training data generated successfully: {output_file}")
        click.echo(f"✓ Diagnostic plots saved to: {plots_path}")

    except Exception as e:
        logger.exception(e)
        click.echo(f"✗ Error generating training data: {e}", err=True)
        raise
