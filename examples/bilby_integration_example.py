#!/usr/bin/env python3
"""
Example: Using Heron Waveforms with Bilby
==========================================

This example demonstrates how to use heron's Gaussian process-based
waveform models with uncertainty in a bilby parameter estimation analysis.

The key features demonstrated are:
1. Creating a HeronWaveformGenerator from a heron waveform model
2. Setting up the HeronGravitationalWaveTransient likelihood with uncertainty
3. Running a parameter estimation with model uncertainty incorporated

Requirements
------------
- bilby
- heron-model
- numpy

"""

import numpy as np
import bilby
from bilby.gw.detector import get_empty_interferometer

# Import heron modules
from heron.models.testing import SineGaussianWaveform  # Or use a real heron model
from heron.bilby import HeronWaveformGenerator, HeronGravitationalWaveTransient


def main():
    """Run a simple parameter estimation with heron waveforms."""
    
    # Set up analysis parameters
    duration = 4.0
    sampling_frequency = 2048
    
    # Define injection parameters
    injection_parameters = {
        'width': 0.02,
        'ra': 1.375,
        'dec': -1.2108,
        'geocent_time': 1.0,
        'psi': 0.659,
        'phase': 1.3,
        'theta_jn': 0.4,
    }
    
    # Create a heron waveform model
    # In a real analysis, you would load a trained heron model here
    heron_model = SineGaussianWaveform()
    
    # Create the waveform generator with heron
    waveform_generator = HeronWaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        heron_waveform=heron_model,
        start_time=injection_parameters['geocent_time'] - duration/2
    )
    
    # Set up interferometers
    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
    
    # Inject a signal
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters['geocent_time'] - duration/2
    )
    
    # Generate the injection signal
    # Note: This is simplified; in practice you'd need to handle
    # the conversion between heron and bilby waveform formats
    
    # Set up priors
    priors = bilby.core.prior.PriorDict()
    priors['width'] = bilby.core.prior.Uniform(0.01, 0.1, name='width')
    priors['ra'] = bilby.core.prior.Uniform(0, 2*np.pi, name='ra', boundary='periodic')
    priors['dec'] = bilby.core.prior.Cosine(name='dec')
    priors['geocent_time'] = bilby.core.prior.Uniform(
        injection_parameters['geocent_time'] - 0.1,
        injection_parameters['geocent_time'] + 0.1,
        name='geocent_time'
    )
    priors['psi'] = bilby.core.prior.Uniform(0, np.pi, name='psi', boundary='periodic')
    priors['phase'] = bilby.core.prior.Uniform(0, 2*np.pi, name='phase', boundary='periodic')
    priors['theta_jn'] = bilby.core.prior.Sine(name='theta_jn')
    
    # Create the likelihood with model uncertainty
    likelihood = HeronGravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=waveform_generator,
        priors=priors,
        include_model_uncertainty=True  # This is the key difference!
    )
    
    # Run the parameter estimation
    # Using a simple sampler for demonstration
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler='dynesty',
        nlive=500,
        outdir='outdir',
        label='heron_with_uncertainty',
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    )
    
    # Make a corner plot
    result.plot_corner()
    
    print("Analysis complete!")
    print(f"Results saved to: {result.outdir}")


def compare_with_without_uncertainty():
    """
    Example comparing results with and without model uncertainty.
    
    This demonstrates the impact of including waveform model uncertainty
    in the analysis.
    """
    
    duration = 4.0
    sampling_frequency = 2048
    
    # Set up the same analysis twice
    heron_model = SineGaussianWaveform()
    
    # Waveform generator
    wfg = HeronWaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        heron_waveform=heron_model,
    )
    
    # Interferometers (same data for both)
    ifos = bilby.gw.detector.InterferometerList(['H1'])
    ifos.set_strain_data_from_zero_noise(
        sampling_frequency=sampling_frequency,
        duration=duration,
    )
    
    # Simple priors
    priors = bilby.core.prior.PriorDict()
    priors['width'] = bilby.core.prior.Uniform(0.01, 0.1, name='width')
    
    # Likelihood WITH uncertainty
    likelihood_with_unc = HeronGravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=wfg,
        include_model_uncertainty=True
    )
    
    # Likelihood WITHOUT uncertainty
    likelihood_no_unc = HeronGravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=wfg,
        include_model_uncertainty=False
    )
    
    # Evaluate at a test point
    test_params = {'width': 0.02}
    
    log_l_with = likelihood_with_unc.log_likelihood_ratio()
    log_l_without = likelihood_no_unc.log_likelihood_ratio()
    
    print(f"Log-likelihood with uncertainty: {log_l_with}")
    print(f"Log-likelihood without uncertainty: {log_l_without}")
    print(f"Difference: {log_l_with - log_l_without}")
    

if __name__ == "__main__":
    # Run the main example
    # Note: This will fail without a proper installation of bilby
    # Uncomment to run:
    # main()
    
    # Or run the comparison
    # compare_with_without_uncertainty()
    
    print("Example script loaded. Uncomment main() or compare_with_without_uncertainty() to run.")
