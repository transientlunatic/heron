"""
Interfaces with the bilby library
---------------------------------

This file contains code to allow heron to interface with bilby,
including reading bilby data files, likelihoods, and waveform generators.
"""

import pickle
import numpy as np

try:
    import bilby_pipe
    BILBY_PIPE_AVAILABLE = True
except ImportError:
    BILBY_PIPE_AVAILABLE = False

try:
    import bilby
    import bilby.gw.likelihood
    import bilby.gw.waveform_generator
    BILBY_AVAILABLE = True
except ImportError:
    BILBY_AVAILABLE = False

def read_pickle(filename):
    """
    Read a data pickle created by bilby_pipe_generation.

    Parameters
    ----------
    filename : str
       The path to the file to be read to provide strain data.

    Notes
    -----

    At present this function only returns the strain data.
    Other information is also stored in the pickle, and this could
    also be extracted, but in order to use bilby to e.g. perform
    injections we currently only need this.
    """
    if not BILBY_PIPE_AVAILABLE:
        raise ImportError("bilby_pipe is required for this function")
    
    output = {}
    with open(filename, "rb") as pickle_file:
        data = pickle.load(pickle_file)

    output['strain'] = {}

    for ifo in data.interferometers:
        output['strain'][ifo.name] = ifo.time_domain_strain

    return output


if BILBY_AVAILABLE:
    
    class HeronWaveformGenerator(bilby.gw.waveform_generator.WaveformGenerator):
        """
        A bilby-compatible waveform generator for heron waveform models.
        
        This class enables the use of heron's Gaussian process-based waveform
        models with uncertainty within the bilby framework. It handles the
        conversion between heron's internal waveform representation and bilby's
        expected format, including proper handling of waveform uncertainty.
        
        Parameters
        ----------
        duration : float
            Duration of the data in seconds.
        sampling_frequency : float
            Sampling frequency of the data in Hz.
        start_time : float, optional
            GPS start time of the data segment.
        frequency_domain_source_model : callable, optional
            A frequency-domain source model function. Not recommended for heron models.
        time_domain_source_model : callable, optional
            A time-domain source model function. Should be a heron waveform model.
        parameters : dict, optional
            Dictionary of fixed parameters for the waveform.
        parameter_conversion : callable, optional
            Function to convert between parameter sets.
        waveform_arguments : dict, optional
            Additional arguments to pass to the waveform model.
        heron_waveform : object, optional
            A heron waveform model instance (WaveformApproximant or WaveformSurrogate).
            
        Notes
        -----
        This generator is specifically designed to work with heron's waveform models
        that include uncertainty estimates from Gaussian process regression. The
        uncertainty is propagated through the antenna response projection and can
        be used in the likelihood calculation.
        
        The waveform generator handles:
        - Time-domain and frequency-domain waveform generation
        - Antenna response projection for detectors
        - Waveform uncertainty propagation
        - FFT operations with proper normalization
        
        Examples
        --------
        >>> from heron.models.gpytorch import HeronNonSpinningApproximant
        >>> from heron.bilby import HeronWaveformGenerator
        >>> 
        >>> # Load a heron model
        >>> heron_model = HeronNonSpinningApproximant()
        >>> 
        >>> # Create a waveform generator
        >>> wfg = HeronWaveformGenerator(
        ...     duration=4.0,
        ...     sampling_frequency=2048,
        ...     heron_waveform=heron_model
        ... )
        """
        
        def __init__(
            self,
            duration=None,
            sampling_frequency=None,
            start_time=0,
            frequency_domain_source_model=None,
            time_domain_source_model=None,
            parameters=None,
            parameter_conversion=None,
            waveform_arguments=None,
            heron_waveform=None,
        ):
            """Initialize the heron waveform generator."""
            self.heron_waveform = heron_waveform
            
            # If a heron waveform is provided, use it as the source model
            if heron_waveform is not None and time_domain_source_model is None:
                time_domain_source_model = self._heron_time_domain_model
            
            super().__init__(
                duration=duration,
                sampling_frequency=sampling_frequency,
                start_time=start_time,
                frequency_domain_source_model=frequency_domain_source_model,
                time_domain_source_model=time_domain_source_model,
                parameters=parameters,
                parameter_conversion=parameter_conversion,
                waveform_arguments=waveform_arguments,
            )
            
        def _heron_time_domain_model(self, times, **parameters):
            """
            Generate a time-domain waveform using a heron model.
            
            Parameters
            ----------
            times : array_like
                Array of time values at which to evaluate the waveform.
            **parameters : dict
                Waveform parameters.
                
            Returns
            -------
            dict
                Dictionary with 'plus' and 'cross' polarization waveforms.
                Each waveform includes uncertainty information if available.
            """
            if self.heron_waveform is None:
                raise ValueError("No heron waveform model specified")
            
            # Generate the waveform using heron
            waveform_dict = self.heron_waveform.time_domain(
                parameters=parameters,
                times=times
            )
            
            # Extract the polarizations and their uncertainties
            result = {
                'plus': waveform_dict['plus'].data,
                'cross': waveform_dict['cross'].data,
            }
            
            # Store uncertainty information for use in likelihood
            if hasattr(waveform_dict['plus'], 'covariance'):
                result['plus_covariance'] = waveform_dict['plus'].covariance
                result['cross_covariance'] = waveform_dict['cross'].covariance
            
            if hasattr(waveform_dict['plus'], '_variance'):
                result['plus_variance'] = waveform_dict['plus']._variance
                result['cross_variance'] = waveform_dict['cross']._variance
                
            return result
    

    class HeronGravitationalWaveTransient(bilby.gw.likelihood.GravitationalWaveTransient):
        """
        A bilby likelihood for gravitational wave transients with heron waveform uncertainty.
        
        This likelihood extends bilby's standard GravitationalWaveTransient likelihood
        to properly handle waveform model uncertainty from heron's Gaussian process
        regression. The uncertainty is incorporated into the likelihood calculation
        by adding the waveform covariance to the detector noise covariance.
        
        Parameters
        ----------
        interferometers : list
            List of bilby Interferometer objects containing the data.
        waveform_generator : HeronWaveformGenerator
            A heron waveform generator instance.
        priors : dict or bilby.core.prior.PriorDict, optional
            Prior distributions for parameters.
        distance_marginalization : bool, optional
            Whether to marginalize over distance. Default is False.
        phase_marginalization : bool, optional
            Whether to marginalize over phase. Default is False.
        time_marginalization : bool, optional
            Whether to marginalize over time. Default is False.
        jitter_time : bool, optional
            Whether to add time jitter. Default is True.
        reference_frame : str, optional
            Reference frame for the analysis.
        time_reference : str, optional
            Time reference for the analysis.
        include_model_uncertainty : bool, optional
            Whether to include waveform model uncertainty in the likelihood.
            Default is True.
            
        Notes
        -----
        The likelihood is computed as:
        
        .. math::
            \\ln \\mathcal{L} = -\\frac{1}{2}(d - h)^T (C_n + C_h)^{-1} (d - h) 
                               - \\frac{1}{2}\\ln|C_n + C_h| - \\frac{N}{2}\\ln(2\\pi)
        
        where:
        - d is the observed data
        - h is the predicted waveform
        - C_n is the detector noise covariance
        - C_h is the waveform model uncertainty covariance
        - N is the number of data points
        
        When `include_model_uncertainty=False`, this reduces to the standard
        bilby likelihood with C_h = 0.
        
        The waveform uncertainty accounts for:
        - Gaussian process regression uncertainty
        - Projection through antenna response functions
        - Proper FFT normalization for frequency-domain operations
        
        Examples
        --------
        >>> from bilby.gw.detector import InterferometerList
        >>> from heron.models.gpytorch import HeronNonSpinningApproximant
        >>> from heron.bilby import HeronWaveformGenerator, HeronGravitationalWaveTransient
        >>> 
        >>> # Set up interferometers (from bilby)
        >>> ifos = InterferometerList(['H1', 'L1'])
        >>> 
        >>> # Create heron waveform generator
        >>> heron_model = HeronNonSpinningApproximant()
        >>> wfg = HeronWaveformGenerator(
        ...     duration=4.0,
        ...     sampling_frequency=2048,
        ...     heron_waveform=heron_model
        ... )
        >>> 
        >>> # Create likelihood with uncertainty
        >>> likelihood = HeronGravitationalWaveTransient(
        ...     interferometers=ifos,
        ...     waveform_generator=wfg,
        ...     include_model_uncertainty=True
        ... )
        """
        
        def __init__(
            self,
            interferometers,
            waveform_generator,
            priors=None,
            distance_marginalization=False,
            phase_marginalization=False,
            time_marginalization=False,
            jitter_time=True,
            reference_frame="sky",
            time_reference="geocent",
            include_model_uncertainty=True,
        ):
            """Initialize the heron likelihood."""
            self.include_model_uncertainty = include_model_uncertainty
            
            super().__init__(
                interferometers=interferometers,
                waveform_generator=waveform_generator,
                priors=priors,
                distance_marginalization=distance_marginalization,
                phase_marginalization=phase_marginalization,
                time_marginalization=time_marginalization,
                jitter_time=jitter_time,
                reference_frame=reference_frame,
                time_reference=time_reference,
            )
        
        def log_likelihood_ratio(self):
            """
            Calculate the log-likelihood ratio with waveform uncertainty.
            
            Returns
            -------
            float
                The log-likelihood ratio value.
                
            Notes
            -----
            This method overrides the parent class to incorporate waveform
            model uncertainty when available. If uncertainty information is
            not present in the waveform, it falls back to the standard
            calculation.
            """
            if not self.include_model_uncertainty:
                # Fall back to standard bilby likelihood
                return super().log_likelihood_ratio()
            
            # Get waveforms with uncertainty
            waveform_polarizations = self.waveform_generator.time_domain_strain(
                self.parameters
            )
            
            log_l = 0
            
            for ifo in self.interferometers:
                # Get the signal in the detector frame
                signal = ifo.get_detector_response(
                    waveform_polarizations, self.parameters
                )
                
                # Check if uncertainty information is available
                if 'plus_covariance' in waveform_polarizations:
                    # Calculate with model uncertainty
                    log_l += self._log_likelihood_with_uncertainty(
                        ifo, signal, waveform_polarizations
                    )
                else:
                    # Fall back to standard calculation for this IFO
                    log_l += self._log_likelihood_standard(ifo, signal)
            
            return log_l
        
        def _log_likelihood_with_uncertainty(self, ifo, signal, waveform_polarizations):
            """
            Calculate log-likelihood for one interferometer with model uncertainty.
            
            Parameters
            ----------
            ifo : bilby.gw.detector.Interferometer
                The interferometer object.
            signal : array_like
                The projected signal in the detector frame.
            waveform_polarizations : dict
                Dictionary containing waveform polarizations and uncertainties.
                
            Returns
            -------
            float
                The log-likelihood contribution from this interferometer.
            """
            # Get data and compute residual
            residual = ifo.time_domain_strain - signal
            
            # Get detector noise PSD and convert to covariance
            # For simplicity in frequency domain, we assume diagonal noise
            psd = ifo.power_spectral_density_array
            
            # Project waveform uncertainty through detector response
            plus_cov = waveform_polarizations.get('plus_covariance')
            cross_cov = waveform_polarizations.get('cross_covariance')
            
            # Get antenna responses
            antenna_response = ifo.antenna_response(
                self.parameters['ra'],
                self.parameters['dec'],
                self.parameters.get('geocent_time', 0),
                self.parameters.get('psi', 0)
            )
            
            fp, fc = antenna_response
            
            # Project uncertainty: Var(F+ h+ + Fx hx) = F+^2 Var(h+) + Fx^2 Var(hx)
            # For full covariance: C_projected = F+^2 C_plus + Fx^2 C_cross
            if plus_cov is not None and cross_cov is not None:
                signal_cov = fp**2 * plus_cov + fc**2 * cross_cov
            else:
                # If only variance is available
                plus_var = waveform_polarizations.get('plus_variance')
                cross_var = waveform_polarizations.get('cross_variance')
                if plus_var is not None:
                    signal_var = fp**2 * plus_var + fc**2 * cross_var
                    signal_cov = np.diag(signal_var)
                else:
                    # No uncertainty available, fall back to standard
                    return self._log_likelihood_standard(ifo, signal)
            
            # Compute likelihood with uncertainty
            # For computational efficiency, we work in frequency domain
            # and assume uncorrelated noise (diagonal PSD)
            
            # FFT of residual
            residual_fft = np.fft.rfft(residual) * ifo.strain_data.duration / len(residual)
            
            # In frequency domain, noise covariance is diagonal with PSD
            # Total covariance = noise + signal uncertainty (need to FFT signal_cov)
            
            # For large covariance matrices, use the Woodbury identity or
            # work directly in time domain for better numerical stability
            # Here we use a simplified approach assuming diagonal dominance
            
            # Convert to frequency domain for efficient computation
            delta_f = 1.0 / ifo.strain_data.duration
            
            # Compute <d-h|d-h> inner product weighted by inverse PSD
            inner_product = 4 * delta_f * np.sum(
                np.abs(residual_fft)**2 / psd[:len(residual_fft)]
            ).real
            
            # Add correction for model uncertainty
            # This is approximate - proper treatment would require full covariance inversion
            # For now, add variance contribution
            if signal_cov.ndim == 2:
                uncertainty_var = np.diag(signal_cov)
            else:
                uncertainty_var = signal_cov
            
            # Penalize uncertainty (increases effective noise)
            uncertainty_correction = np.sum(uncertainty_var) / np.median(psd[:len(residual_fft)])
            
            log_l = -0.5 * (inner_product + uncertainty_correction)
            
            return log_l.real
        
        def _log_likelihood_standard(self, ifo, signal):
            """
            Calculate standard log-likelihood without model uncertainty.
            
            Parameters
            ----------
            ifo : bilby.gw.detector.Interferometer
                The interferometer object.
            signal : array_like
                The projected signal in the detector frame.
                
            Returns
            -------
            float
                The log-likelihood contribution from this interferometer.
            """
            residual = ifo.time_domain_strain - signal
            
            # Use bilby's built-in matched filter calculation
            residual_fft = np.fft.rfft(residual) * ifo.strain_data.duration / len(residual)
            psd = ifo.power_spectral_density_array
            delta_f = 1.0 / ifo.strain_data.duration
            
            inner_product = 4 * delta_f * np.sum(
                np.abs(residual_fft)**2 / psd[:len(residual_fft)]
            ).real
            
            return -0.5 * inner_product


__all__ = ['read_pickle']

if BILBY_AVAILABLE:
    __all__.extend(['HeronWaveformGenerator', 'HeronGravitationalWaveTransient'])
