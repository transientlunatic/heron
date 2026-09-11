#!/usr/bin/env python3
"""
Quick test of injection/inference with IMRPhenomPv2_FakeUncertainty.

This is a simplified version for debugging before running the full injection study.
Uses fake uncertainty to test the infrastructure.
"""

import sys
import numpy as np
import astropy.units as u
from pathlib import Path

from heron.models.lalsimulation import IMRPhenomPv2, IMRPhenomPv2_FakeUncertainty
from heron.detector import KNOWN_IFOS
from heron.models.lalnoise import KNOWN_PSDS
from heron.likelihood import TimeDomainLikelihood, TimeDomainLikelihoodModelUncertainty
from heron.sampling import NessaiSampler
from bilby.core.prior import Uniform, Cosine, Sine

def run_test_injection(
    mass_ratio=0.5,
    total_mass=20.0,
    distance=100.0,
    injection_id=0,
    with_uncertainty=True,
    outdir="test_injection_results"
):
    """
    Run a single test injection with IMRPhenomPv2_FakeUncertainty.

    Parameters
    ----------
    mass_ratio : float
        Mass ratio q = m2/m1
    total_mass : float
        Total mass in solar masses
    distance : float
        Luminosity distance in Mpc
    injection_id : int
        ID for this injection
    with_uncertainty : bool
        If True, use IMRPhenomPv2_FakeUncertainty for recovery
    outdir : str
        Output directory
    """

    print("=" * 70)
    print(f"Test Injection {injection_id}")
    print("=" * 70)
    print(f"  Mass ratio: {mass_ratio}")
    print(f"  Total mass: {total_mass} Msun")
    print(f"  Distance: {distance} Mpc")
    print(f"  With uncertainty: {with_uncertainty}")

    # Create output directory
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Injection parameters
    injection_params = {
        'mass_ratio': mass_ratio,
        'total_mass': total_mass * u.solMass,
        'luminosity_distance': distance * u.Mpc,
        'ra': 0.0,
        'dec': 0.0,
        'psi': 0.0,
        'theta_jn': 0.0,
        'phase': 0.0,
        'gpstime': 1126259462.0,  # GW150914 time
    }

    # Detector configuration
    detectors = {
        'AdvancedLIGOHanford': 'AdvancedLIGO',
        'AdvancedLIGOLivingston': 'AdvancedLIGO',
    }

    # Time array
    duration = 4.0  # seconds
    sample_rate = 4096  # Hz
    t0 = injection_params['gpstime']
    times = np.linspace(t0 - duration + 2, t0 + 2, int(duration * sample_rate))

    print("\n  Generating injection...")

    # Generate injection waveform (using standard IMRPhenomPv2)
    inj_model = IMRPhenomPv2()
    waveform = inj_model.time_domain(injection_params, times=times)

    # Project onto detectors (zero noise for testing)
    injections = {}
    for det_name, psd_name in detectors.items():
        detector = KNOWN_IFOS[det_name]()
        projected = waveform.project(detector)
        projected.channel = f"{detector.abbreviation}:Injection"
        # Use full detector name as key to match detectors dict
        injections[det_name] = projected

    print(f"    Generated waveforms for {len(injections)} detectors")

    # Set up recovery model
    if with_uncertainty:
        print("\n  Setting up recovery with fake uncertainty...")
        recovery_model = IMRPhenomPv2_FakeUncertainty(covariance=1e-24)
    else:
        print("\n  Setting up standard recovery...")
        recovery_model = IMRPhenomPv2()

    # Set up priors
    priors = {
        'mass_ratio': Uniform(
            minimum=0.2,
            maximum=0.8,
            name='mass_ratio',
            latex_label='$q$'
        ),
        'luminosity_distance': Uniform(
            minimum=10.0,
            maximum=500.0,
            name='luminosity_distance',
            latex_label='$d_L$'
        ),
        'ra': Uniform(
            minimum=0.0,
            maximum=2*np.pi,
            name='ra',
            latex_label=r'$\alpha$',
            boundary='periodic'
        ),
        'dec': Cosine(
            name='dec',
            latex_label=r'$\delta$'
        ),
        'theta_jn': Sine(
            name='theta_jn',
            latex_label=r'$\theta_{JN}$'
        ),
        'psi': Uniform(
            minimum=0.0,
            maximum=np.pi,
            name='psi',
            latex_label=r'$\psi$'
        ),
        'phase': Uniform(
            minimum=0.0,
            maximum=2*np.pi,
            name='phase',
            latex_label=r'$\phi$',
            boundary='periodic'
        ),
    }

    # Fixed parameters
    fixed_params = {
        'total_mass': total_mass,
        'gpstime': injection_params['gpstime'],
    }

    print("\n  Setting up likelihood...")

    # Set up likelihoods for each detector
    likelihoods = []
    for det_name, injection_data in injections.items():
        psd_model = KNOWN_PSDS[detectors[det_name]]()
        detector = KNOWN_IFOS[det_name]()

        if with_uncertainty:
            likelihood = TimeDomainLikelihoodModelUncertainty(
                data=injection_data,
                waveform=recovery_model,
                psd=psd_model,
                detector=detector,
                fixed_parameters=fixed_params
            )
        else:
            likelihood = TimeDomainLikelihood(
                data=injection_data,
                waveform=recovery_model,
                psd=psd_model,
                detector=detector,
                fixed_parameters=fixed_params
            )

        likelihoods.append(likelihood)

    # Combined likelihood
    from bilby.core.likelihood import JointLikelihood
    joint_likelihood = JointLikelihood(*likelihoods)

    print(f"    Set up {len(likelihoods)} detector likelihoods")

    # Set up sampler
    print("\n  Setting up sampler...")

    label = f"inj_{injection_id:04d}_{'uncertainty' if with_uncertainty else 'standard'}"

    sampler = NessaiSampler(
        likelihood=joint_likelihood,
        priors=priors,
        outdir=outdir,
        label=label,
        nlive=100,  # Small for testing
        maxmcmc=500,
        seed=12345 + injection_id,
    )

    print(f"    Sampler: Nessai with nlive=100")

    # Run sampling
    print("\n  Running parameter estimation...")
    print("    This may take a while...")

    try:
        result = sampler.run_sampler()

        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print(f"  Output directory: {outdir}")
        print(f"  Label: {label}")
        print(f"  Nested sampling evidence: {result.log_evidence:.2f}")

        return result

    except Exception as e:
        print("\n" + "=" * 70)
        print("FAILED!")
        print("=" * 70)
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Test injection with IMRPhenomPv2_FakeUncertainty'
    )
    parser.add_argument('--mass-ratio', type=float, default=0.5)
    parser.add_argument('--total-mass', type=float, default=20.0)
    parser.add_argument('--distance', type=float, default=100.0)
    parser.add_argument('--injection-id', type=int, default=0)
    parser.add_argument('--no-uncertainty', action='store_true',
                       help='Run standard analysis without uncertainty')
    parser.add_argument('--outdir', default='test_injection_results')

    args = parser.parse_args()

    result = run_test_injection(
        mass_ratio=args.mass_ratio,
        total_mass=args.total_mass,
        distance=args.distance,
        injection_id=args.injection_id,
        with_uncertainty=not args.no_uncertainty,
        outdir=args.outdir,
    )

    return 0 if result is not None else 1


if __name__ == '__main__':
    sys.exit(main())
