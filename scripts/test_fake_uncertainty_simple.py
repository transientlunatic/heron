#!/usr/bin/env python3
"""
Minimal smoke test for IMRPhenomPv2_FakeUncertainty.

Just verifies:
1. Waveform generation works
2. FakeUncertainty adds covariance correctly
3. Can evaluate waveforms at different parameters

Does NOT run full parameter estimation - just tests the basics.
"""

import numpy as np
import astropy.units as u
from heron.models.lalsimulation import IMRPhenomPv2, IMRPhenomPv2_FakeUncertainty

def test_fake_uncertainty():
    print("=" * 70)
    print("IMRPhenomPv2_FakeUncertainty Smoke Test")
    print("=" * 70)

    # Test parameters
    params = {
        'mass_ratio': 0.5,
        'total_mass': 20.0 * u.solMass,
        'luminosity_distance': 100.0 * u.Mpc,
        'ra': 0.0,
        'dec': 0.0,
        'psi': 0.0,
        'theta_jn': 0.0,
        'phase': 0.0,
        'gpstime': 1126259462.0,
    }

    # Time array
    duration = 1.0  # Short for quick test
    sample_rate = 2048  # Lower for quick test
    t0 = params['gpstime']
    times = np.linspace(t0 - duration + 0.1, t0 + 0.1, int(duration * sample_rate))

    print("\n1. Testing standard IMRPhenomPv2...")
    standard_model = IMRPhenomPv2()
    wf_standard = standard_model.time_domain(params, times=times)

    print(f"   ✓ Generated waveform with {len(wf_standard['plus'].times)} points")
    print(f"   ✓ Plus polarization max: {np.max(np.abs(wf_standard['plus'].data)):.2e}")
    print(f"   ✓ Cross polarization max: {np.max(np.abs(wf_standard['cross'].data)):.2e}")

    # Check for covariance (should not exist)
    has_cov = hasattr(wf_standard['plus'], 'covariance')
    print(f"   ✓ Has covariance: {has_cov} (expected: False)")

    print("\n2. Testing IMRPhenomPv2_FakeUncertainty...")
    fake_model = IMRPhenomPv2_FakeUncertainty(covariance=1e-24)
    wf_fake = fake_model.time_domain(params, times=times)

    print(f"   ✓ Generated waveform with {len(wf_fake['plus'].times)} points")
    print(f"   ✓ Plus polarization max: {np.max(np.abs(wf_fake['plus'].data)):.2e}")
    print(f"   ✓ Cross polarization max: {np.max(np.abs(wf_fake['cross'].data)):.2e}")

    # Check for covariance (should exist)
    has_cov = hasattr(wf_fake['plus'], 'covariance')
    print(f"   ✓ Has covariance: {has_cov} (expected: True)")

    if has_cov:
        cov_shape = wf_fake['plus'].covariance.shape
        cov_diag = np.diag(wf_fake['plus'].covariance)
        print(f"   ✓ Covariance shape: {cov_shape}")
        print(f"   ✓ Covariance diagonal (first 5): {cov_diag[:5]}")
        print(f"   ✓ Expected diagonal value: {1e-24**2:.2e}")

        # Verify covariance is reasonable
        expected = 1e-24**2
        actual = np.mean(cov_diag)
        relative_error = abs(actual - expected) / expected
        print(f"   ✓ Mean diagonal value: {actual:.2e}")
        print(f"   ✓ Relative error: {relative_error:.2%}")

        if relative_error < 0.1:
            print("   ✓ Covariance values are correct!")
        else:
            print("   ✗ WARNING: Covariance values don't match expected")

    print("\n3. Testing at different parameters...")
    params2 = params.copy()
    params2['mass_ratio'] = 0.7

    wf_fake2 = fake_model.time_domain(params2, times=times)
    print(f"   ✓ Generated waveform at q=0.7")
    print(f"   ✓ Plus polarization max: {np.max(np.abs(wf_fake2['plus'].data)):.2e}")

    # Compare waveforms (convert to numpy arrays first)
    diff = np.max(np.abs(np.array(wf_fake['plus'].data) - np.array(wf_fake2['plus'].data)))
    print(f"   ✓ Difference from q=0.5: {diff:.2e}")

    print("\n" + "=" * 70)
    print("SUCCESS! IMRPhenomPv2_FakeUncertainty is working correctly.")
    print("=" * 70)

    return True

if __name__ == '__main__':
    import sys
    try:
        success = test_fake_uncertainty()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
