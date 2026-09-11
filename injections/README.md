# Injection Study with Fake Uncertainty

This directory contains settings files for testing the `IMRPhenomPv2_FakeUncertainty` waveform model in injection/inference studies.

## Test Cases

### Fake Uncertainty Tests
- `fake_uncertainty_test_q03.yaml` - q=0.3, tests low mass ratio
- `fake_uncertainty_test_q05.yaml` - q=0.5, fiducial test case
- `fake_uncertainty_test_q07.yaml` - q=0.7, tests high mass ratio

**Key features:**
- Injection: `IMRPhenomPv2` (standard, no uncertainty)
- Inference: `IMRPhenomPv2_FakeUncertainty` (with model uncertainty)
- Likelihood: `TimeDomainLikelihoodModelUncertainty`

### Baseline Comparison
- `standard_test_q05.yaml` - Standard inference without uncertainty

**Key features:**
- Injection: `IMRPhenomPv2`
- Inference: `IMRPhenomPv2` (same as injection)
- Likelihood: `TimeDomainLikelihood` (no uncertainty)

## Workflow

### Option 1: Using Asimov (Recommended)

Asimov manages the full injection → inference workflow automatically:

```bash
# Initialize asimov with the ledger
asimov manage injections/asimov_ledger.yaml

# Submit all injection jobs
asimov apply injections/asimov_ledger.yaml --all --pipeline "heron injection"

# After injections complete, submit inference jobs
asimov apply injections/asimov_ledger.yaml --all --pipeline "heron"

# Monitor status
asimov monitor injections/asimov_ledger.yaml
```

**Benefits:**
- Automatic dependency management (inference waits for injection)
- Job monitoring and status tracking
- Automatic result collection
- Web page generation

### Option 2: Manual Workflow

#### Step 1: Create Injections

Run on cluster:
```bash
cd /home/daniel/repositories/ligo/heron
condor_submit injections/run_injection_study.sub
```

Or run locally for a single test case:
```bash
heron injection --settings injections/fake_uncertainty_test_q05.yaml
```

This creates:
- `H1_injection.gwf` - Frame file for Hanford
- `L1_injection.gwf` - Frame file for Livingston
- PSDs and diagnostic plots in the pages directory

#### Step 2: Run Inference

After injections complete, run inference:
```bash
heron inference --settings injections/fake_uncertainty_test_q05.yaml
```

This:
- Loads the injection frame files
- Runs nessai sampler with specified priors
- Generates posterior samples and plots
- Saves results to the pages directory

## Comparison Strategy

The goal is to compare:

1. **Standard analysis** (standard_test_q05.yaml):
   - No model uncertainty
   - Baseline posterior widths
   - Should recover injection parameters perfectly (zero-noise)

2. **Fake uncertainty analysis** (fake_uncertainty_test_*.yaml):
   - Includes model uncertainty via covariance matrix
   - Posterior should be wider due to waveform uncertainty
   - Tests infrastructure before using real GP model

## Expected Results

For zero-noise injections:
- Standard analysis: Tight posteriors centered on injection values
- Fake uncertainty: Wider posteriors, centered on injection values
- Posterior widths should scale with uncertainty covariance

## Output Locations

Results go to `/data/www.astro/daniel/heron-paper/`:
- `fake-uncertainty/q03/` - q=0.3 test
- `fake-uncertainty/q05/` - q=0.5 test
- `fake-uncertainty/q07/` - q=0.7 test
- `standard/q05/` - Baseline comparison

## Parameters

All test cases use:
- Total mass: 20 M_sun (matches GP training data)
- Distance: 100 Mpc (good SNR for testing)
- Detectors: H1, L1 (LIGO)
- PSDs: AdvancedLIGO design sensitivity
- Duration: 4 seconds
- Sample rate: 4096 Hz

## Next Steps

After validating fake uncertainty:
1. Replace `IMRPhenomPv2_FakeUncertainty` with trained GP model
2. Run full injection campaign with varying parameters
3. Compare posteriors with/without GP uncertainty
4. Generate publication figures
