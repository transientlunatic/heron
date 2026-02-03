# Heron GPR Training Pipeline

This guide explains how to use the complete training pipeline for GPR waveform models, both standalone and via asimov orchestration.

## Overview

The training pipeline consists of two stages:

1. **Training Data Generation** - Generate waveform manifolds from approximants or NR catalogs
2. **GPR Model Training** - Train Gaussian Process Regression models on the training data

Both stages can be run:
- **Standalone**: Using the `heron` CLI commands directly
- **Via Asimov**: Orchestrated through HTCondor for production workflows

## Standalone Usage

### Stage 1: Generate Training Data

Create a configuration file (see `training_data_config.yml` for example):

```yaml
name: IMRPhenomPv2_NonSpinning_Test

waveform_source:
  type: approximant
  approximant: IMRPhenomPv2

parameter_space:
  fixed:
    total_mass: 60.0
    sample_rate: 4096
    duration: 0.5
  varied:
    mass_ratio:
      lower: 0.1
      upper: 1.0
      step: 0.1

polarizations:
  - plus
  - cross

output:
  file: training_data_test.h5
  group_name: IMR_training_test

pages directory: plots_test
```

Run the training data generation:

```bash
heron training-data --config training_data_config.yml
```

This will:
- Generate waveforms across the parameter space
- Store them in HDF5 format
- Create diagnostic plots in `plots_test/`

### Stage 2: Train GPR Model

Create a training configuration file (see `gpr_training_config.yml` for example):

```yaml
name: IMRPhenomPv2_NonSpinning_GPR

training:
  data_file: training_data_test.h5
  group_name: IMR_training_test

model_name: gpr_imr_nonspin

hyperparameters:
  iterations: 1000
  learning_rate: 0.05
  output_scale: 1.0e27
  warp_scale: 2
  checkpoint_frequency: 100

pages directory: plots_gpr_test
```

Run the training:

```bash
heron train-gpr --config gpr_training_config.yml
```

This will:
- Load training data from the HDF5 file
- Train GPR models for both polarizations
- Save model states back to the same HDF5 file
- Generate diagnostic plots during training
- Create validation plots comparing predictions to training data

### Output Files

After both stages, you'll have:

**HDF5 File (`training_data_test.h5`)** containing:
- Training data in `/training data/` group
- Trained model states in `/model states/` group

**Diagnostic Plots**:
- `plots_test/` - Training data diagnostics
  - `parameter_space.png` - Parameter coverage
  - `waveform_samples.png` - Example waveforms
  - `manifold_heatmap.png` - Waveform manifold visualization

- `plots_gpr_test/` - Training diagnostics
  - `training_loss.png` - Loss evolution
  - `hyperparameters.png` - Hyperparameter evolution
  - `predictions_plus.png` - Plus polarization validation
  - `predictions_cross.png` - Cross polarization validation

## Asimov Integration

For production workflows, use asimov to orchestrate the pipeline.

### Setting Up Asimov Productions

Create an asimov ledger entry for your event, then add two productions:

#### Production 1: Training Data Generation

```yaml
productions:
  - name: IMRPhenomPv2_training_data
    pipeline: heron training data
    status: ready
    meta:
      waveform_source:
        type: approximant
        approximant: IMRPhenomPv2
      parameter_space:
        fixed:
          total_mass: 60.0
          sample_rate: 4096
          duration: 0.5
        varied:
          mass_ratio:
            lower: 0.1
            upper: 1.0
            step: 0.1
      polarizations:
        - plus
        - cross
      output:
        group_name: IMR_training
```

#### Production 2: GPR Training (with dependency)

```yaml
  - name: IMRPhenomPv2_gpr_training
    pipeline: heron gpr training
    status: ready
    dependencies:
      - IMRPhenomPv2_training_data
    meta:
      model_name: gpr_imr
      training:
        group_name: IMR_training
      hyperparameters:
        iterations: 1000
        learning_rate: 0.05
        output_scale: 1.0e27
        warp_scale: 2
        checkpoint_frequency: 100
```

**Key points**:
- The GPR training production has `IMRPhenomPv2_training_data` in its `dependencies`
- Asimov will automatically wait for the training data to complete before starting GPR training
- The training data file is passed automatically using the `collect_assets()` pattern
- Both productions can be in `ready` status - asimov handles the dependency ordering

### Submitting to HTCondor

```bash
asimov manage submit --event <event_name> --production IMRPhenomPv2_training_data
asimov manage submit --event <event_name> --production IMRPhenomPv2_gpr_training
```

Asimov will:
1. Submit the training data generation job
2. Monitor for completion (looks for HDF5 file)
3. Automatically submit the GPR training job when data is ready
4. Monitor GPR training (looks for model states in HDF5)
5. Generate HTML diagnostic pages accessible via the asimov web interface

### Monitoring Progress

View progress through the asimov web interface:
- Training data page shows parameter space coverage and example waveforms
- GPR training page shows loss curves, hyperparameter evolution, and validation plots
- Both update automatically as jobs run

## Advanced Configuration

### Using Peak Sampling (for Inspiral Waveforms)

For long inspiral waveforms, reduce data size by sampling at peaks:

```yaml
optimization:
  use_peak_sampling: true
  peak_threshold: 0.5  # Optional
```

### Validation Subsampling

For faster training during development, subsample the training data:

```yaml
hyperparameters:
  validation_samples: 10000  # Use only 10k samples
```

### Custom Lengthscale Constraints

The GPR model uses RBF kernels with constraints. To modify (requires code changes):

See `heron/models/gpytorch.py` lines 52-61 for the kernel definition.

## Troubleshooting

### Out of Memory / Training Gets Killed

GPR training is memory-intensive. If your training process gets killed or runs out of memory:

**1. Reduce training data size** (most effective):
```yaml
hyperparameters:
  validation_samples: 5000  # Start with 5000, adjust as needed
```

**2. Use CPU instead of GPU** (GPUs have limited memory):
Edit `heron/train_gpr.py`:
```python
disable_cuda = True  # Line 28
```

**3. Reduce iterations** (won't help memory but speeds up testing):
```yaml
hyperparameters:
  iterations: 500
```

**Memory scaling**: Exact GP memory usage scales as O(N²) where N is the number of training points. For example:
- 5,000 samples ≈ 1-2 GB
- 10,000 samples ≈ 4-8 GB
- 20,000 samples ≈ 16-32 GB

The code automatically enables garbage collection every 100 iterations to help manage memory.

### Training Loss Not Decreasing

Try adjusting:
- Learning rate (increase to 0.1 or decrease to 0.01)
- Number of iterations (increase to 2000+)
- Output scale (try 1e26 or 1e28)

### Missing Dependencies

If you get `ModuleNotFoundError` for `lal`, `gpytorch`, or `torch`, ensure you have:
```bash
pip install lalsuite gpytorch torch
```

For GPU support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Using Trained Models

Once trained, models can be loaded and used for inference:

```python
from heron.training.data import DataWrapper
from heron.models.gpytorch import ExactGPModelKeOps
import torch

# Load the data file with trained states
data = DataWrapper("training_data_test.h5")

# Get the model state
state_dict = data.get_states("gpr_imr_nonspin_plus")

# Initialize a model with the same architecture
model = ExactGPModelKeOps(train_x, train_y)

# Load the trained parameters
model.load_state_dict(state_dict['hyperparameters'])
model.eval()

# Use for prediction
with torch.no_grad():
    predictions = model(test_x)
```

See `heron/models/gpytorch.py` for the full `HeronNonSpinningApproximant` class which handles this automatically.

## File Locations

All configuration templates and examples are in:
- `examples/training_data_config.yml` - Training data generation example
- `examples/gpr_training_config.yml` - GPR training example
- `heron/asimov/heron_training_data_template.yml` - Asimov template for data generation
- `heron/asimov/heron_gpr_training_template.yml` - Asimov template for GPR training

## References

- GPyTorch documentation: https://gpytorch.ai/
- Asimov documentation: https://github.com/lscsoft/asimov
- LALSimulation: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/
