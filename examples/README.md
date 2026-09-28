# Heron Examples

Example configurations for training waveform surrogate models.

## Training Configs

### Exact GP on a fixed grid

**File:** `train_exact_gp.yaml`

Trains an exact GP surrogate at 10 mass ratios from IMRPhenomPv2. This is the simplest training mode — good for getting started.

```bash
pip install heron[lal]   # needs lalsuite for waveform generation
heron train --settings examples/train_exact_gp.yaml
```

### Sparse GP with active learning

**File:** `train_sparse_active.yaml`

Trains a sparse variational GP using iterative active learning. Starts with Sobol-sampled points, then adds training data where the model is most uncertain. Uses a Newtonian inspiral mean function so the GP only learns the merger/ringdown residual.

```bash
heron train --settings examples/train_sparse_active.yaml
```

### Train from pre-existing data

**File:** `train_from_data.yaml`

Trains from an HDF5 training set (e.g. from a previous run or an NR catalogue). Does not require lalsuite.

```bash
heron train --settings examples/train_from_data.yaml
```

## Evaluation

**File:** `evaluate.yaml`

Evaluates a trained surrogate against a reference approximant. Computes mismatch distributions and uncertainty calibration metrics, generates diagnostic plots.

```bash
heron evaluate --settings examples/evaluate.yaml
```

Chain training and evaluation:

```bash
heron train --settings examples/train_exact_gp.yaml && \
heron evaluate --settings examples/evaluate.yaml
```

Use `reference: SineGaussian` in the config to evaluate without lalsuite.

## Using a trained model

```python
from heron.models.gp.exact import ExactGPSurrogate

model = ExactGPSurrogate.load("checkpoints/exact_gp.pt")
wf = model.predict({
    "mass_ratio": 0.5,
    "time": {"lower": -0.5, "upper": 0.02, "number": 500},
})

strain = wf["plus"].data            # (500,) array
covariance = wf["plus"].covariance  # (500, 500) matrix
```

## Legacy examples

The following files are from the old heron architecture and may not work with the current code:

- `bilby_integration_example.py` — bilby likelihood with waveform uncertainty (inference code has moved)
- `gpr_training_config.yml` — old two-stage training pipeline config
- `example_gpr_with_imrphenomd_mean.py` — old IMRPhenomD mean function example
