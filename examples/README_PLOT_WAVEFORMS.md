# Plotting Waveform Draws from Trained GPR Models

This guide explains how to use the `plot-waveforms` command to visualize waveform samples drawn from trained Gaussian Process Regression models.

## Overview

The `plot-waveforms` command loads a trained GPR model and generates plots showing:
- **Posterior samples**: Multiple waveform draws from the GP posterior distribution
- **Mean prediction**: The expected waveform at given parameters
- **Uncertainty quantification**: 95% confidence intervals
- **Comparison across parameters**: How waveforms vary with mass ratio

This is useful for:
- Understanding the uncertainty in your model's predictions
- Visualizing what the trained model has learned
- Identifying regions where the model is confident vs uncertain
- Comparing waveform morphology across the parameter space

## Prerequisites

You need:
1. A trained GPR model (created with `heron train-gpr`)
2. The HDF5 file containing the model state and training data
3. A configuration file specifying plotting parameters

## Basic Usage

### Single Waveform Plot

Generate a plot showing waveform draws at a single mass ratio:

```bash
heron plot-waveforms --config plot_waveforms_config.yml
```

This will create plots showing:
- 10 random samples from the posterior (translucent blue lines)
- The mean prediction (red line)
- 95% confidence interval (red shaded region)

### Compare Multiple Mass Ratios

Generate a comparison plot showing how waveforms change across mass ratios:

```bash
heron plot-waveforms --config plot_waveforms_config.yml --compare-mass-ratios
```

This creates a multi-panel plot with one subplot per mass ratio.

### Both Polarizations

Plot both plus and cross polarizations:

```bash
heron plot-waveforms --config plot_waveforms_config.yml --polarization both
```

### Custom Number of Samples

Control how many waveform samples to draw:

```bash
heron plot-waveforms --config plot_waveforms_config.yml --num-samples 20
```

### Override Mass Ratio

Plot a specific mass ratio not in the config:

```bash
heron plot-waveforms --config plot_waveforms_config.yml --mass-ratio 0.75
```

## Configuration File

See [plot_waveforms_config.yml](plot_waveforms_config.yml) for a complete example.

### Key Configuration Sections

#### Model Data
```yaml
training:
  data_file: training_data_test.h5  # HDF5 file with trained model
  group_name: IMR_training_test      # Training group name

model_name: gpr_imr_nonspin  # Model name (used to load state)
```

#### Hyperparameters
Must match those used during training:
```yaml
hyperparameters:
  warp_scale: 2       # Time warping factor
  output_scale: 1.0   # Output scaling (deprecated)
```

#### Mean Function
If your model was trained with a mean function (e.g., IMRPhenomD):
```yaml
mean_function:
  enabled: true
  parameters:
    total_mass: 20.0
    distance: 100.0
    delta_t: 0.000244140625  # 1/4096
    f_lower: 20.0
    f_ref: 20.0
```

#### Plotting Settings
```yaml
plotting:
  # Time range (geometric units)
  time_min: -0.1
  time_max: 0.05
  time_points: 500

  # Mass ratio(s)
  mass_ratio: 0.8  # For single plots
  mass_ratios: [0.5, 0.7, 0.9, 1.0]  # For comparison plots

  # Whether to overlay training data
  show_training_data: false

  # Output directory
  output_dir: plots_waveforms
```

## Output Files

The command generates PNG files in the specified output directory:

### Single Mass Ratio Mode
- `waveform_draws_plus_q0.80.png` - Plus polarization
- `waveform_draws_cross_q0.80.png` - Cross polarization

### Comparison Mode
- `waveform_draws_comparison_plus.png` - Multi-panel plus polarization
- `waveform_draws_comparison_cross.png` - Multi-panel cross polarization

## Understanding the Plots

### Posterior Samples (Blue Lines)
Each translucent blue line is a random draw from the GP posterior. These represent plausible waveforms given the training data and model assumptions.

- **Dense clustering**: High confidence - model is certain
- **Wide spread**: High uncertainty - model is unsure
- **Variations**: Show the range of waveforms consistent with the data

### Mean Prediction (Red Line)
The expected waveform - the average over all possible waveforms weighted by their probability.

### Confidence Interval (Red Shaded)
The 95% confidence region (mean ± 2 standard deviations). If the model is well-calibrated:
- ~95% of new observations should fall in this region
- Narrow regions indicate confidence
- Wide regions indicate uncertainty

### Typical Patterns
- **Near training data**: Narrow confidence, samples cluster around mean
- **Far from training data**: Wide confidence, samples spread out
- **Pre-merger region**: Often lower uncertainty (smooth behavior)
- **Merger/ringdown**: May have higher uncertainty (complex dynamics)

## Examples

### Visualize a Trained Model
After training with `heron train-gpr`, immediately check the results:

```bash
heron plot-waveforms --config plot_waveforms_config.yml --polarization both --num-samples 20
```

### Compare Mass Ratio Dependence
Understand how waveforms change with mass ratio:

```bash
heron plot-waveforms --config plot_waveforms_config.yml --compare-mass-ratios --num-samples 10
```

### High-Resolution Single Waveform
Generate a detailed plot at a specific mass ratio:

```bash
heron plot-waveforms --config plot_waveforms_config.yml \
  --mass-ratio 0.65 \
  --num-samples 50 \
  --polarization plus
```

## Tips

1. **More samples** → Better sense of uncertainty distribution, but slower
2. **Fewer samples** → Faster, but may miss tail behavior
3. **Time range**: Adjust `time_min`/`time_max` to focus on interesting regions
4. **Mass ratios**: Choose ratios near training data for best results
5. **Mean function**: Ensure settings match training configuration

## Troubleshooting

### "Model file not found"
- Check that `training.data_file` points to the correct HDF5 file
- Verify the file was created by `heron train-gpr`

### "Failed to load model state"
- Ensure `model_name` matches the name used in training
- Check that `group_name` matches the training data group
- Verify the model was fully trained (not an intermediate checkpoint)

### "Failed to load training data"
- Ensure `group_name` matches the training configuration
- Check that the HDF5 file contains the expected group

### Wide/Unrealistic Uncertainty
- May indicate insufficient training data
- Try training for more iterations
- Check that hyperparameters are appropriate

### All Samples Look Identical
- Model may be overconfident (very low noise level)
- Check training diagnostics - may need regularization
- Try drawing more samples to see subtle variations

## Integration with Training Pipeline

Typical workflow:

```bash
# 1. Generate training data
heron training-data --config training_data_config.yml

# 2. Train GPR model
heron train-gpr --config gpr_training_config.yml

# 3. Plot waveform draws
heron plot-waveforms --config plot_waveforms_config.yml --polarization both

# 4. Compare mass ratios
heron plot-waveforms --config plot_waveforms_config.yml --compare-mass-ratios

# 5. Check specific regions of interest
heron plot-waveforms --config plot_waveforms_config.yml --mass-ratio 0.95 --num-samples 30
```

## Advanced Usage

### Overlaying Training Data
Enable in config or modify code to show training points:
```yaml
plotting:
  show_training_data: true
```

This helps verify that:
- The model interpolates well through training data
- Uncertainty grows away from training points
- The model isn't overfitting

### Custom Time Ranges
Focus on specific waveform features:
```yaml
plotting:
  # Just the merger
  time_min: -0.01
  time_max: 0.01

  # Extended inspiral
  time_min: -0.2
  time_max: 0.05
```

### Multiple Configurations
Create different configs for different purposes:
- `plot_overview.yml` - Quick check with few samples
- `plot_detailed.yml` - Publication-quality with many samples
- `plot_comparison.yml` - Multi-ratio comparison

## See Also

- [Training Data Generation](README_TRAINING_PIPELINE.md) - Generate training data
- [GPR Training](README_TRAINING_PIPELINE.md) - Train GPR models
- [IMRPhenomD Mean Function](README_IMRPHENOMD_MEAN.md) - Using mean functions
