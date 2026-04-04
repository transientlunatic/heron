"""
CLI entry point for evaluating trained surrogate models.

Usage:
  heron evaluate --settings config.yaml

Can be chained after training:
  heron train --settings train.yaml && heron evaluate --settings eval.yaml
"""

import logging
import os

import click

from heron.utils import load_yaml

logger = logging.getLogger("heron.evaluate")

# Model registry — same as train.py
MODEL_REGISTRY = {
    "exact": ("heron.models.gp.exact", "ExactGPSurrogate"),
    "sparse": ("heron.models.gp.sparse", "SparseGPSurrogate"),
}

APPROXIMANT_REGISTRY = {
    "IMRPhenomPv2": ("heron.models.lalsimulation", "IMRPhenomPv2"),
    "SEOBNRv3": ("heron.models.lalsimulation", "SEOBNRv3"),
    "SineGaussian": ("heron.models.testing", "SineGaussianWaveform"),
}


def _import_class(module_path: str, class_name: str):
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def heron_evaluate(settings):
    """Run the full evaluation pipeline.

    Parameters
    ----------
    settings : str or dict
        Path to a YAML settings file, or a parsed settings dict.

    Returns
    -------
    EvaluationReport
    """
    if isinstance(settings, str):
        settings = load_yaml(settings)

    eval_settings = settings["evaluation"]

    if "logging" in settings:
        level = settings.get("logging", {}).get("level", "warning")
        LOGGER_LEVELS = {
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "warning": logging.WARNING,
        }
        logging.basicConfig(level=LOGGER_LEVELS[level])

    # --- Load surrogate ---
    checkpoint = eval_settings["checkpoint"]
    model_type = eval_settings.get("model", "exact")
    device = eval_settings.get("device", "cpu")

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. Available: {list(MODEL_REGISTRY)}")

    model_cls = _import_class(*MODEL_REGISTRY[model_type])
    model = model_cls.load(checkpoint, device=device)
    logger.info(f"Loaded {model_type} model from {checkpoint}")

    # --- Load reference ---
    ref_name = eval_settings.get("reference", "IMRPhenomPv2")
    if ref_name not in APPROXIMANT_REGISTRY:
        raise ValueError(f"Unknown reference '{ref_name}'. Available: {list(APPROXIMANT_REGISTRY)}")
    ref_cls = _import_class(*APPROXIMANT_REGISTRY[ref_name])
    reference = ref_cls()
    logger.info(f"Reference: {ref_name}")

    # --- Parameters ---
    parameter_bounds = eval_settings.get("parameter_bounds")
    if parameter_bounds is not None:
        parameter_bounds = {k: tuple(v) for k, v in parameter_bounds.items()}

    time_config = eval_settings.get("time", {"lower": -0.5, "upper": 0.02, "number": 512})
    seed = eval_settings.get("seed")
    name = eval_settings.get("name", os.path.splitext(os.path.basename(checkpoint))[0])

    # --- Run evaluations ---
    from heron.evaluation.mismatch import MismatchEvaluator
    from heron.evaluation.calibration import CalibrationEvaluator
    from heron.evaluation.report import EvaluationReport

    mismatch_result = None
    calibration_result = None

    run_mismatch = eval_settings.get("mismatch", True)
    run_calibration = eval_settings.get("calibration", True)

    if run_mismatch:
        n_mismatch = eval_settings.get("n_mismatch", 200)
        logger.info(f"Running mismatch evaluation ({n_mismatch} points)")
        mm_eval = MismatchEvaluator(model, reference)
        mismatch_result = mm_eval.evaluate(
            n_points=n_mismatch,
            parameter_bounds=parameter_bounds,
            time_config=time_config,
            seed=seed,
        )
        click.echo(mismatch_result.summary())

    if run_calibration:
        n_calibration = eval_settings.get("n_calibration", 100)
        logger.info(f"Running calibration evaluation ({n_calibration} points)")
        cal_eval = CalibrationEvaluator(model, reference)
        calibration_result = cal_eval.evaluate(
            n_points=n_calibration,
            parameter_bounds=parameter_bounds,
            time_config=time_config,
            seed=seed,
        )
        click.echo(calibration_result.summary())

    # --- Report ---
    report = EvaluationReport(
        mismatch=mismatch_result,
        calibration=calibration_result,
        name=name,
    )

    output_dir = eval_settings.get("output_dir", "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    report.save_summary(os.path.join(output_dir, "report.txt"))
    click.echo(f"\nSummary saved to {output_dir}/report.txt")

    if eval_settings.get("plots", True):
        paths = report.plot_all(os.path.join(output_dir, "plots"))
        if paths:
            click.echo(f"Generated {len(paths)} plots in {output_dir}/plots/")
        else:
            click.echo("No plots generated (matplotlib not available)")

    return report


@click.command()
@click.option("--settings", required=True, help="Path to evaluation settings YAML")
def evaluate(settings):
    """Evaluate a trained heron surrogate model."""
    click.echo("Evaluating heron surrogate model")
    heron_evaluate(settings)
    click.echo("Evaluation complete")
