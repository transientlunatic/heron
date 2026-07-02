"""
Evaluation report generation.

Produces summary statistics and optional matplotlib plots for
mismatch and calibration evaluations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .mismatch import MismatchResult, _build_reference_params
from .calibration import CalibrationResult

logger = logging.getLogger("heron.evaluation.report")


@dataclass
class EvaluationReport:
    """Combined evaluation report from mismatch and calibration results.

    Parameters
    ----------
    mismatch : MismatchResult or None
        Mismatch evaluation results.
    calibration : CalibrationResult or None
        Calibration evaluation results.
    name : str
        Model name for plot titles.
    surrogate : WaveformSurrogate or None
        Trained surrogate, used to generate waveform comparison plots.
    reference : WaveformApproximant or None
        Reference approximant, used to generate waveform comparison plots.
    """

    mismatch: MismatchResult | None = None
    calibration: CalibrationResult | None = None
    name: str = "surrogate"
    surrogate: object = None
    reference: object = None

    def summary(self) -> str:
        """Return a text summary of all evaluation results."""
        lines = [f"Evaluation report: {self.name}", "=" * 40]

        if self.mismatch is not None:
            lines.append("")
            lines.append(self.mismatch.summary())

        if self.calibration is not None:
            lines.append("")
            lines.append(self.calibration.summary())

        if self.mismatch is None and self.calibration is None:
            lines.append("(no evaluation results)")

        return "\n".join(lines)

    def save_summary(self, path: str | Path) -> None:
        """Write text summary to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.summary())
        logger.info(f"Summary saved to {path}")

    def plot_all(self, output_dir: str | Path) -> list[Path]:
        """Generate all available plots and save to output_dir.

        Requires matplotlib (optional dependency).

        Returns
        -------
        list[Path]
            Paths to the generated plot files.
        """
        try:
            import matplotlib
            matplotlib.use("agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed — skipping plots")
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        if self.mismatch is not None:
            saved.extend(self._plot_mismatch(plt, output_dir))

        if self.calibration is not None:
            saved.extend(self._plot_calibration(plt, output_dir))

        if self.surrogate is not None and self.reference is not None:
            saved.extend(self._plot_waveform_comparison(plt, output_dir))

        return saved

    def _plot_mismatch(self, plt, output_dir: Path) -> list[Path]:
        """Generate mismatch distribution plots."""
        result = self.mismatch
        saved = []

        fig, ax = plt.subplots(figsize=(8, 5))
        valid = result.mismatches[np.isfinite(result.mismatches)]
        if len(valid) == 0:
            plt.close(fig)
            return saved

        ax.hist(np.log10(np.maximum(valid, 1e-15)), bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(np.log10(1e-3), color="red", linestyle="--", label="Detection grade (1e-3)")
        ax.axvline(np.log10(1e-2), color="orange", linestyle="--", label="PE grade (1e-2)")
        ax.set_xlabel("log10(mismatch)")
        ax.set_ylabel("Count")
        ax.set_title(f"Mismatch distribution — {self.name}")
        ax.legend()
        fig.tight_layout()

        path = output_dir / "mismatch_histogram.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
        logger.info(f"Saved {path}")

        for param_name, param_vals in result.parameters.items():
            fig, ax = plt.subplots(figsize=(8, 5))
            finite_mask = np.isfinite(result.mismatches)
            mms = np.maximum(result.mismatches[finite_mask], 1e-15)
            ax.semilogy(param_vals[finite_mask], mms, "o", markersize=4, alpha=0.6)
            ax.axhline(1e-3, color="red", linestyle="--", alpha=0.5, label="1e-3")
            ax.axhline(1e-2, color="orange", linestyle="--", alpha=0.5, label="1e-2")
            ax.set_xlabel(param_name)
            ax.set_ylabel("Mismatch")
            ax.set_title(f"Mismatch vs {param_name} — {self.name}")
            ax.legend()
            fig.tight_layout()

            path = output_dir / f"mismatch_vs_{param_name}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(path)
            logger.info(f"Saved {path}")

        return saved

    def _plot_calibration(self, plt, output_dir: Path) -> list[Path]:
        """Generate calibration diagnostic plots."""
        result = self.calibration
        saved = []

        theoretical, observed = result.qq_data
        if len(theoretical) == 0:
            return saved

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(theoretical, observed, ".", markersize=1, alpha=0.5)
        lims = [min(theoretical[0], observed[0]), max(theoretical[-1], observed[-1])]
        ax.plot(lims, lims, "r--", label="Perfect calibration")
        ax.set_xlabel("Theoretical quantiles N(0,1)")
        ax.set_ylabel("Observed z-score quantiles")
        ax.set_title(f"Q-Q plot — {self.name}")
        ax.set_aspect("equal")
        ax.legend()
        fig.tight_layout()

        path = output_dir / "qq_plot.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
        logger.info(f"Saved {path}")

        flat_z = result.z_scores.ravel()
        flat_z = flat_z[np.isfinite(flat_z)]
        if len(flat_z) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(flat_z, bins=50, density=True, edgecolor="black", alpha=0.7, label="Observed")
            from scipy import stats
            x_grid = np.linspace(flat_z.min(), flat_z.max(), 200)
            ax.plot(x_grid, stats.norm.pdf(x_grid), "r-", linewidth=2, label="N(0,1)")
            ax.set_xlabel("z-score")
            ax.set_ylabel("Density")
            ax.set_title(f"Z-score distribution — {self.name}")
            ax.legend()
            fig.tight_layout()

            path = output_dir / "zscore_histogram.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(path)
            logger.info(f"Saved {path}")

        if result.coverage:
            fig, ax = plt.subplots(figsize=(6, 5))
            levels = sorted([float(k.strip("%")) / 100 for k in result.coverage])
            observed_cov = [result.coverage[f"{l:.0%}"] for l in levels]
            ax.plot(levels, observed_cov, "o-", label="Observed")
            ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
            ax.set_xlabel("Nominal coverage level")
            ax.set_ylabel("Observed coverage fraction")
            ax.set_title(f"Coverage — {self.name}")
            ax.legend()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            fig.tight_layout()

            path = output_dir / "coverage.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(path)
            logger.info(f"Saved {path}")

        return saved

    def _plot_waveform_comparison(self, plt, output_dir: Path) -> list[Path]:
        """Plot surrogate mean ± 2σ vs reference at three mass ratios.

        Produces a single figure with one column per mass ratio (low, mid, high),
        showing how well the surrogate reproduces the reference waveform and
        whether the uncertainty band comfortably contains the reference.
        """
        saved = []
        surrogate = self.surrogate
        reference = self.reference

        bounds = surrogate.parameter_bounds
        if "mass_ratio" not in bounds:
            logger.warning("Cannot generate waveform comparison: surrogate has no mass_ratio bounds")
            return saved

        q_lo, q_hi = bounds["mass_ratio"]
        q_mid = 0.5 * (q_lo + q_hi)
        mass_ratios = [q_lo + 0.05 * (q_hi - q_lo), q_mid, q_hi - 0.05 * (q_hi - q_lo)]
        labels = ["low q", "mid q", "high q"]

        time_config = {"lower": -0.5, "upper": 0.02, "number": 512}

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

        for ax, q, label in zip(axes, mass_ratios, labels):
            params = {"mass_ratio": q, "time": time_config}
            try:
                surr_wf = surrogate.predict(params)
                ref_params = _build_reference_params(surrogate, params)
                ref_wf = reference.time_domain(ref_params, times=surr_wf["plus"].times)

                times = surr_wf["plus"].times
                mean = surr_wf["plus"].data
                std = surr_wf["plus"].std
                ref = ref_wf["plus"].data

                n = min(len(mean), len(ref))
                ax.plot(times[:n], ref[:n], color="C0", linewidth=1.0, label="Reference")
                ax.plot(times[:n], mean[:n], color="C1", linewidth=1.0,
                        linestyle="--", label="Surrogate mean")
                if std is not None:
                    ax.fill_between(
                        times[:n],
                        (mean - 2 * std)[:n],
                        (mean + 2 * std)[:n],
                        color="C1", alpha=0.25, label="±2σ",
                    )
                ax.set_xlabel("Time (s)")
                ax.set_title(f"{label} (q={q:.2f})")
                if ax is axes[0]:
                    ax.set_ylabel("Strain")
                    ax.legend(fontsize=7)
            except Exception as e:
                logger.warning(f"Waveform comparison failed at q={q:.2f}: {e}")
                ax.text(0.5, 0.5, f"Failed\n{e}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8)
                ax.set_title(f"{label} (q={q:.2f})")

        fig.suptitle(f"Waveform comparison — {self.name}", fontsize=12)
        fig.tight_layout()

        path = output_dir / "waveform_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
        logger.info(f"Saved {path}")

        return saved
