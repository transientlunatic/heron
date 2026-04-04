"""End-to-end tests for MismatchEvaluator, CalibrationEvaluator, and EvaluationReport.

Uses a SineGaussianWaveform as both the 'surrogate' (wrapped to implement
WaveformSurrogate.predict()) and the reference approximant. When the surrogate
*is* the reference, mismatch should be ~0 and z-scores should be well-behaved.
"""

import numpy as np
import pytest

from heron.models.testing import SineGaussianWaveform
from heron.types import Waveform, WaveformDict
from heron.evaluation.mismatch import MismatchEvaluator, MismatchResult
from heron.evaluation.calibration import CalibrationEvaluator, CalibrationResult
from heron.evaluation.report import EvaluationReport


class _SineGaussianSurrogate:
    """Thin wrapper around SineGaussianWaveform that satisfies the
    WaveformSurrogate protocol expected by the evaluators."""

    def __init__(self):
        self._sg = SineGaussianWaveform()

    @property
    def parameter_bounds(self):
        return {"amplitude": (0.5, 2.0)}

    @property
    def parameter_names(self):
        return ["amplitude"]

    def predict(self, parameters):
        time_cfg = parameters.get("time")
        if time_cfg is not None:
            times = np.linspace(time_cfg["lower"], time_cfg["upper"], time_cfg["number"])
        else:
            times = None
        return self._sg.time_domain(parameters, times=times)


class TestMismatchEvaluator:

    @pytest.fixture
    def evaluator(self):
        surrogate = _SineGaussianSurrogate()
        reference = SineGaussianWaveform()
        return MismatchEvaluator(surrogate, reference)

    def test_self_mismatch_near_zero(self, evaluator):
        """Surrogate vs itself: mismatch should be ~0."""
        result = evaluator.evaluate(
            n_points=5,
            parameter_bounds={"amplitude": (0.8, 1.2)},
            time_config={"lower": -0.1, "upper": 0.1, "number": 256},
            seed=42,
        )
        assert isinstance(result, MismatchResult)
        assert len(result.mismatches) == 5
        valid = result.mismatches[np.isfinite(result.mismatches)]
        assert len(valid) > 0
        # Self-comparison: all mismatches should be tiny
        assert np.all(valid < 1e-6)

    def test_result_has_statistics(self, evaluator):
        result = evaluator.evaluate(
            n_points=5,
            parameter_bounds={"amplitude": (0.8, 1.2)},
            time_config={"lower": -0.1, "upper": 0.1, "number": 256},
            seed=42,
        )
        assert result.worst_mismatch >= 0
        assert result.median_mismatch >= 0
        assert 0 <= result.fraction_below_1e3 <= 1
        assert 0 <= result.fraction_below_1e2 <= 1

    def test_result_summary(self, evaluator):
        result = evaluator.evaluate(
            n_points=3,
            parameter_bounds={"amplitude": (0.9, 1.1)},
            time_config={"lower": -0.05, "upper": 0.05, "number": 128},
            seed=0,
        )
        summary = result.summary()
        assert "Mismatch evaluation" in summary
        assert "Median" in summary
        assert "Worst" in summary

    def test_result_parameters_recorded(self, evaluator):
        result = evaluator.evaluate(
            n_points=4,
            parameter_bounds={"amplitude": (0.5, 2.0)},
            time_config={"lower": -0.1, "upper": 0.1, "number": 128},
            seed=7,
        )
        assert "amplitude" in result.parameters
        assert len(result.parameters["amplitude"]) == 4


class TestCalibrationEvaluator:

    @pytest.fixture
    def evaluator(self):
        surrogate = _SineGaussianSurrogate()
        reference = SineGaussianWaveform()
        return CalibrationEvaluator(surrogate, reference)

    def test_self_calibration(self, evaluator):
        """When surrogate is the reference, z-scores should be ~0."""
        result = evaluator.evaluate(
            n_points=5,
            parameter_bounds={"amplitude": (0.8, 1.2)},
            time_config={"lower": -0.05, "upper": 0.05, "number": 64},
            seed=42,
        )
        assert isinstance(result, CalibrationResult)
        assert result.z_scores.shape[0] == 5

        # z-scores for self-comparison: residual is 0, so z should be 0
        flat_z = result.z_scores.ravel()
        finite_z = flat_z[np.isfinite(flat_z)]
        assert len(finite_z) > 0
        assert np.all(np.abs(finite_z) < 1e-3)

    def test_coverage_computed(self, evaluator):
        result = evaluator.evaluate(
            n_points=5,
            parameter_bounds={"amplitude": (0.8, 1.2)},
            time_config={"lower": -0.05, "upper": 0.05, "number": 64},
            seed=42,
        )
        assert len(result.coverage) > 0
        # Self-comparison: all z~0, so coverage at all levels should be 100%
        for level, frac in result.coverage.items():
            assert frac == pytest.approx(1.0, abs=0.01)

    def test_ks_test_computed(self, evaluator):
        result = evaluator.evaluate(
            n_points=5,
            parameter_bounds={"amplitude": (0.8, 1.2)},
            time_config={"lower": -0.05, "upper": 0.05, "number": 64},
            seed=42,
        )
        # KS statistic should be a number
        assert result.ks_statistic >= 0

    def test_qq_data(self, evaluator):
        result = evaluator.evaluate(
            n_points=5,
            parameter_bounds={"amplitude": (0.8, 1.2)},
            time_config={"lower": -0.05, "upper": 0.05, "number": 64},
            seed=42,
        )
        theoretical, observed = result.qq_data
        assert len(theoretical) == len(observed)
        assert len(theoretical) > 0

    def test_summary(self, evaluator):
        result = evaluator.evaluate(
            n_points=3,
            parameter_bounds={"amplitude": (0.9, 1.1)},
            time_config={"lower": -0.05, "upper": 0.05, "number": 64},
            seed=0,
        )
        summary = result.summary()
        assert "calibration" in summary.lower()
        assert "coverage" in summary.lower()


class TestEvaluationReport:

    def _make_results(self):
        surrogate = _SineGaussianSurrogate()
        reference = SineGaussianWaveform()
        bounds = {"amplitude": (0.8, 1.2)}
        time_cfg = {"lower": -0.05, "upper": 0.05, "number": 64}

        mm_eval = MismatchEvaluator(surrogate, reference)
        mm_result = mm_eval.evaluate(n_points=3, parameter_bounds=bounds,
                                     time_config=time_cfg, seed=42)

        cal_eval = CalibrationEvaluator(surrogate, reference)
        cal_result = cal_eval.evaluate(n_points=3, parameter_bounds=bounds,
                                       time_config=time_cfg, seed=42)

        return mm_result, cal_result

    def test_summary_text(self):
        mm, cal = self._make_results()
        report = EvaluationReport(mismatch=mm, calibration=cal, name="test_model")
        summary = report.summary()
        assert "test_model" in summary
        assert "Mismatch" in summary
        assert "calibration" in summary.lower()

    def test_summary_mismatch_only(self):
        mm, _ = self._make_results()
        report = EvaluationReport(mismatch=mm, name="mm_only")
        summary = report.summary()
        assert "Mismatch" in summary

    def test_summary_calibration_only(self):
        _, cal = self._make_results()
        report = EvaluationReport(calibration=cal, name="cal_only")
        summary = report.summary()
        assert "calibration" in summary.lower()

    def test_summary_empty(self):
        report = EvaluationReport(name="empty")
        summary = report.summary()
        assert "no evaluation results" in summary

    def test_save_summary(self, tmp_path):
        mm, cal = self._make_results()
        report = EvaluationReport(mismatch=mm, calibration=cal, name="test")
        path = tmp_path / "report.txt"
        report.save_summary(path)
        assert path.exists()
        text = path.read_text()
        assert "Mismatch" in text

    def test_plot_all(self, tmp_path):
        """Plots should be generated if matplotlib is available."""
        pytest.importorskip("matplotlib")
        mm, cal = self._make_results()
        report = EvaluationReport(mismatch=mm, calibration=cal, name="test_plots")
        paths = report.plot_all(tmp_path / "plots")
        assert len(paths) > 0
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"

    def test_plot_all_without_matplotlib(self, tmp_path, monkeypatch):
        """Should gracefully return empty list if matplotlib is missing."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        mm, cal = self._make_results()
        report = EvaluationReport(mismatch=mm, calibration=cal, name="no_mpl")
        monkeypatch.setattr(builtins, "__import__", mock_import)
        paths = report.plot_all(tmp_path / "plots")
        assert paths == []
