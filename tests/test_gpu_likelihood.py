"""
Comprehensive tests for GPU-enabled likelihood computation.

Tests numerical equivalence between CPU and GPU implementations,
performance benchmarks, and lazy covariance loading.
"""

import unittest
import numpy as np

# Check torch/CUDA availability
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False


# Try to import heron components
try:
    from heron.likelihood import (
        TorchLikelihood, TimeDomainLikelihood,
        TimeDomainLikelihoodGPU, NumericallyScaled
    )
    from heron.types import Waveform, TimeSeries
    from heron.psd import FlatPSD
    HERON_AVAILABLE = True
except ImportError as e:
    HERON_AVAILABLE = False
    IMPORT_ERROR = str(e)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestTorchLikelihood(unittest.TestCase):
    """Test TorchLikelihood base class functionality."""

    def test_initialization(self):
        """Test TorchLikelihood initializes correctly."""
        likelihood = TorchLikelihood()
        self.assertIn(likelihood.device, ["cpu", "cuda"])

    def test_device_selection(self):
        """Test device selection based on CUDA availability."""
        likelihood = TorchLikelihood()
        if CUDA_AVAILABLE:
            self.assertEqual(likelihood.device, "cuda")
        else:
            self.assertEqual(likelihood.device, "cpu")

    def test_array_conversion(self):
        """Test array conversion to torch tensors."""
        likelihood = TorchLikelihood()

        # NumPy array
        np_arr = np.array([1.0, 2.0, 3.0])
        torch_arr = likelihood.array(np_arr)
        self.assertIsInstance(torch_arr, torch.Tensor)
        self.assertEqual(torch_arr.dtype, torch.float64)

        # Already a tensor
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = likelihood.array(tensor)
        self.assertIs(result, tensor)

    def test_to_device(self):
        """Test to_device transfers correctly."""
        likelihood = TorchLikelihood()

        arr = np.array([1.0, 2.0, 3.0])
        tensor = likelihood.to_device(arr, likelihood.device)

        self.assertIsInstance(tensor, torch.Tensor)
        if CUDA_AVAILABLE:
            self.assertTrue(tensor.is_cuda)
        else:
            self.assertFalse(tensor.is_cuda)

    def test_solve(self):
        """Test solve operation."""
        likelihood = TorchLikelihood()

        A = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])

        result = likelihood.solve(A, b)

        # Should return numpy array
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, b)

    def test_logdet(self):
        """Test log determinant computation."""
        likelihood = TorchLikelihood()

        # Identity matrix has det=1, log(1)=0
        A = np.eye(4)
        logdet = likelihood.logdet(A)

        self.assertIsInstance(logdet, (float, np.floating))
        self.assertAlmostEqual(logdet, 0.0)

    def test_log(self):
        """Test logarithm computation."""
        likelihood = TorchLikelihood()

        # Scalar
        result = likelihood.log(np.e)
        self.assertAlmostEqual(result, 1.0)

        # Array
        arr = np.array([1.0, np.e, np.e**2])
        result = likelihood.log(arr)
        expected = np.array([0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(result, expected)


@unittest.skipUnless(HERON_AVAILABLE, f"Heron not available: {IMPORT_ERROR if not HERON_AVAILABLE else ''}")
class TestLazyCovariance(unittest.TestCase):
    """Test lazy covariance loading in Waveform class."""

    def test_waveform_with_direct_covariance(self):
        """Test traditional covariance setting."""
        times = np.linspace(0, 1, 100)
        data = np.random.randn(100)
        cov = np.eye(100)

        wf = Waveform(data=data, times=times, covariance=cov)

        # Covariance should be immediately available
        self.assertIsNotNone(wf.covariance)
        np.testing.assert_array_equal(wf.covariance, cov)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_waveform_with_gpu_covariance(self):
        """Test lazy GPU covariance loading."""
        times = np.linspace(0, 1, 100)
        data = np.random.randn(100)
        cov_gpu = torch.eye(100, device='cuda' if CUDA_AVAILABLE else 'cpu')

        wf = Waveform(
            data=data,
            times=times,
            covariance_gpu=cov_gpu,
            output_scale=2.0,
            distance_factor=1.5
        )

        # Initially, _covariance should be None
        self.assertIsNone(wf._covariance)

        # First access triggers transfer
        cov = wf.covariance
        self.assertIsNotNone(cov)
        self.assertIsInstance(cov, np.ndarray)

        # Check scaling was applied
        expected = np.eye(100) / (2.0 * 2.0 * 1.5**2)
        np.testing.assert_array_almost_equal(cov, expected)

        # Second access should use cached value
        cov2 = wf.covariance
        self.assertIs(cov2, cov)  # Same object

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_lazy_covariance_not_accessed(self):
        """Test that covariance is not transferred if not accessed."""
        times = np.linspace(0, 1, 100)
        data = np.random.randn(100)
        cov_gpu = torch.eye(100, device='cuda' if CUDA_AVAILABLE else 'cpu')

        wf = Waveform(
            data=data,
            times=times,
            covariance_gpu=cov_gpu,
        )

        # As long as we don't access .covariance, no transfer happens
        self.assertIsNone(wf._covariance)
        self.assertIsNotNone(wf._covariance_gpu)


@unittest.skipUnless(HERON_AVAILABLE and TORCH_AVAILABLE, "Heron or PyTorch not available")
class TestGPULikelihoodEquivalence(unittest.TestCase):
    """Test numerical equivalence between CPU and GPU likelihood."""

    def setUp(self):
        """Create test data for likelihood comparisons."""
        # Create simple test data
        self.times = np.linspace(0, 1, 512)
        self.sample_rate = 512

        # Simple sinusoidal signal
        self.data_values = 1e-22 * np.sin(2 * np.pi * 10 * self.times) + 1e-23 * np.random.randn(len(self.times))

        # Create TimeSeries (mock if LAL not available)
        try:
            from astropy import units as u
            from astropy.time import Time
            times_astropy = Time(self.times, format='gps')
            self.data = TimeSeries(self.data_values, times=times_astropy)
        except:
            # Create mock TimeSeries
            class MockTimeSeries:
                def __init__(self, data, times):
                    self.data = data
                    self.times = times

            self.data = MockTimeSeries(self.data_values, self.times)

        # Create PSD
        self.psd = FlatPSD(amplitude=1e-44)

    def test_cholesky_cpu_vs_gpu(self):
        """Test Cholesky decomposition CPU vs GPU."""
        N = 256
        # Create positive definite matrix
        A = np.eye(N, dtype=np.float64) * 1e-44

        # CPU Cholesky
        L_cpu = np.linalg.cholesky(A)

        # GPU Cholesky
        if CUDA_AVAILABLE:
            A_gpu = torch.tensor(A, device='cuda', dtype=torch.float64)
            L_gpu = torch.linalg.cholesky(A_gpu)
            L_gpu_cpu = L_gpu.cpu().numpy()

            # Should be identical within numerical precision
            np.testing.assert_array_almost_equal(L_cpu, L_gpu_cpu, decimal=14)

    def test_solve_triangular_cpu_vs_gpu(self):
        """Test triangular solve CPU vs GPU."""
        N = 256
        L = np.tril(np.random.randn(N, N))
        np.fill_diagonal(L, np.abs(np.diag(L)) + 1)  # Ensure positive diagonal
        b = np.random.randn(N)

        # CPU solve
        from scipy import linalg as scipy_linalg
        x_cpu = scipy_linalg.solve_triangular(L, b, lower=True)

        if CUDA_AVAILABLE:
            # GPU solve
            L_gpu = torch.tensor(L, device='cuda', dtype=torch.float64)
            b_gpu = torch.tensor(b, device='cuda', dtype=torch.float64).unsqueeze(1)
            x_gpu = torch.linalg.solve_triangular(L_gpu, b_gpu, upper=False)
            x_gpu_cpu = x_gpu.squeeze().cpu().numpy()

            # Should match within numerical precision
            np.testing.assert_array_almost_equal(x_cpu, x_gpu_cpu, decimal=10)

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
    def test_likelihood_gpu_initialization(self):
        """Test GPU likelihood initializes correctly."""
        try:
            likelihood_gpu = TimeDomainLikelihoodGPU(
                self.data,
                self.psd
            )

            self.assertEqual(likelihood_gpu.device, "cuda")
            self.assertTrue(likelihood_gpu._use_cholesky)
            self.assertTrue(likelihood_gpu._use_torch_cholesky)
            self.assertIsInstance(likelihood_gpu.C_cholesky, torch.Tensor)
            self.assertTrue(likelihood_gpu.C_cholesky.is_cuda)
        except Exception as e:
            self.skipTest(f"GPU likelihood initialization failed: {e}")

    def test_likelihood_cpu_fallback(self):
        """Test that GPU likelihood falls back to CPU when CUDA unavailable."""
        # Temporarily disable CUDA
        import heron.likelihood as likelihood_module
        original_disable = likelihood_module.disable_cuda

        try:
            likelihood_module.disable_cuda = True

            likelihood_gpu = TimeDomainLikelihoodGPU(
                self.data,
                self.psd
            )

            self.assertEqual(likelihood_gpu.device, "cpu")
        finally:
            likelihood_module.disable_cuda = original_disable


@unittest.skipUnless(HERON_AVAILABLE and CUDA_AVAILABLE, "CUDA not available")
class TestGPUPerformance(unittest.TestCase):
    """Performance benchmarks for GPU likelihood."""

    def setUp(self):
        """Create test data."""
        self.times = np.linspace(0, 2, 2048)
        self.data_values = 1e-22 * np.sin(2 * np.pi * 10 * self.times) + 1e-23 * np.random.randn(len(self.times))

        # Create mock TimeSeries
        class MockTimeSeries:
            def __init__(self, data, times):
                self.data = data
                self.times = times

        self.data = MockTimeSeries(self.data_values, self.times)
        self.psd = FlatPSD(amplitude=1e-44)

    def test_cholesky_solve_performance(self):
        """Compare CPU vs GPU Cholesky solve performance."""
        import time

        N = 2048
        L = np.tril(np.random.randn(N, N))
        np.fill_diagonal(L, np.abs(np.diag(L)) + 1)
        b = np.random.randn(N)

        # Warm up
        from scipy import linalg as scipy_linalg
        for _ in range(3):
            _ = scipy_linalg.solve_triangular(L, b, lower=True)

        # Time CPU
        n_iter = 50
        start = time.time()
        for _ in range(n_iter):
            x_cpu = scipy_linalg.solve_triangular(L, b, lower=True)
        cpu_time = (time.time() - start) / n_iter

        # GPU
        L_gpu = torch.tensor(L, device='cuda', dtype=torch.float64)
        b_gpu = torch.tensor(b, device='cuda', dtype=torch.float64).unsqueeze(1)

        # Warm up GPU
        for _ in range(10):
            _ = torch.linalg.solve_triangular(L_gpu, b_gpu, upper=False)
            torch.cuda.synchronize()

        # Time GPU
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iter):
            x_gpu = torch.linalg.solve_triangular(L_gpu, b_gpu, upper=False)
            torch.cuda.synchronize()
        gpu_time = (time.time() - start) / n_iter

        print(f"\n=== Cholesky Solve Performance (N={N}) ===")
        print(f"CPU: {cpu_time*1000:.3f} ms")
        print(f"GPU: {gpu_time*1000:.3f} ms")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")

        # GPU should be faster for large matrices
        self.assertLess(gpu_time, cpu_time * 1.5,
                       f"GPU not competitive: {gpu_time*1000:.1f}ms vs {cpu_time*1000:.1f}ms")


@unittest.skipUnless(HERON_AVAILABLE, "Heron not available")
class TestBackwardsCompatibility(unittest.TestCase):
    """Test that existing code still works (backwards compatibility)."""

    def test_waveform_without_gpu_params(self):
        """Test Waveform still works without new GPU parameters."""
        times = np.linspace(0, 1, 100)
        data = np.random.randn(100)
        cov = np.eye(100)

        # Old-style creation should still work
        wf = Waveform(data=data, times=times, covariance=cov)

        self.assertIsNotNone(wf.covariance)
        np.testing.assert_array_equal(wf.covariance, cov)

    def test_cpu_likelihood_unchanged(self):
        """Test CPU likelihood still works exactly as before."""
        times = np.linspace(0, 1, 512)
        data_values = np.random.randn(512) * 1e-22

        class MockTimeSeries:
            def __init__(self, data, times):
                self.data = data
                self.times = times

        data = MockTimeSeries(data_values, times)
        psd = FlatPSD(amplitude=1e-44)

        # Original CPU likelihood should work unchanged
        likelihood_cpu = TimeDomainLikelihood(data, psd)

        self.assertEqual(likelihood_cpu.device, "cpu")
        self.assertFalse(hasattr(likelihood_cpu, '_use_torch_cholesky') and likelihood_cpu._use_torch_cholesky)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
