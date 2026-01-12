"""
Investigate GPU/CPU data transfer patterns and identify optimization opportunities.

This test suite analyzes where data moves between GPU and CPU in the likelihood
evaluation pipeline, focusing on:
1. Waveform generation (GPU) -> likelihood evaluation (CPU)
2. Unnecessary transfers within likelihood computation
3. Opportunities to keep data on GPU
"""

import unittest
import numpy as np

# Check if torch/GPU is available
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False


class TestDataFlowAnalysis(unittest.TestCase):
    """Analyze data flow patterns in likelihood evaluation."""

    def test_current_data_flow_pattern(self):
        """
        Document the current data flow pattern.

        Typical workflow:
        1. Waveform generated on GPU (GPyTorch model)
        2. Waveform transferred to CPU (.cpu())
        3. Likelihood computed on CPU (NumPy)
        4. Result returned (scalar)

        Question: Can we keep more on GPU?
        """
        # print("\n=== Current Data Flow Pattern ===")
        # print("1. GPyTorch Model (GPU)")
        # print("   ├─ Input: parameters (CPU numpy) -> GPU tensor")
        # print("   ├─ Compute: GP prediction on GPU")
        # print("   └─ Output: waveform.data.cpu(), waveform.covariance.cpu()")
        # print("")
        # print("2. Likelihood Computation (CPU)")
        # print("   ├─ Input: waveform.data (CPU numpy), detector.data (CPU numpy)")
        # print("   ├─ Compute: overlap detection, residual, Cholesky solve")
        # print("   └─ Output: log_likelihood (scalar)")
        # print("")
        # print("Transfers per likelihood call:")
        # print("   GPU -> CPU: waveform data (~2048 floats), covariance (~2048x2048 floats)")
        # print("   CPU -> GPU: none (likelihood stays on CPU)")

    def test_identify_transfer_costs(self):
        """Measure the cost of GPU->CPU transfers for typical waveform sizes."""
        if not TORCH_AVAILABLE:
            self.skipTest("CUDA not available")

        import torch
        import time

        # Typical waveform sizes for GW analysis
        sizes = [512, 1024, 2048, 4096]

        # print("\n=== GPU->CPU Transfer Costs ===")
        # print(f"{'Size':<10} {'1D Transfer':<15} {'2D Transfer (NxN)':<20} {'Memory'}")

        for N in sizes:
            # Create data on GPU
            waveform_gpu = torch.randn(N, device='cuda')
            covariance_gpu = torch.randn(N, N, device='cuda')

            # Warm up
            for _ in range(10):
                _ = waveform_gpu.cpu()
                _ = covariance_gpu.cpu()

            # Time 1D transfer (waveform data)
            torch.cuda.synchronize()
            n_iter = 100
            start = time.time()
            for _ in range(n_iter):
                wf_cpu = waveform_gpu.cpu()
                torch.cuda.synchronize()
            wf_time = (time.time() - start) / n_iter

            # Time 2D transfer (covariance matrix)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(n_iter):
                cov_cpu = covariance_gpu.cpu()
                torch.cuda.synchronize()
            cov_time = (time.time() - start) / n_iter

            wf_kb = N * 4 / 1024  # 4 bytes per float32
            cov_kb = N * N * 4 / 1024

            # print(f"{N:<10} {wf_time*1e6:>10.2f} µs   {cov_time*1e6:>15.2f} µs       "
                  f"{wf_kb:.1f} KB + {cov_kb:.1f} KB")

    def test_likelihood_computation_cpu_only(self):
        """
        Current implementation: All likelihood computation on CPU.

        Question: What if we kept data on GPU and used PyTorch for linalg?
        """
        # print("\n=== CPU Likelihood Computation ===")
        # print("Current: scipy.linalg.solve_triangular (CPU)")
        # print("         numpy matrix operations (CPU)")
        # print("")
        # print("Potential GPU alternative:")
        # print("   torch.linalg.solve_triangular (GPU)")
        # print("   torch matrix operations (GPU)")
        # print("")
        # print("Trade-off:")
        # print("   Pro: Keep data on GPU, no transfers")
        # print("   Con: GPU overhead for small matrices (<2000x2000)")
        # print("   Pro: Batch multiple likelihood evaluations")

    @unittest.skipUnless(TORCH_AVAILABLE, "CUDA not available")
    def test_cpu_vs_gpu_cholesky_solve(self):
        """Compare CPU vs GPU for Cholesky solve at typical GW sizes."""
        import torch
        import time
        from scipy import linalg as scipy_linalg

        sizes = [512, 1024, 2048]

        # print("\n=== CPU vs GPU Cholesky Solve ===")
        # print(f"{'Size':<10} {'CPU Time':<15} {'GPU Time':<15} {'Transfer Time':<15} {'Total GPU':<15} {'Winner'}")

        for N in sizes:
            # Create positive definite matrix
            A_cpu = np.eye(N, dtype=np.float32)
            b_cpu = np.random.randn(N).astype(np.float32)

            # Compute Cholesky on CPU
            L_cpu = np.linalg.cholesky(A_cpu)

            # Warm up CPU
            for _ in range(5):
                _ = scipy_linalg.solve_triangular(L_cpu, b_cpu, lower=True)

            # Time CPU solve
            n_iter = 100
            start = time.time()
            for _ in range(n_iter):
                y = scipy_linalg.solve_triangular(L_cpu, b_cpu, lower=True)
                scipy_linalg.solve_triangular(L_cpu.T, y, lower=False)
            cpu_time = (time.time() - start) / n_iter

            # Transfer to GPU
            L_gpu = torch.from_numpy(L_cpu).cuda()
            b_gpu = torch.from_numpy(b_cpu).cuda().unsqueeze(1)  # Make 2D for solve_triangular

            # Warm up GPU
            for _ in range(10):
                _ = torch.linalg.solve_triangular(L_gpu, b_gpu, upper=False)

            # Time GPU solve (assuming data already on GPU)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(n_iter):
                y = torch.linalg.solve_triangular(L_gpu, b_gpu, upper=False)
                torch.linalg.solve_triangular(L_gpu.T, y, upper=True)
                torch.cuda.synchronize()
            gpu_time = (time.time() - start) / n_iter

            # Time transfer cost
            start = time.time()
            for _ in range(n_iter):
                _ = torch.from_numpy(b_cpu).cuda()
                torch.cuda.synchronize()
            transfer_time = (time.time() - start) / n_iter

            total_gpu = gpu_time + transfer_time
            winner = "CPU" if cpu_time < total_gpu else "GPU"

            # print(f"{N:<10} {cpu_time*1e6:>10.2f} µs   {gpu_time*1e6:>10.2f} µs   "
                  f"{transfer_time*1e6:>10.2f} µs   {total_gpu*1e6:>10.2f} µs   {winner}")


class TestOptimizationOpportunities(unittest.TestCase):
    """Identify specific optimization opportunities."""

    def test_opportunity_1_keep_waveform_on_gpu(self):
        """
        Opportunity 1: Keep waveform on GPU until needed.

        Current:
            waveform.data = mean.cpu()  # Immediate transfer
            likelihood_eval(waveform)    # Uses CPU numpy

        Optimized:
            waveform.data_gpu = mean     # Keep on GPU
            likelihood_eval_gpu(waveform) # Compute on GPU

        Benefit: Eliminate one GPU->CPU transfer per likelihood call
        Cost: Need GPU-enabled likelihood computation
        """
        # print("\n=== Opportunity 1: Keep Waveform on GPU ===")
        # print("Current overhead: ~20-50 µs for waveform transfer (2048 samples)")
        # print("Potential saving: Transfer eliminated if likelihood on GPU")
        # print("")
        # print("Implementation:")
        # print("  1. Add .data_gpu attribute to Waveform")
        # print("  2. Create TorchLikelihood class with GPU operations")
        # print("  3. Only transfer final log_likelihood result (scalar)")

    def test_opportunity_2_batch_likelihood_evaluations(self):
        """
        Opportunity 2: Batch multiple likelihood evaluations.

        Current:
            for params in parameter_samples:
                waveform = model(params)     # GPU
                ll = likelihood(waveform)     # CPU

        Optimized:
            waveforms = model(params_batch)   # GPU, batched
            lls = likelihood_batch(waveforms) # GPU, batched

        Benefit: Amortize transfer costs, vectorized computation
        Cost: Requires batched waveform generation and likelihood
        """
        # print("\n=== Opportunity 2: Batch Evaluations ===")
        # print("Current: Sequential evaluation, one transfer per waveform")
        # print("Optimized: Batch evaluation, one transfer for N waveforms")
        # print("")
        # print("Speedup potential:")
        # print("  Transfer overhead: ~50 µs/waveform * N waveforms")
        # print("  Batched transfer: ~50 µs for N waveforms")
        # print("  For N=100: ~5000 µs -> 50 µs (100x on transfers)")
        # print("")
        # print("GPU vectorization benefit:")
        # print("  Single: N * compute_time")
        # print("  Batched: ~1.5 * compute_time (rough estimate)")

    def test_opportunity_3_minimize_covariance_transfers(self):
        """
        Opportunity 3: Avoid transferring full covariance matrix.

        Current:
            covariance = observed_pred.covariance_matrix.cpu()  # N x N transfer

        Options:
            1. Only transfer if actually used (TimeDomainLikelihoodModelUncertainty)
            2. Compute likelihood on GPU (no transfer needed)
            3. Transfer compressed representation (Cholesky, eigendecomp)

        Benefit: Huge - covariance is N^2, much larger than waveform (N)
        """
        # print("\n=== Opportunity 3: Covariance Transfer ===")

        N = 2048
        wf_size = N * 4 / 1024  # KB
        cov_size = N * N * 4 / 1024  # KB

        # print(f"For N={N}:")
        # print(f"  Waveform size: {wf_size:.1f} KB")
        # print(f"  Covariance size: {cov_size:.1f} KB")
        # print(f"  Ratio: {cov_size/wf_size:.0f}x larger!")
        # print("")
        # print("Options:")
        # print("  1. Lazy transfer: Only transfer covariance if needed")
        # print("     (TimeDomainLikelihood doesn't use it)")
        # print("  2. Transfer Cholesky factor: Same size but might be useful")
        # print("  3. GPU likelihood: No transfer needed")


class TestRecommendations(unittest.TestCase):
    """Generate specific recommendations."""

    def test_recommendation_summary(self):
        """Summarize findings and recommendations."""
        # print("\n" + "="*70)
        # print("RECOMMENDATIONS FOR GPU/CPU TRANSFER OPTIMIZATION")
        # print("="*70)
        # print("")
        # print("SHORT-TERM (Easy wins):")
        # print("  1. Lazy covariance transfer")
        # print("     - Only transfer in TimeDomainLikelihoodModelUncertainty")
        # print("     - Saves ~16 MB transfer for 2048 samples")
        # print("     - Low effort, high impact")
        # print("")
        # print("  2. Waveform property caching")
        # print("     - Cache .cpu() result to avoid repeated transfers")
        # print("     - Useful if waveform accessed multiple times")
        # print("")
        # print("MEDIUM-TERM (More involved):")
        # print("  3. GPU-enabled likelihood class")
        # print("     - Implement TorchLikelihood with PyTorch operations")
        # print("     - Keep waveform on GPU, compute likelihood on GPU")
        # print("     - Transfer only scalar log_likelihood result")
        # print("     - Benefit increases with batch evaluation")
        # print("")
        # print("LONG-TERM (Major refactor):")
        # print("  4. Batched evaluation pipeline")
        # print("     - Batch waveform generation and likelihood evaluation")
        # print("     - Amortize transfer costs over many samples")
        # print("     - Useful for nested sampling, MCMC warmup")
        # print("")
        # print("ANALYSIS:")
        # print("  - Single evaluation: GPU overhead may not be worth it (<2048)")
        # print("  - Batched evaluation: GPU becomes very beneficial")
        # print("  - Current bottleneck: Cholesky recomputation (already optimized)")
        # print("  - Transfer cost: ~100-200 µs (small compared to Cholesky)")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
