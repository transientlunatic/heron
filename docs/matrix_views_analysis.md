# Matrix Views and Submatrix Cholesky Caching Analysis

## Executive Summary

**TL;DR:** NumPy already uses memory views efficiently. The real optimization opportunity is **caching Cholesky decompositions of frequently-used submatrices** for partial overlap scenarios, which provides up to **34x speedup**.

## Background

In the likelihood computation, when there's partial overlap between data and waveform, we extract a submatrix:
```python
C_scaled = self.C_scaled[a[0]:a[1], a[0]:a[1]]
L_sub = np.linalg.cholesky(C_scaled)
```

The question was: Are we making unnecessary copies, and can we optimize this?

## Findings

### 1. Memory Views Behavior

NumPy **already creates views** for submatrix slicing:
- `C[100:200, 100:200]` creates a view (shares memory with parent)
- Memory saved: ~7.8 MB for a 1000x1000 submatrix
- Views are non-contiguous, but this doesn't hurt performance

### 2. Contiguity Testing

Making data contiguous with `np.ascontiguousarray()` **does not improve performance**:
```
Cholesky on 1000x1000 submatrix:
  Non-contiguous view: 26.409 ms
  Contiguous array:    28.264 ms
  Speedup:             0.93x (SLOWER!)
```

**Conclusion:** Don't use `ascontiguousarray()` - it adds overhead without benefit.

### 3. Real Optimization Opportunity

For partial overlap scenarios, the bottleneck is **recomputing Cholesky decomposition**:

```
Full Likelihood Evaluation (N=2048, overlap=1500):
  Current (recompute Cholesky each time): 120.394 ms
  Cached submatrix Cholesky:                3.567 ms
  Speedup:                                 33.76x
```

## Recommendation

### Current Code (Optimal for Views)
```python
# In log_likelihood(), line 226
C_scaled = self.C_scaled[a[0]:a[1], a[0]:a[1]]  # This is fine - it's a view
L_sub = np.linalg.cholesky(C_scaled)  # THIS is the bottleneck
```

### Proposed Optimization: Submatrix Cholesky Cache

For scenarios with repeated partial overlaps (common in MCMC sampling), cache the Cholesky:

```python
# In __init__ or as needed
self._submatrix_cholesky_cache = {}  # key: (start, end), value: L_sub

# In log_likelihood()
cache_key = (a[0], a[1])
if cache_key in self._submatrix_cholesky_cache:
    L_sub = self._submatrix_cholesky_cache[cache_key]
else:
    C_sub = self.C_scaled[a[0]:a[1], a[0]:a[1]]
    L_sub = np.linalg.cholesky(C_sub)
    self._submatrix_cholesky_cache[cache_key] = L_sub
```

### Cache Management

**Pros:**
- 34x speedup for repeated evaluations
- Common in MCMC: many likelihood calls with same data/waveform overlap
- Low memory cost: Only stores frequently-used submatrices

**Cons:**
- Extra memory for cache
- Need cache invalidation strategy
- Only helps when overlap pattern repeats

**When to use:**
- MCMC sampling (same data, varying parameters)
- Nested sampling with fixed data segments
- Multiple likelihood evaluations on same overlap region

**When NOT useful:**
- Single likelihood evaluations
- Constantly changing overlap regions
- Streaming/online analysis

### Implementation Strategy

1. **Simple LRU cache** (max 10-20 entries)
2. **Key:** `(start_idx, end_idx, hash(C_scaled))`  # Include data fingerprint
3. **Invalidate** when PSD/data changes
4. **Optional feature** (disabled by default, enable with `cache_submatrix=True`)

## Performance Summary

| Scenario | Time | vs Baseline |
|----------|------|-------------|
| Current implementation (view) | 120 ms | 1.0x |
| With ascontiguousarray | 124 ms | 0.97x (worse) |
| Cached submatrix Cholesky | 3.6 ms | 33.8x (much better!) |

## Test Coverage

All findings verified with:
- `tests/test_matrix_views.py` - View behavior analysis (11 tests)
- `tests/test_matrix_contiguity_optimization.py` - Contiguity impact (4 tests)

## Conclusion

**No changes needed for view handling** - NumPy is already optimal.

**Optional enhancement:** Implement submatrix Cholesky caching for MCMC/nested sampling use cases where repeated partial overlaps occur. This is a high-value, low-risk optimization for parameter estimation workflows.
