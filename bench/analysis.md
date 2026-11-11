# mojo-antropy Performance Analysis

**Benchmark run:** `pixi run bench` on Intel Xeon E5-2697 v4 @ 2.30GHz (Linux x86-64)  
**Upstream:** antropy 0.2.2 (Python + Numba where applicable)  
**All times:** Best of 3 runs, warmed, identical contiguous inputs

---

## Ranked Hotspot Table

| Rank | Function | Input Size | Mojo Time | Upstream Time | Ratio | Bytes Touched (Read) | Est. FLOPs | FLOPs/Byte | Bound | Time Location |
|------|----------|------------|-----------|---------------|-------|---------------------|------------|------------|-------|---------------|
| 1 | `lziv_complexity` (binary) | 50k symbols | 161.18 ms | 801.42 ms | **4.97× faster** | 6.25 KB (packed) + 400 KB (windows) | ~3.2G bitwise ops | ~6,400 | **Compute** | Mojo kernel (bit-packed parser) |
| 2 | `sample_entropy` (order=2) | 4k samples | 4.41 ms | 46.13 ms | **10.45× faster** | 32 KB | ~16M SIMD compares | ~500 | **Compute** | Mojo kernel (SIMD + parallel) |
| 3 | `hjorth_params` | 1000×10k | 88.98 ms | 267.74 ms | **3.01× faster** | 80 MB × 6 passes = 480 MB | ~300M | ~0.6 | **Memory** | Mojo kernel (multi-pass variance) |
| 4 | `detrended_fluctuation` | 500k samples | 92.61 ms | 351.84 ms | **3.80× faster** | 4 MB × ~15 scales = 60 MB | ~200M | ~3.3 | **Compute** | Mojo kernel (linear regression per window) |
| 5 | `perm_entropy` (order=3) | 1M samples | 8.67 ms | 37.40 ms | **4.31× faster** | 8 MB | ~6M compares + log | ~0.8 | **Memory** | Mojo kernel (specialized counter) |
| 6 | `katz_fd` | 1000×10k | 24.68 ms | 171.44 ms | **6.95× faster** | 80 MB | ~20M abs+max+add | ~0.25 | **Memory** | Mojo kernel (single pass) |
| 7 | `petrosian_fd` | 1000×10k | 14.99 ms | 63.91 ms | **4.26× faster** | 80 MB | ~10M sub+cmp | ~0.12 | **Memory** | Mojo kernel (single pass) |
| 8 | `higuchi_fd` (kmax=10) | 500k samples | 3.91 ms | 7.65 ms | **1.96× faster** | 4 MB × 10 passes = 40 MB | ~50M SIMD abs+add | ~1.2 | **Memory** | Mojo kernel (SIMD length sum) |
| 9 | `num_zerocross` | 1000×10k | 11.75 ms | 24.39 ms | **2.08× faster** | 80 MB | ~10M compares | ~0.12 | **Memory** | Mojo kernel (single pass) |

**Note:** FLOPs/Byte is approximate (integer ops counted as 1 FLOP).  
Memory-bound threshold ~1-2 FLOPs/byte on this CPU.

---

## Python/ctypes Overhead Breakdown

| Function | Wrapper Time | Kernel Time | Overhead | Notes |
|----------|--------------|-------------|----------|-------|
| `perm_entropy` (order=3) | 9.98 ms | 8.54 ms | 1.44 ms (14%) | Single ctypes call, no Python loops |
| `hjorth_params` | 91.32 ms | 90.08 ms | 1.24 ms (1.3%) | Single ctypes call per batch |
| `sample_entropy` (order=2) | 6.24 ms (parallel) | 18.37 ms (serial) | **Negative** | Parallel wrapper uses ThreadPoolExecutor; serial kernel is slower |
| `num_zerocross` | ~11.7 ms | 11.25 ms | ~0.5 ms (4%) | Single ctypes call |
| `petrosian_fd` | ~15.0 ms | 14.67 ms | ~0.3 ms (2%) | Single ctypes call |
| `katz_fd` | ~24.7 ms | 24.10 ms | ~0.6 ms (2%) | Single ctypes call |

**Key findings:**
- No per-element ctypes re-entry. All batching happens inside Mojo.
- `sample_entropy` parallel path (8 threads) beats the serial kernel 3× despite Python thread overhead.
- Wrapper overhead is negligible (< 2%) for all functions except `perm_entropy` order=3 where it's 14% due to the tiny kernel time.

---

## Per-Call Allocations / Copies

| Function | Python-side Allocations | Copies |
|----------|------------------------|--------|
| All functions | 1 result array + scratch (keys, counts, work, walk) | Input converted to C-contiguous float64 once |
| `lziv_complexity` (binary) | Packed bytes + windows array | `np.packbits` copy (~6 KB) |
| `sample_entropy` (parallel) | 8 × 2 int64 counts array | None (shared input address) |

No per-iteration allocations inside hot loops.

---

## Python-Level Loops That Re-enter Ctypes

**None found.** All exported Mojo functions process entire batches (all rows) in a single ctypes call. The Python wrapper prepares contiguous buffers and calls once.

---

## Hotspot → Next Phase Mapping

| Function | Verdict | Reasoning (FLOPs/Byte) |
|----------|---------|------------------------|
| `lziv_complexity` (binary) | **simd** | 6,400 FLOPs/byte → compute-bound. Bitwise ops on packed data are scalar; could use 64-bit SIMD for parallel window compare. Branch-heavy but data-dependent. |
| `sample_entropy` (order=2) | **parallel** (already done) | 500 FLOPs/byte → compute-bound. Already uses 8-thread parallel + SIMD. GPU? Launch + copy cost (~10 μs) > kernel time (4 ms). Not worthwhile. |
| `hjorth_params` | **simd** | 0.6 FLOPs/byte → memory-bound. Multi-pass variance. Fuse passes + SIMD reduction (Welford or two-pass) to cut memory traffic 3-6×. |
| `detrended_fluctuation` | **unroll + simd** | 3.3 FLOPs/byte → compute-bound. Linear regression per window per scale. Inner loops are scalar; unroll + SIMD dot products. Parallel across scales/windows. |
| `perm_entropy` (order=3) | **simd** | 0.8 FLOPs/byte → memory-bound (but small working set). Specialized counter already optimal for order=3. General path (order>4) uses hash table — could SIMD the ordinal rank. |
| `katz_fd` | **simd** | 0.25 FLOPs/byte → memory-bound. Single pass, abs+max+sum. SIMD reduction for curve length + farthest. |
| `petrosian_fd` | **simd** | 0.12 FLOPs/byte → memory-bound. Sign-change count. SIMD compare + popcount for vectorized crossing detection. |
| `higuchi_fd` | **simd** (already partial) | 1.2 FLOPs/byte → memory-bound. Already uses SIMD for length sum. Remainder loop and smaller-curve loops are scalar; unroll + vectorize. |
| `num_zerocross` | **simd** | 0.12 FLOPs/byte → memory-bound. Compare + XOR + popcount per SIMD lane. |

**GPU verdict:** None meet the ~2 FLOPs/byte *and* have sufficient parallelism to hide launch+copy latency. `sample_entropy` is compute-dense but only 4k elements; `detrended_fluctuation` has parallelism across scales but irregular memory access. CPU SIMD + threading is the right tier.

---

## Slower-Than-Upstream Flags

**None.** All 9 measured entry points are faster in mojo-antropy (1.96× to 10.45×).

---

## Additional Measurements (Non-Bench Suite Sizes)

| Function | Input | Mojo Kernel | Upstream | Ratio |
|----------|-------|-------------|----------|-------|
| `perm_entropy` (order=5) | 10k | 0.78 ms | 1.54 ms | **1.97× faster** |
| `app_entropy` (order=2) | 5k | 202.14 ms | 42.03 ms | **4.8× SLOWER** ⚠️ |
| `sample_entropy` (order=3) | 3k | 23.53 ms | 23.06 ms | ~1× (parity) |
| `lziv_complexity` (general) | 50k | 942.78 ms | 1038.36 ms | **1.1× faster** |

**Critical finding:** `app_entropy` is **4.8× slower** than upstream at 5k samples. Its O(n²) double loop (no SIMD, no early exit) is the top optimization target. The upstream uses Numba JIT which likely optimizes the inner loop better than the current scalar Mojo code.

---

## Recommended Phase Order

1. **app_entropy** → **simd + unroll** (O(n²) scalar loop, 4.8× regression)
2. **hjorth_params** → **simd + fuse passes** (memory-bound, 6 passes over 80 MB)
3. **detrended_fluctuation** → **unroll + simd + parallel** (compute-bound, irregular but parallelizable)
4. **perm_entropy (general)** → **simd ordinal rank** (hash table path for order>4)
5. **katz_fd / petrosian_fd / num_zerocross / higuchi_fd** → **simd** (all memory-bound, simple vectorizable loops)
6. **lziv_complexity (binary)** → **simd** (compute-bound, bitwise parallelism)
7. **sample_entropy** → **none** (already optimal: SIMD + 8-thread parallel)