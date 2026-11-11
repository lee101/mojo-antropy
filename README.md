# mojo-antropy

Entropy and time-series complexity measures from
[AntroPy](https://github.com/raphaelvallat/antropy), implemented as native
[Mojo](https://www.modular.com/mojo) kernels with a NumPy-friendly Python API.

The covered functions keep AntroPy's public names and signatures for the
documented subset, so code can switch imports:

```python
import numpy as np
import mojo_antropy as ant

rng = np.random.default_rng(42)
x = rng.normal(size=10_000)

print(ant.perm_entropy(x, order=3, normalize=True))
print(ant.sample_entropy(x))
print(ant.higuchi_fd(x, kmax=10))
```

## Coverage

| area | functions |
| --- | --- |
| Entropy | `perm_entropy`, `app_entropy`, `sample_entropy` |
| Complexity | `lziv_complexity` |
| Time domain | `num_zerocross`, `hjorth_params` |
| Fractal dimension | `petrosian_fd`, `katz_fd`, `higuchi_fd`, `detrended_fluctuation` |

This is 10 of AntroPy 0.2.2's 12 public measures. `spectral_entropy` and
`svd_entropy` are not covered: their compute-heavy operations are SciPy's
FFT/Welch implementation and LAPACK SVD, so a thin Mojo wrapper would not be a
meaningful port.

Permutation entropy supports 1D input through order 20, 2D input for orders 3
and 4, multiple delays, stable tie handling, and normalization. Approximate and
sample entropy currently support `metric="chebyshev"` and
`metric="euclidean"`; AntroPy's other KD-tree metrics are not implemented.
Petrosian FD, Katz FD, zero crossings, and Hjorth parameters support N-D arrays
and `axis`. The remaining measures accept 1D signals, matching upstream.
Signal kernels convert real inputs to contiguous `float64`; complex values,
extended-precision floats, and integers outside float64's exact range are
rejected instead of silently narrowed. LZ numeric sequences must contain
discrete values representable as `uint32`.

## Install and run

```bash
git clone https://github.com/lee101/mojo-antropy.git
cd mojo-antropy
pixi install
pixi run build
pixi run python -c "import mojo_antropy as ant; print(ant.perm_entropy([1, 2, 3, 4]))"
```

`pixi run build` compiles the single Mojo compilation unit into
`dist/libmojo-antropy.so`. The Python wrapper also builds it on first import if
the library is absent. Development checks are `pixi run test` and
`pixi run bench`.

## Correctness

The test suite compares every covered function against the real `antropy`
0.2.2 package on random, periodic, monotonic, tied, multidimensional, and
explicit-tolerance inputs. It also exercises both upstream sample-entropy
branches around the 5,000-sample Numba/KD-tree cutoff, the sample-entropy
parallel cutoff and SIMD remainder, the packed-binary LZ path, and the Higuchi
SIMD remainder path. Run the 118 parity, validation, and behavior tests with
`pixi run test`; the 14 GPU parity tests skip when no device is usable.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz
(Linux x86-64). Each implementation is warmed first and the table reports the
best of three runs on identical contiguous inputs.

| measure | mojo-antropy | antropy 0.2.2 | result |
| --- | ---: | ---: | ---: |
| `perm_entropy` order=3 (1M) | 5.94 ms | 56.74 ms | 9.55x faster |
| `sample_entropy` order=2 (4k) | 4.12 ms | 36.31 ms | 8.82x faster |
| `lziv_complexity` binary (50k) | 192.24 ms | 776.77 ms | 4.04x faster |
| `num_zerocross` (1000x10k) | 10.96 ms | 32.99 ms | 3.01x faster |
| `hjorth_params` (1000x10k) | 88.45 ms | 2969.70 ms | 33.58x faster |
| `petrosian_fd` (1000x10k) | 13.55 ms | 744.28 ms | 54.92x faster |
| `katz_fd` (1000x10k) | 23.43 ms | 2096.35 ms | 89.48x faster |
| `higuchi_fd` kmax=10 (500k) | 3.67 ms | 8.11 ms | 2.21x faster |
| `detrended_fluctuation` (500k) | 91.13 ms | 329.81 ms | 3.62x faster |

The order-2 sample-entropy path compares candidate templates in float64 SIMD
lanes, reduces match masks into counters, and handles the remainder with a
scalar tail. Signals of at least 3,000 samples are split into eight independent
native ranges; smaller signals remain serial to avoid thread-launch overhead.
Binary LZ inputs of at least 1,024 symbols are bit-packed and compared through
precomputed rolling 64-bit windows, while other alphabets use the general
serial parser. The order-3 permutation and Higuchi paths retain their existing
fixed-counter and SIMD implementations.

## GPU

The GPU path covers `app_entropy` only. It is the one measure in the
[analysis](bench/analysis.md) that clears the 2 flops/byte gate: the match loop
is O(n²) over an n-sample signal, so at n = 10k it performs roughly 5×10⁸
comparisons over an 80 KB signal. Every other entry point streams its input once
or a few times per sample (0.12–1.2 flops/byte in the analysis table) and stays
on the CPU.

The path is opt-in and the CPU path remains the default and the fallback:

```python
import numpy as np
import mojo_antropy as ant

ant.gpu(True)          # opt in; calls below the threshold still use the CPU
print(ant.app_entropy(np.random.default_rng(0).normal(size=20_000)))
ant.gpu(False)         # back to CPU-only
```

One block is launched per template row, with the block's lanes splitting that
row's columns and keeping partial match counts. The signal is copied to the
device once, the kernel is launched once per `φ` phase, and the partial counts
are copied back once. The per-row reduction and the log sum then run on the
host in the same order as the CPU path, so the two agree exactly rather than to
a tolerance. Grids are sized to `vectors` blocks, and lanes walk
`j = lane; j < vectors; j += lanes`, so the column tail is covered explicitly;
the tests use sizes such as 6001 and 6143 that are not multiples of the 256
lanes. Device allocations are bounded below 1.5 GB by halving the lane count
for very large inputs.

### Measured before and after

`pixi run` on the host in this repository: Intel Xeon E5-2697 v4 (72 threads)
and a shared NVIDIA RTX 5090 that was running other production workloads
throughout (57–100% utilisation, ~26 GB already resident). Best of five warmed
runs, identical contiguous inputs, `float64`. "Before" is the CPU path shipped
in the previous phase; "after" is the GPU path.

| n (samples) | CPU, 1 worker | CPU, 4 workers | GPU | GPU vs CPU 4 workers |
| ---: | ---: | ---: | ---: | ---: |
| 5,000 | 172.58 ms | 65.76 ms | 67.76 ms | 0.97x (loses) |
| 6,000 | 248.61 ms | 87.10 ms | **6.85 ms** | **12.71x faster** |
| 8,000 | 463.69 ms | 157.59 ms | **11.23 ms** | **14.04x faster** |
| 16,000 | 1872.79 ms | 499.78 ms | **38.07 ms** | **13.13x faster** |
| 32,000 | 7462.16 ms | 1849.52 ms | **73.43 ms** | **25.19x faster** |
| 64,000 | 30052.66 ms | 7089.43 ms | **209.12 ms** | **33.90x faster** |

The crossover is between 5,000 and 6,000 samples, which is why the native
threshold is 6,000. Below it the transfer and launch cost is not repaid and the
GPU is marginally slower, so those calls stay on the CPU and the README does
not claim a win there. Real callers of an O(n²) estimator do pass this size:
the analysis flags `app_entropy` at 5k samples as its top regression (4.8x
slower than upstream), and the 8k–64k range is where the GPU is now 13x–34x
faster than the four-worker CPU path.

The GPU was under heavy contention from other workloads for every measurement
above, so these are conservative: an idle device would show a different, almost
certainly better, ratio. They were not adjusted to look better.

### Tests and skipped cases

`tests/test_gpu.py` runs every covered size through the CPU serial, CPU
parallel, and GPU paths and requires exact equality, plus the Euclidean metric,
orders 1–3, the below-threshold CPU-only case, and the opt-in switch. On a host
with no usable device the GPU tests skip with a reason naming that cause, and
`pixi run test` still passes. That skip path was exercised on this machine with
the device hidden from CUDA: 14 tests **skipped**, 2 passed. There is no
CPU-only machine available here, so the tests have not been run on real
GPU-less hardware.

The device query (`ant.gpu_available()`) constructs a real device context, so it
is intended for a known-good host. On the pinned toolchain
(`mojo ==1.2.0.dev2026092605`), a driver that is present but hides every device
(for example `CUDA_VISIBLE_DEVICES=""`) aborts the process rather than raising.
`app_entropy` is not exposed to that, because it only enters the GPU path when
`gpu(True)` has been set explicitly; with the switch off it never touches the
device. The in-kernel `except` fallback covers ordinary device errors, but it
cannot catch that process abort.

## How it works

Python validates AntroPy-compatible arguments, moves the requested axis to the
end, and converts input once to C-contiguous float64. It owns result and
scratch arrays. A ctypes call passes each buffer as an integer address along
with its dimensions; the exported `abi("C")` Mojo function reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` inside the library.

Signals are row-major, with time along the last dimension. Batched kernels walk
each row without Python callbacks. Mojo performs the ordinal-pattern hash
table, template matching, Lempel-Ziv parsing, derivatives, reductions, fractal
curve traversal, and DFA detrending. No Mojo allocation or ownership crosses
the FFI boundary.

## License

MIT
