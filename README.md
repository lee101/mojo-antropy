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
SIMD remainder path. Run the 79 parity, validation, and behavior tests with
`pixi run test`.

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

No GPU path is provided. Sample matching performs fewer than roughly two
arithmetic operations per byte loaded, and LZ parsing is branch-heavy and
data-dependent, so neither target has enough arithmetic intensity to repay
device transfer and launch costs.

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
