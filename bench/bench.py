from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import antropy as ant  # noqa: E402
import mojo_antropy as mant  # noqa: E402


def timeit(fn, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def benchmark(name, ours, theirs, repeat=3):
    ours()
    theirs()
    mojo_time = timeit(ours, repeat)
    upstream_time = timeit(theirs, repeat)
    return name, mojo_time, upstream_time


def main():
    rng = np.random.default_rng(42)
    rows = np.ascontiguousarray(rng.normal(size=(1000, 10_000)))
    permutation_data = np.ascontiguousarray(rng.normal(size=1_000_000))
    sample_data = np.ascontiguousarray(rng.normal(size=4_000))
    fractal_data = np.ascontiguousarray(rng.normal(size=500_000))
    binary = rng.integers(0, 2, size=50_000, dtype=np.uint32)

    cases = [
        (
            "perm_entropy order=3 (1M)",
            lambda: mant.perm_entropy(permutation_data),
            lambda: ant.perm_entropy(permutation_data),
            3,
        ),
        (
            "sample_entropy order=2 (4k)",
            lambda: mant.sample_entropy(sample_data),
            lambda: ant.sample_entropy(sample_data),
            3,
        ),
        (
            "lziv_complexity binary (50k)",
            lambda: mant.lziv_complexity(binary),
            lambda: ant.lziv_complexity(binary),
            3,
        ),
        (
            "num_zerocross (1000x10k)",
            lambda: mant.num_zerocross(rows),
            lambda: ant.num_zerocross(rows),
            3,
        ),
        (
            "hjorth_params (1000x10k)",
            lambda: mant.hjorth_params(rows),
            lambda: ant.hjorth_params(rows),
            3,
        ),
        (
            "petrosian_fd (1000x10k)",
            lambda: mant.petrosian_fd(rows),
            lambda: ant.petrosian_fd(rows),
            3,
        ),
        (
            "katz_fd (1000x10k)",
            lambda: mant.katz_fd(rows),
            lambda: ant.katz_fd(rows),
            3,
        ),
        (
            "higuchi_fd kmax=10 (500k)",
            lambda: mant.higuchi_fd(fractal_data),
            lambda: ant.higuchi_fd(fractal_data),
            3,
        ),
        (
            "detrended_fluctuation (500k)",
            lambda: mant.detrended_fluctuation(fractal_data),
            lambda: ant.detrended_fluctuation(fractal_data),
            3,
        ),
    ]

    results = [benchmark(*case) for case in cases]
    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print()
    print("| measure | mojo-antropy | antropy 0.2.2 | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_time, upstream_time in results:
        ratio = upstream_time / mojo_time
        label = (
            f"{ratio:.2f}x faster"
            if ratio >= 1.0
            else f"{1.0 / ratio:.2f}x slower"
        )
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | {label} |"
        )


if __name__ == "__main__":
    main()
