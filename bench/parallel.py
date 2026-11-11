"""Parallel-path measurements for mojo-antropy.

Every row is the same input measured through the public API with the worker
count pinned: 1 is the serial path, 2/3/4 are the parallel path at those worker
counts, and 8 checks that a request above the host cap (4) is clamped rather
than honoured.

Each configuration is timed in its own subprocess, for two reasons: the host is
shared, so a fresh process avoids one measurement inheriting another's thread
pool warm-up, and it keeps memory out of the timing. Times are the best of
`repeat` warmed runs, in milliseconds.

Run with `pixi run bench-parallel`. The table is what the README quotes; the
thresholds in src/antropy.mojo are set from it plus a margin.
"""

from __future__ import annotations

import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import mojo_antropy as mant
from mojo_antropy._lib import parallel_cap


ROOT = Path(__file__).resolve().parent.parent
WORKERS = (1, 2, 3, 4, 8)


def run_one(kind: str, workers: int, size: int, repeat: int) -> float:
    """Best-of-`repeat` milliseconds for one pinned configuration."""
    rng = np.random.default_rng(5)
    if kind == "hjorth":
        data = np.ascontiguousarray(rng.normal(size=(size, 10_000)))
        fn = lambda: mant.hjorth_params(data)  # noqa: E731
    elif kind == "app":
        data = np.ascontiguousarray(rng.normal(size=size))
        fn = lambda: mant.app_entropy(data)  # noqa: E731
    elif kind == "dfa":
        data = np.ascontiguousarray(rng.normal(size=size))
        fn = lambda: mant.detrended_fluctuation(data)  # noqa: E731
    elif kind == "zerocross":
        data = np.ascontiguousarray(rng.normal(size=(size, 10_000)))
        fn = lambda: mant.num_zerocross(data)  # noqa: E731
    else:  # pragma: no cover - guard for typos in table definitions
        raise SystemExit(f"unknown benchmark kind {kind!r}")
    fn()
    mant.parallel_workers(workers)
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best * 1e3


def main() -> None:
    if len(sys.argv) == 5:
        # Internal: one configuration, printed raw.
        print(f"{run_one(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])):.2f}")
        return

    print(f"worker cap: {parallel_cap()}\n")
    table = [
        ("`hjorth_params`", "hjorth", (128, 256, 512, 1000), "rows x 10k", 3),
        ("`app_entropy`", "app", (1000, 1500, 2000, 4000, 8000), "samples", 2),
        ("`detrended_fluctuation`", "dfa", (20_000, 50_000, 64_000, 100_000, 500_000), "samples", 3),
        ("`num_zerocross` (serial only)", "zerocross", (128, 512, 1000), "rows x 10k", 3),
    ]
    for label, kind, sizes, unit, repeat in table:
        print(f"### {label} ({unit})\n")
        header = " | ".join(
            f"{w} worker{'s' if w > 1 else ''}" + (" above cap" if w > parallel_cap() else "")
            for w in WORKERS
        )
        print(f"| input | {header} |")
        print("| --- | " + " | ".join("---:" for _ in WORKERS) + " |")
        for size in sizes:
            cells = []
            for workers in WORKERS:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        kind,
                        str(workers),
                        str(size),
                        str(repeat),
                    ],
                    capture_output=True,
                    text=True,
                    cwd=ROOT,
                    timeout=1800,
                )
                if proc.returncode != 0:
                    cells.append("crashed")
                else:
                    cells.append(f"{float(proc.stdout.strip()):.2f} ms")
            print(f"| {size} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
