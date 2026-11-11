from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB_PATH = os.path.join(ROOT, "dist", "libmojo-antropy.so")
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "ma_parallel_cap": ([], I),
    "ma_parallel_capacity": ([I], I),
    "ma_parallel_workers": ([I], None),
    "ma_perm_entropy": ([I] * 11, None),
    "ma_perm_entropy_order3": ([I] * 6, None),
    "ma_app_entropy": ([I, I, I, I, F, I, I], F),
    "ma_sample_entropy": ([I, I, I, F, I], F),
    "ma_sample_entropy_order2_counts": ([I, I, I, F, I, I, I], None),
    "ma_lziv_complexity": ([I, I], I),
    "ma_lziv_complexity_packed_binary": ([I, I, I], I),
    "ma_num_zerocross": ([I, I, I, I], None),
    "ma_hjorth_params": ([I, I, I, I, I, F, I], None),
    "ma_petrosian_fd": ([I, I, I, I], None),
    "ma_katz_fd": ([I, I, I, I], None),
    "ma_higuchi_fd": ([I, I, I], F),
    "ma_detrended_fluctuation": ([I, I, I, I], F),
}

_lib: ctypes.CDLL | None = None


def build() -> str:
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB_PATH


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not os.path.exists(LIB_PATH):
            build()
        _lib = ctypes.CDLL(LIB_PATH)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


# Default 1 keeps every kernel on the serial path until a caller opts in, so
# nothing gets slower just because a module was imported. ``None`` resets to it.
_WORKER_SLOT = np.ones(1, dtype=np.uint64)


def parallel_workers(count: int | None = None) -> None:
    """Pin the worker count for parallel kernels, or None for automatic.

    ``1`` forces every kernel down its serial path, which makes a serial run
    and a parallel run comparable on identical input. Values above the native
    cap are clamped there. The worker count lives in a UInt64 owned by this
    module: Mojo rejects mutable globals, so the library is handed its address
    rather than keeping its own knob.
    """
    if count is not None and count < 0:
        raise ValueError("worker count must be non-negative.")
    _WORKER_SLOT[0] = 1 if count is None else int(count)
    lib().ma_parallel_workers(addr(_WORKER_SLOT))


def parallel_cap() -> int:
    """Native worker ceiling; requests above it are clamped there."""
    return lib().ma_parallel_cap()


def parallel_capacity(size: int) -> int:
    """Int64 slots needed for a parallel scratch buffer covering ``size``."""
    return lib().ma_parallel_capacity(int(size))


def workers() -> int:
    """Address of the live worker knob, passed to parallel-capable kernels."""
    return addr(_WORKER_SLOT)


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays.")
    if array.size == 0:
        raise ValueError("FFI buffers must not be empty.")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous.")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("FFI buffers must have a non-null address.")
    return address


def f64(array) -> np.ndarray:
    source = np.asarray(array)
    if source.dtype.kind == "c":
        raise TypeError("complex-valued signals are not supported.")
    if source.dtype.kind in "iu" and source.size:
        limit = 2**53
        if np.any(source > limit) or np.any(source < -limit):
            raise ValueError("integer signal values must be exactly representable as float64.")
    if source.dtype.kind == "f" and source.dtype.itemsize > 8:
        raise TypeError("floating-point signal dtype must be float64 or narrower.")
    return np.ascontiguousarray(source, dtype=np.float64)
