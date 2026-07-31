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
    "ma_perm_entropy": ([I] * 11, None),
    "ma_perm_entropy_order3": ([I] * 6, None),
    "ma_app_entropy": ([I, I, I, F, I], F),
    "ma_sample_entropy": ([I, I, I, F, I], F),
    "ma_lziv_complexity": ([I, I], I),
    "ma_num_zerocross": ([I, I, I, I], None),
    "ma_hjorth_params": ([I, I, I, I, I, F], None),
    "ma_petrosian_fd": ([I, I, I, I], None),
    "ma_katz_fd": ([I, I, I, I], None),
    "ma_higuchi_fd": ([I, I, I], F),
    "ma_detrended_fluctuation": ([I, I, I], F),
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
