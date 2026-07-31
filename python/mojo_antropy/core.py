from __future__ import annotations

from math import factorial, log

import numpy as np

from ._lib import addr, f64, lib


def _integral(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer.")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        raise TypeError(f"{name} must be an integer.") from None
    if converted != value:
        raise ValueError(f"{name} must be an integer.")
    return converted


def _rows_last(x, axis: int):
    array = np.asarray(x)
    moved = np.moveaxis(array, axis, -1)
    shape = moved.shape[:-1]
    rows = f64(moved).reshape(-1, moved.shape[-1])
    return rows, shape


def _finish(values: np.ndarray, shape):
    result = values.reshape(shape)
    return result.item() if result.ndim == 0 else result


def perm_entropy(x, order=3, delay=1, normalize=False):
    if isinstance(delay, (list, np.ndarray, range)):
        return np.mean(
            [perm_entropy(x, order=order, delay=d, normalize=normalize) for d in delay],
            axis=0,
        )
    array = np.asarray(x)
    if array.ndim not in (1, 2):
        raise ValueError("x must be 1D or 2D.")
    order = _integral(order, "order")
    delay = _integral(delay, "delay")
    if order < 2:
        raise ValueError("Order has to be at least 2.")
    if delay < 1:
        raise ValueError("delay must be greater than zero.")
    n = array.shape[-1]
    windows = n - (order - 1) * delay
    if windows <= 0:
        raise ValueError("The signal is too short for the given order and delay.")
    if array.ndim == 2 and order not in (3, 4):
        raise ValueError("2D input is only supported for order=3 and order=4.")
    if order > 20:
        raise ValueError("order must be 20 or less.")

    rows = f64(array).reshape(-1, n)
    result = np.empty(rows.shape[0], dtype=np.float64)
    if order == 3:
        lib().ma_perm_entropy_order3(
            addr(rows),
            addr(result),
            rows.shape[0],
            n,
            delay,
            int(bool(normalize)),
        )
        return result.item() if array.ndim == 1 else result

    capacity = 1
    while capacity < 2 * windows:
        capacity *= 2
    keys = np.empty(capacity, dtype=np.int64)
    counts = np.empty(capacity, dtype=np.int64)
    work = np.empty(order, dtype=np.int64)
    lib().ma_perm_entropy(
        addr(rows),
        addr(result),
        rows.shape[0],
        n,
        order,
        delay,
        int(bool(normalize)),
        addr(keys),
        addr(counts),
        addr(work),
        capacity,
    )
    return result.item() if array.ndim == 1 else result


def _metric_code(metric: str) -> int:
    if metric == "chebyshev":
        return 0
    if metric == "euclidean":
        return 1
    raise ValueError("mojo-antropy supports metric='chebyshev' or 'euclidean'.")


def _entropy_input(x, order, tolerance):
    array = f64(x)
    if array.ndim != 1:
        raise ValueError("x must be one-dimensional.")
    order = _integral(order, "order")
    if order < 1 or array.size <= order:
        raise ValueError("order must be positive and smaller than the signal.")
    if tolerance is None:
        radius = 0.2 * np.std(array, ddof=0)
    else:
        if not isinstance(tolerance, (float, int)):
            raise TypeError(
                "tolerance must be a float or int, got %s."
                % type(tolerance).__name__
            )
        radius = float(tolerance)
    return array, order, radius


def app_entropy(x, order=2, tolerance=None, metric="chebyshev"):
    array, order, radius = _entropy_input(x, order, tolerance)
    return lib().ma_app_entropy(
        addr(array), array.size, order, radius, _metric_code(metric)
    )


def sample_entropy(x, order=2, tolerance=None, metric="chebyshev"):
    array, order, radius = _entropy_input(x, order, tolerance)
    return lib().ma_sample_entropy(
        addr(array), array.size, order, radius, _metric_code(metric)
    )


def _sequence_u32(sequence) -> np.ndarray:
    if not isinstance(sequence, (str, list, np.ndarray)):
        raise TypeError("sequence must be a str, list, or np.ndarray.")
    if isinstance(sequence, (list, np.ndarray)):
        sequence = np.asarray(sequence)
        if sequence.dtype.kind == "b":
            return np.ascontiguousarray(sequence, dtype=np.uint32)
        if sequence.dtype.kind in "iu":
            if sequence.size and (
                np.any(sequence < 0) or np.any(sequence > np.iinfo(np.uint32).max)
            ):
                raise ValueError("integer sequence values must fit in uint32.")
            return np.ascontiguousarray(sequence, dtype=np.uint32)
        if sequence.dtype.kind == "f":
            if sequence.size and (
                not np.all(np.isfinite(sequence))
                or np.any(sequence != np.floor(sequence))
                or np.any(sequence < 0)
                or np.any(sequence > np.iinfo(np.uint32).max)
            ):
                raise ValueError(
                    "float sequence values must be finite uint32-valued integers."
                )
            return np.ascontiguousarray(sequence, dtype=np.uint32)
        return np.fromiter(
            map(ord, "".join(sequence.astype(str))), dtype=np.uint32
        )
    return np.fromiter(map(ord, sequence), dtype=np.uint32)


def lziv_complexity(sequence, normalize=False):
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a bool.")
    encoded = _sequence_u32(sequence)
    if encoded.size == 0:
        if normalize:
            raise ValueError("cannot normalize an empty sequence.")
        return 1
    complexity = lib().ma_lziv_complexity(addr(encoded), encoded.size)
    if not normalize:
        return complexity
    n = encoded.size
    base = max(2, np.unique(encoded).size)
    return complexity / (n / log(n, base))


def num_zerocross(x, normalize=False, axis=-1):
    rows, shape = _rows_last(x, axis)
    if rows.shape[1] == 0:
        raise ValueError("x must contain at least one sample along axis.")
    if rows.shape[0] == 0:
        return _finish(np.empty(0, dtype=np.float64 if normalize else np.int64), shape)
    result = np.empty(rows.shape[0], dtype=np.int64)
    lib().ma_num_zerocross(addr(rows), addr(result), rows.shape[0], rows.shape[1])
    if normalize:
        return _finish(result.astype(np.float64) / rows.shape[1], shape)
    return _finish(result, shape)


def hjorth_params(x, sf=None, axis=-1):
    if sf is not None and not isinstance(sf, (int, float)):
        raise TypeError(
            "sf must be a numeric value (int or float), got %s."
            % type(sf).__name__
        )
    rows, shape = _rows_last(x, axis)
    if rows.shape[1] < 3:
        raise ValueError("x must contain at least three samples along axis.")
    if rows.shape[0] == 0:
        empty = np.empty(0, dtype=np.float64)
        return _finish(empty, shape), _finish(empty.copy(), shape)
    mobility = np.empty(rows.shape[0], dtype=np.float64)
    complexity = np.empty(rows.shape[0], dtype=np.float64)
    lib().ma_hjorth_params(
        addr(rows),
        addr(mobility),
        addr(complexity),
        rows.shape[0],
        rows.shape[1],
        1.0 if sf is None else float(sf),
    )
    return _finish(mobility, shape), _finish(complexity, shape)


def petrosian_fd(x, axis=-1):
    rows, shape = _rows_last(x, axis)
    if rows.shape[1] == 0:
        raise ValueError("x must contain at least one sample along axis.")
    if rows.shape[0] == 0:
        return _finish(np.empty(0, dtype=np.float64), shape)
    result = np.empty(rows.shape[0], dtype=np.float64)
    lib().ma_petrosian_fd(addr(rows), addr(result), rows.shape[0], rows.shape[1])
    return _finish(result, shape)


def katz_fd(x, axis=-1):
    rows, shape = _rows_last(x, axis)
    if rows.shape[1] < 2:
        raise ValueError("x must contain at least two samples along axis.")
    if rows.shape[0] == 0:
        return _finish(np.empty(0, dtype=np.float64), shape)
    result = np.empty(rows.shape[0], dtype=np.float64)
    lib().ma_katz_fd(addr(rows), addr(result), rows.shape[0], rows.shape[1])
    return _finish(result, shape)


def higuchi_fd(x, kmax=10):
    array = f64(x)
    if array.ndim != 1:
        raise ValueError("x must be one-dimensional.")
    kmax = _integral(kmax, "kmax")
    if kmax < 2 or kmax >= array.size:
        raise ValueError("kmax must be at least 2 and smaller than the signal.")
    return lib().ma_higuchi_fd(addr(array), array.size, kmax)


def detrended_fluctuation(x):
    array = f64(x)
    if array.ndim != 1:
        raise ValueError("x must be one-dimensional.")
    if array.size == 0:
        raise ValueError("x must contain at least one sample.")
    walk = np.empty_like(array)
    return lib().ma_detrended_fluctuation(addr(array), addr(walk), array.size)
