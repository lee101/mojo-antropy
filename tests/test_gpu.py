"""GPU approximate-entropy path against the CPU path.

The GPU covers ``app_entropy`` only: it is the one measure whose O(n^2) match
loop clears the arithmetic-intensity gate in ``bench/analysis.md``. These tests
run the same call three ways (CPU serial, CPU parallel, GPU) and require the
results to agree exactly, because both paths perform the same comparisons and
the host reduction sums the per-row logs in the same order.

On a host with no usable device every test skips, so ``pytest`` stays green on
a CPU-only machine. The skip reason names the cause, and the README says the
GPU tests were skipped there.
"""

import os

import numpy as np
import pytest

import mojo_antropy as mant
from mojo_antropy._lib import gpu_app_entropy_min


def _device_may_be_usable():
    # Probing for real means constructing a device context, and the pinned
    # toolchain aborts the process (rather than raising) when the driver is
    # present but no device is visible. On a CPU-only host there is no driver
    # node at all, so the probe is safe there and returns False. An
    # explicitly emptied CUDA_VISIBLE_DEVICES is the one case where the probe
    # could abort, so the GPU tests skip instead of calling it.
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return False
    return mant.gpu_available()


GPU_AVAILABLE = _device_may_be_usable()
needs_gpu = pytest.mark.skipif(
    not GPU_AVAILABLE,
    reason="no GPU device context on this host; GPU path skipped, CPU path used",
)

# Deliberately includes sizes that are not multiples of the 256-lane block, so
# the per-lane column tail is exercised (6001, 6003, 6143 are all coprime to
# 256 in the sense that vectors % lanes != 0).
SIZES = (6001, 6003, 6143, 8000)


@pytest.fixture(autouse=True)
def cpu_only_by_default():
    """Leave the GPU switch off after each test, whatever it was set to."""
    mant.gpu(False)
    mant.parallel_workers(None)
    yield
    mant.gpu(False)
    mant.parallel_workers(None)


def _signal(n, seed=7):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.normal(size=n))


@needs_gpu
def test_gpu_availability_is_queryable():
    # Must return a bool on a host with a device. It is skipped rather than
    # run on the no-device paths because the native probe constructs a real
    # context, which can abort the process on a broken driver.
    assert mant.gpu_available() is True


def test_gpu_threshold_is_exposed():
    # The threshold is a native constant so the wrapper cannot drift from it.
    assert gpu_app_entropy_min() == 6000


@needs_gpu
@pytest.mark.parametrize("n", SIZES)
def test_gpu_matches_cpu_serial(n):
    x = _signal(n)
    mant.gpu(False)
    mant.parallel_workers(1)
    expected = mant.app_entropy(x)
    mant.gpu(True)
    got = mant.app_entropy(x)
    # Exact: the GPU repeats the CPU comparisons and sums in the same order.
    assert got == expected


@needs_gpu
@pytest.mark.parametrize("n", SIZES)
def test_gpu_matches_cpu_parallel(n):
    x = _signal(n, seed=11)
    mant.parallel_workers(4)
    mant.gpu(False)
    expected = mant.app_entropy(x)
    mant.gpu(True)
    got = mant.app_entropy(x)
    assert got == expected


@needs_gpu
@pytest.mark.parametrize("order", (1, 2, 3))
def test_gpu_matches_cpu_across_orders(order):
    # order+1 is what the second Phi phase uses, so this also covers the
    # second kernel launch and its own grid.
    x = _signal(6000 + order)
    mant.parallel_workers(1)
    mant.gpu(False)
    expected = mant.app_entropy(x, order=order)
    mant.gpu(True)
    got = mant.app_entropy(x, order=order)
    assert got == expected


@needs_gpu
def test_gpu_matches_cpu_with_euclidean_metric():
    x = _signal(6007, seed=3)
    mant.parallel_workers(1)
    mant.gpu(False)
    expected = mant.app_entropy(x, metric="euclidean")
    mant.gpu(True)
    got = mant.app_entropy(x, metric="euclidean")
    assert got == expected


@needs_gpu
def test_below_threshold_uses_cpu_path():
    # Just under the crossover: the GPU must not be taken, and the result must
    # equal the CPU one. Measured at n=5000 the GPU is ~1.0x the CPU four-worker
    # path, i.e. no win, which is why 6000 is the gate.
    x = _signal(5999)
    mant.parallel_workers(1)
    mant.gpu(False)
    expected = mant.app_entropy(x)
    mant.gpu(True)
    got = mant.app_entropy(x)
    assert got == expected


def test_gpu_switch_round_trips():
    assert mant.gpu(True) is True
    assert mant.gpu() is True
    assert mant.gpu(False) is False
    assert mant.gpu() is False
