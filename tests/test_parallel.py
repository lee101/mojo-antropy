"""Serial and parallel paths must agree on the same input.

`parallel_workers(1)` forces every kernel down its serial path and
`parallel_workers(None)` restores automatic dispatch, so each test can run the
same call both ways.

Hjorth and random-kernel batches are bit-identical: each row computes the same
expression and no cross-row reduction exists. AppEnumerate is also
bit-identical, because the workers record one exact integer match count per row
and the logs are then summed serially in row order. DFA sums per-window
residuals; each window is computed identically but the serial path adds them
through libm's vectorised partial sums while the parallel path adds the
per-worker partials, so it agrees to within one ulp.
"""

import numpy as np
import pytest

import mojo_antropy as mant
from mojo_antropy._lib import parallel_cap


@pytest.fixture(autouse=True)
def automatic_workers():
    mant.parallel_workers(None)
    yield
    mant.parallel_workers(None)


def serial(fn):
    mant.parallel_workers(1)
    return fn()


def parallel(fn, workers):
    mant.parallel_workers(workers)
    return fn()


def test_worker_cap_is_four():
    assert parallel_cap() == 4


@pytest.mark.parametrize("workers", [2, 4, 8])
def test_hjorth_params_identical_above_threshold(workers):
    x = np.random.default_rng(0).normal(size=(400, 2_000))
    baseline = serial(lambda: mant.hjorth_params(x, sf=100))
    result = parallel(lambda: mant.hjorth_params(x, sf=100), workers)
    assert np.array_equal(baseline[0], result[0])
    assert np.array_equal(baseline[1], result[1])


def test_hjorth_params_identical_below_threshold():
    # Under HJORTH_ROWS_PARALLEL_MIN the call stays serial whatever the knob
    # says, and the values are the ones upstream produces.
    x = np.random.default_rng(1).normal(size=(7, 500))
    auto = mant.hjorth_params(x)
    mant.parallel_workers(4)
    pinned = mant.hjorth_params(x)
    assert np.array_equal(auto[0], pinned[0])
    assert np.array_equal(auto[1], pinned[1])


@pytest.mark.parametrize("workers", [2, 4, 8])
def test_app_entropy_identical_above_threshold(workers):
    x = np.random.default_rng(2).normal(size=2_500)
    baseline = serial(lambda: mant.app_entropy(x, order=2))
    assert parallel(lambda: mant.app_entropy(x, order=2), workers) == baseline
    for order in (3, 4):
        baseline = serial(lambda: mant.app_entropy(x, order=order))
        assert parallel(lambda: mant.app_entropy(x, order=order), workers) == baseline


def test_app_entropy_identical_below_threshold():
    x = np.random.default_rng(3).normal(size=1_000)
    baseline = serial(lambda: mant.app_entropy(x, order=2))
    mant.parallel_workers(4)
    assert mant.app_entropy(x, order=2) == baseline


@pytest.mark.parametrize("workers", [2, 4, 8])
def test_detrended_fluctuation_matches_serial_above_threshold(workers):
    x = np.random.default_rng(4).normal(size=200_000)
    baseline = serial(lambda: mant.detrended_fluctuation(x))
    result = parallel(lambda: mant.detrended_fluctuation(x), workers)
    assert result == pytest.approx(baseline, rel=0, abs=8 * np.spacing(baseline))


def test_detrended_fluctuation_identical_below_threshold():
    # Below DFA_PARALLEL_MIN the call is serial whatever the knob says.
    x = np.random.default_rng(5).normal(size=30_000)
    baseline = serial(lambda: mant.detrended_fluctuation(x))
    mant.parallel_workers(4)
    assert mant.detrended_fluctuation(x) == baseline


@pytest.mark.parametrize(
    "name,extra",
    [
        ("num_zerocross", {}),
        ("petrosian_fd", {}),
        ("katz_fd", {}),
        ("perm_entropy", {"order": 4}),
    ],
)
def test_serial_only_kernels_ignore_the_worker_knob(name, extra):
    # These were measured to lose with workers on the benchmark host, so they
    # must be unchanged by the knob at any size (see README "Parallelism").
    fn = getattr(mant, name)
    x = np.random.default_rng(6).normal(size=(600, 3_000))
    baseline = np.asarray(serial(lambda: fn(x, **extra)))
    for workers in (2, 4, 8):
        mant.parallel_workers(workers)
        assert np.array_equal(baseline, np.asarray(fn(x, **extra)))


@pytest.mark.parametrize("order", [2, 3])
def test_sample_entropy_ignores_the_worker_knob(order):
    x = np.random.default_rng(7).normal(size=1_600)
    baseline = serial(lambda: mant.sample_entropy(x, order=order))
    mant.parallel_workers(4)
    assert mant.sample_entropy(x, order=order) == baseline


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_parallel_paths_match_upstream(workers):
    # Identical serial/parallel output only proves they agree with each other;
    # this checks the shared value against antropy on the parallel sizes.
    ant = pytest.importorskip("antropy")
    rng = np.random.default_rng(8)
    hjorth = rng.normal(size=(300, 1_000))
    app = rng.normal(size=2_200)
    dfa = rng.normal(size=80_000)
    mant.parallel_workers(workers)
    ours = mant.hjorth_params(hjorth)
    theirs = ant.hjorth_params(hjorth)
    assert np.allclose(ours[0], theirs[0], rtol=2e-12, atol=2e-12, equal_nan=True)
    assert np.allclose(ours[1], theirs[1], rtol=2e-12, atol=2e-12, equal_nan=True)
    assert mant.app_entropy(app) == pytest.approx(
        ant.app_entropy(app), abs=5e-9
    )
    assert mant.detrended_fluctuation(dfa) == pytest.approx(
        ant.detrended_fluctuation(dfa), abs=5e-8
    )


def test_parallel_workers_validation():
    mant.parallel_workers(None)
    mant.parallel_workers(0)
    with pytest.raises(ValueError):
        mant.parallel_workers(-1)
