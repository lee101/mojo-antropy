import numpy as np
import pytest

ant = pytest.importorskip("antropy")
import mojo_antropy as mant
from mojo_antropy._lib import lib


@pytest.fixture(scope="module")
def signals():
    rng = np.random.default_rng(42)
    random = rng.normal(size=1200)
    t = np.arange(1200)
    sine = np.sin(2 * np.pi * t / 100)
    trend = np.arange(1200, dtype=np.float64)
    return random, sine, trend


@pytest.mark.parametrize(
    "order,delay,normalize",
    [(2, 1, False), (3, 1, True), (4, 3, False), (5, 2, True), (7, 1, False)],
)
def test_perm_entropy_parity(signals, order, delay, normalize):
    x = signals[0]
    assert mant.perm_entropy(
        x, order=order, delay=delay, normalize=normalize
    ) == pytest.approx(
        ant.perm_entropy(x, order=order, delay=delay, normalize=normalize),
        abs=5e-9,
    )


def test_perm_entropy_ties_match_stable_argsort():
    x = np.array([0, 0, 1, 1, 0, 2, 2, 1, 0, 0, 3, 3], dtype=np.int64)
    for order in (3, 4, 5):
        assert mant.perm_entropy(x, order=order) == pytest.approx(
            ant.perm_entropy(x, order=order), abs=5e-9
        )


def test_perm_entropy_multiple_delays(signals):
    delays = [1, 2, 4]
    ours = mant.perm_entropy(signals[0], delay=delays, normalize=True)
    theirs = ant.perm_entropy(signals[0], delay=delays, normalize=True)
    assert ours == pytest.approx(theirs, abs=5e-9)


@pytest.mark.parametrize("order", [3, 4])
def test_perm_entropy_2d(order):
    rng = np.random.default_rng(8)
    x = rng.integers(0, 10, size=(5, 600))
    assert np.allclose(
        mant.perm_entropy(x, order=order, normalize=True),
        ant.perm_entropy(x, order=order, normalize=True),
        atol=5e-9,
    )


@pytest.mark.parametrize("metric", ["chebyshev", "euclidean"])
@pytest.mark.parametrize("order", [2, 3])
def test_app_entropy_parity(signals, metric, order):
    x = signals[0][:700]
    assert mant.app_entropy(x, order=order, metric=metric) == pytest.approx(
        ant.app_entropy(x, order=order, metric=metric), abs=5e-9
    )


@pytest.mark.parametrize("metric", ["chebyshev", "euclidean"])
@pytest.mark.parametrize("order", [2, 3])
def test_sample_entropy_parity(signals, metric, order):
    x = signals[0][:700]
    assert mant.sample_entropy(x, order=order, metric=metric) == pytest.approx(
        ant.sample_entropy(x, order=order, metric=metric), abs=5e-9
    )


def test_entropy_explicit_tolerance(signals):
    x = signals[0][:500]
    for name in ("app_entropy", "sample_entropy"):
        ours = getattr(mant, name)(x, order=2, tolerance=0.35)
        theirs = getattr(ant, name)(x, order=2, tolerance=0.35)
        assert ours == pytest.approx(theirs, abs=5e-9)


def test_sample_entropy_kdtree_size_branch():
    rng = np.random.default_rng(99)
    x = rng.normal(size=5001)
    assert mant.sample_entropy(x) == pytest.approx(
        ant.sample_entropy(x), rel=2e-9, abs=2e-9
    )


@pytest.mark.parametrize(
    "sequence",
    [
        "1001111011000010",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "HELLO WORLD! HELLO WORLD! HELLO WORLD!",
        [1, 0, 1, 0, 1, 0, 1, 0],
        np.array([True, False, True, True, False]),
        np.array(["AB", "CD", "AB"]),
    ],
)
def test_lziv_complexity_parity(sequence):
    assert mant.lziv_complexity(sequence) == ant.lziv_complexity(sequence)
    assert mant.lziv_complexity(sequence, normalize=True) == pytest.approx(
        ant.lziv_complexity(sequence, normalize=True), abs=1e-12
    )


@pytest.mark.parametrize("normalize", [False, True])
def test_num_zerocross_1d(signals, normalize):
    for x in signals:
        assert mant.num_zerocross(x, normalize=normalize) == pytest.approx(
            ant.num_zerocross(x, normalize=normalize)
        )


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_num_zerocross_nd(axis):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 40))
    assert np.allclose(
        mant.num_zerocross(x, axis=axis),
        ant.num_zerocross(x, axis=axis),
    )


@pytest.mark.parametrize("sf", [None, 100])
def test_hjorth_1d(signals, sf):
    for x in signals:
        ours = mant.hjorth_params(x, sf=sf)
        with np.errstate(invalid="ignore"):
            theirs = ant.hjorth_params(x, sf=sf)
        assert np.allclose(
            ours, theirs, rtol=2e-12, atol=2e-12, equal_nan=True
        )


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_hjorth_nd(axis):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(30, 40))
    ours = mant.hjorth_params(x, axis=axis)
    theirs = ant.hjorth_params(x, axis=axis)
    assert np.allclose(ours[0], theirs[0], rtol=2e-12)
    assert np.allclose(ours[1], theirs[1], rtol=2e-12)


@pytest.mark.parametrize("name", ["petrosian_fd", "katz_fd"])
def test_vectorized_fractal_parity(signals, name):
    fn_ours = getattr(mant, name)
    fn_theirs = getattr(ant, name)
    for x in signals:
        assert fn_ours(x) == pytest.approx(fn_theirs(x), abs=5e-9)
    matrix = np.stack(signals)
    assert np.allclose(fn_ours(matrix), fn_theirs(matrix), atol=5e-9)
    assert np.allclose(
        fn_ours(matrix.T, axis=0), fn_theirs(matrix.T, axis=0), atol=5e-9
    )


@pytest.mark.parametrize("kmax", [5, 10, 20])
def test_higuchi_fd_parity(signals, kmax):
    for x in signals:
        assert mant.higuchi_fd(x, kmax=kmax) == pytest.approx(
            ant.higuchi_fd(x, kmax=kmax), abs=5e-8
        )


def test_higuchi_fd_simd_tail():
    x = np.random.default_rng(123).normal(size=1203)
    assert mant.higuchi_fd(x, kmax=11) == pytest.approx(
        ant.higuchi_fd(x, kmax=11), abs=5e-8
    )


def test_detrended_fluctuation_parity(signals):
    for x in signals:
        assert mant.detrended_fluctuation(x) == pytest.approx(
            ant.detrended_fluctuation(x), abs=5e-8
        )


@pytest.mark.parametrize(
    "call,error",
    [
        (lambda: mant.perm_entropy([1, 2], order=1), ValueError),
        (lambda: mant.perm_entropy([1, 2, 3], delay=0), ValueError),
        (lambda: mant.perm_entropy(np.ones((2, 5)), order=5), ValueError),
        (lambda: mant.app_entropy([1, 2, 3], metric="manhattan"), ValueError),
        (lambda: mant.higuchi_fd([1, 2, 3], kmax=3), ValueError),
        (lambda: mant.lziv_complexity("101", normalize=1), TypeError),
    ],
)
def test_validation(call, error):
    with pytest.raises(error):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: mant.perm_entropy([1, 2, 3], order=2.5),
        lambda: mant.perm_entropy([1, 2, 3], delay=1.5),
        lambda: mant.higuchi_fd(np.arange(10), kmax=2.5),
        lambda: mant.num_zerocross([]),
        lambda: mant.hjorth_params([1, 2]),
        lambda: mant.petrosian_fd([]),
        lambda: mant.katz_fd([1]),
        lambda: mant.detrended_fluctuation([]),
    ],
)
def test_unsafe_or_narrowing_inputs_are_rejected(call):
    with pytest.raises((TypeError, ValueError)):
        call()


def test_empty_lziv_complexity_does_not_cross_ffi():
    assert mant.lziv_complexity("") == ant.lziv_complexity("")
    with pytest.raises(ValueError):
        mant.lziv_complexity("", normalize=True)


@pytest.mark.parametrize(
    "call",
    [
        lambda: mant.num_zerocross(np.array([1 + 2j])),
        lambda: mant.num_zerocross(np.array([2**53 + 1], dtype=np.uint64)),
        lambda: mant.lziv_complexity([-1, 0, 1]),
        lambda: mant.lziv_complexity([0.5, 1.0]),
        lambda: mant.lziv_complexity([2**32]),
    ],
)
def test_lossy_dtype_narrowing_is_rejected(call):
    with pytest.raises((TypeError, ValueError)):
        call()


@pytest.mark.parametrize(
    "name", ["num_zerocross", "hjorth_params", "petrosian_fd", "katz_fd"]
)
def test_empty_batch_does_not_cross_ffi(name):
    x = np.empty((0, 8))
    ours = getattr(mant, name)(x)
    theirs = getattr(ant, name)(x)
    if isinstance(ours, tuple):
        assert all(part.shape == (0,) for part in ours)
        assert all(part.shape == (0,) for part in theirs)
    else:
        assert ours.shape == theirs.shape == (0,)


@pytest.mark.parametrize(
    "name", ["num_zerocross", "hjorth_params", "petrosian_fd", "katz_fd"]
)
def test_nd_axis_parity(name):
    x = np.random.default_rng(71).normal(size=(3, 4, 20))
    ours = getattr(mant, name)(x, axis=1)
    theirs = getattr(ant, name)(x, axis=1)
    if isinstance(ours, tuple):
        assert np.allclose(ours[0], theirs[0])
        assert np.allclose(ours[1], theirs[1])
    else:
        assert np.allclose(ours, theirs)


def test_exported_abi_rejects_null_buffers():
    native = lib()
    native.ma_perm_entropy(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    native.ma_perm_entropy_order3(0, 0, 0, 0, 0, 0)
    assert np.isnan(native.ma_app_entropy(0, 0, 0, 0.0, 0))
    assert np.isnan(native.ma_sample_entropy(0, 0, 0, 0.0, 0))
    assert native.ma_lziv_complexity(0, 0) == 1
    native.ma_num_zerocross(0, 0, 0, 0)
    native.ma_hjorth_params(0, 0, 0, 0, 0, 1.0)
    native.ma_petrosian_fd(0, 0, 0, 0)
    native.ma_katz_fd(0, 0, 0, 0)
    assert np.isnan(native.ma_higuchi_fd(0, 0, 0))
    assert np.isnan(native.ma_detrended_fluctuation(0, 0, 0))
