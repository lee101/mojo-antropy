from std.math import abs, floor, log, sqrt
from std.memory import UnsafePointer
from std.bit import count_trailing_zeros
from std.sys import simd_width_of as simdwidthof
from std.collections import List
from max.algorithm import parallelize
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext


comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime UPtr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime LN2 = 0.693147180559945309417232121458
comptime REG_EPS = 1.0e-9

# The worker cap is a host rule: this machine is shared, so four workers is the
# ceiling unless a benchmark shows more helps.
comptime WORKER_CAP = 4

# Crossovers measured on a 72-CPU Xeon E5-2697 v4 with bench/parallel.py and
# spot-checked in repeated runs, then given a margin above the break-even point.
# Only kernels where the benchmark showed a repeatable win are parallel; the
# rest keep the serial loop and say so at the call site.
#
# Hjorth: three passes of arithmetic per row over 10k samples. Four workers were
# 2.2x faster at 256 rows and 3.0x at 1000; break-even is below 128 rows, and
# 256 leaves the margin. Below it the serial pass costs well under a
# millisecond.
comptime HJORTH_ROWS_PARALLEL_MIN = 256
# num_zerocross / petrosian_fd / katz_fd are ~0.1-0.25 flops per byte: the row
# loop is already limited by memory bandwidth, and four workers measured 15-35%
# SLOWER at 512-1000 rows on this host. They stay serial.
#
# AppEnumerate: the O(n^2) match loop needs its outer index split across workers
# to move the needle. Serial and four workers tie at 1000 samples (7.2 vs
# 7.5 ms), four workers win 1.1x at 1500, 1.7x at 2000, 3.2x at 4000 and 3.1x
# at 8000. 2000 is the measured break-even, and the threshold sits there rather
# than above it because the win grows steeply with n and callers rarely pass
# smaller signals to an O(n^2) estimator.
comptime APP_ENTROPY_PARALLEL_MIN = 2000
# DFA: break-even is ~40k samples; below it the run is a handful of
# milliseconds and fan-out costs more than it saves. 64k is the margin.
comptime DFA_PARALLEL_MIN = 64_000

def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def up(addr: Int) -> UPtr:
    return UPtr(unsafe_from_address=addr)


def nan_value() -> Float64:
    return 0.0 / Float64(0)


def infinity() -> Float64:
    return 1.0 / Float64(0)


def ordinal_rank(
    x: FPtr,
    start: Int,
    order: Int,
    delay: Int,
    jitter: Float64,
    work: IPtr,
) -> Int64:
    for i in range(order):
        work[i] = Int64(i)
    for i in range(1, order):
        var j = i
        while j > 0:
            var left = Int(work[j - 1])
            var right = Int(work[j])
            var left_value = x[start + left * delay] + Float64(left) * jitter
            var right_value = x[start + right * delay] + Float64(right) * jitter
            if left_value <= right_value:
                break
            work[j - 1] = Int64(right)
            work[j] = Int64(left)
            j -= 1

    var rank = Int64(0)
    for i in range(order):
        var smaller = Int64(0)
        for j in range(i + 1, order):
            if work[j] < work[i]:
                smaller += 1
        rank = rank * Int64(order - i) + smaller
    return rank


def permutation_entropy_row(
    x: FPtr,
    n: Int,
    order: Int,
    delay: Int,
    normalize: Bool,
    keys: IPtr,
    counts: IPtr,
    work: IPtr,
    capacity: Int,
) -> Float64:
    for i in range(capacity):
        keys[i] = -1
        counts[i] = 0

    var windows = n - (order - 1) * delay
    var jitter = Float64(0)
    if order == 3 or order == 4:
        var magnitude = Float64(0)
        for i in range(n):
            magnitude = max(magnitude, abs(x[i]))
        jitter = 2.220446049250313e-16 * (magnitude + 1.0)
    for start in range(windows):
        var rank = ordinal_rank(x, start, order, delay, jitter, work)
        var slot = Int(rank) & (capacity - 1)
        while keys[slot] != -1 and keys[slot] != rank:
            slot = (slot + 1) & (capacity - 1)
        if keys[slot] == -1:
            keys[slot] = rank
        counts[slot] += 1

    var entropy = Float64(0)
    for i in range(capacity):
        if counts[i] > 0:
            var probability = Float64(counts[i]) / Float64(windows)
            entropy -= probability * log(probability) / LN2
    if normalize:
        var factorial = Float64(1)
        for i in range(2, order + 1):
            factorial *= Float64(i)
        entropy /= log(factorial) / LN2
        entropy = min(max(entropy, 0.0), 1.0)
    return entropy


def permutation_entropy_order3_row(
    x: FPtr,
    n: Int,
    delay: Int,
    normalize: Bool,
) -> Float64:
    var count0 = Int64(0)
    var count1 = Int64(0)
    var count2 = Int64(0)
    var count3 = Int64(0)
    var count4 = Int64(0)
    var count5 = Int64(0)
    var windows = n - 2 * delay
    for start in range(windows):
        var a = x[start]
        var b = x[start + delay]
        var c = x[start + 2 * delay]
        if a <= b:
            if b <= c:
                count0 += 1
            elif a <= c:
                count1 += 1
            else:
                count4 += 1
        elif a <= c:
            count2 += 1
        elif b <= c:
            count3 += 1
        else:
            count5 += 1

    var entropy = Float64(0)
    if count0 > 0:
        var p = Float64(count0) / Float64(windows)
        entropy -= p * log(p) / LN2
    if count1 > 0:
        var p = Float64(count1) / Float64(windows)
        entropy -= p * log(p) / LN2
    if count2 > 0:
        var p = Float64(count2) / Float64(windows)
        entropy -= p * log(p) / LN2
    if count3 > 0:
        var p = Float64(count3) / Float64(windows)
        entropy -= p * log(p) / LN2
    if count4 > 0:
        var p = Float64(count4) / Float64(windows)
        entropy -= p * log(p) / LN2
    if count5 > 0:
        var p = Float64(count5) / Float64(windows)
        entropy -= p * log(p) / LN2
    if normalize:
        entropy /= log(6.0) / LN2
        entropy = min(max(entropy, 0.0), 1.0)
    return entropy


def vectors_match(
    x: FPtr,
    a: Int,
    b: Int,
    order: Int,
    tolerance: Float64,
    metric: Int,
    inclusive: Bool,
) -> Bool:
    if metric == 0:
        var distance = Float64(0)
        for k in range(order):
            distance = max(distance, abs(x[a + k] - x[b + k]))
        return distance <= tolerance if inclusive else distance < tolerance

    var squared = Float64(0)
    for k in range(order):
        var delta = x[a + k] - x[b + k]
        squared += delta * delta
    var radius2 = tolerance * tolerance
    return squared <= radius2 if inclusive else squared < radius2


def worker_count(slot: U64Ptr) -> Int:
    # The Python wrapper owns the knob: a one-element UInt64 whose address is
    # handed over by ma_parallel_workers. Mojo rejects mutable globals, so this
    # is how the caller pins the worker count (1 = serial) without any state
    # living in the shared library.
    return max(1, min(Int(slot.unsafe_load(0)), WORKER_CAP))


def workers_param(slot_addr: Int) -> Int:
    if slot_addr == 0:
        return 1
    return worker_count(U64Ptr(unsafe_from_address=slot_addr))


# Row batches turn the batch into an explicit list of (row start, row stop)
# ranges and apply one range function to each range. A worker builds its range
# list once; the lists partition [0, rows). Every row is independent, so the
# parallel path differs from the serial one only in which thread owns a row: no
# row is ever split, each row computes the identical floating-point expression,
# and the two paths are bit-identical rather than merely within tolerance.
def split_ranges(count: Int, ranges: Int) -> List[List[Int]]:
    var pieces = ranges if count > 0 else 0
    var result = List[List[Int]]()
    result.reserve(pieces)
    if pieces == 0:
        return result^
    var base = count // pieces
    var extra = count % pieces
    var start = 0
    for piece in range(pieces):
        var stop = start + base + (1 if piece < extra else 0)
        var bounds = List[Int]()
        bounds.append(start)
        bounds.append(stop)
        result.append(bounds^)
        start = stop
    return result^


def empty_partials(count: Int) -> FPtr:
    return FPtr(unsafe_from_address=Int(alloc[Float64](count=count)))


# AppEnumerate is a double loop over template starts. Splitting the outer index
# across workers and summing integer match counts is exact: the integer totals
# do not depend on summation order. The entropy sum then runs serially over the
# rows in the same order as the serial version, so the logs accumulate in the
# same order and the two paths agree bit-for-bit.
def approximate_phi_parallel(
    x: FPtr,
    n: Int,
    order: Int,
    tolerance: Float64,
    metric: Int,
    counts: IPtr,
    workers: Int,
) -> Float64:
    var vectors = n - order + 1
    def work(start: Int) {imm}:
        var i = start
        while i < vectors:
            var matches = Int64(0)
            for j in range(vectors):
                if vectors_match(x, i, j, order, tolerance, metric, True):
                    matches += 1
            counts.unsafe_store(i, matches)
            i += workers
    parallelize(work, workers, workers)
    # Serial reduction in row order: identical floating-point operations to
    # approximate_phi, whose per-row logs are summed here in the same order.
    var total = Float64(0)
    for i in range(vectors):
        total += log(Float64(counts.unsafe_load(i)) / Float64(vectors))
    return total / Float64(vectors)


def approximate_entropy_parallel(
    x: FPtr,
    n: Int,
    order: Int,
    tolerance: Float64,
    metric: Int,
    counts: IPtr,
    workers: Int,
) -> Float64:
    var phi_short = approximate_phi_parallel(
        x, n, order, tolerance, metric, counts, workers
    )
    var phi_long = approximate_phi_parallel(
        x, n, order + 1, tolerance, metric, counts, workers
    )
    return phi_short - phi_long


def approximate_phi(
    x: FPtr, n: Int, order: Int, tolerance: Float64, metric: Int
) -> Float64:
    var vectors = n - order + 1
    var total = Float64(0)
    for i in range(vectors):
        var matches = Int(0)
        for j in range(vectors):
            if vectors_match(x, i, j, order, tolerance, metric, True):
                matches += 1
        total += log(Float64(matches) / Float64(vectors))
    return total / Float64(vectors)


def approximate_entropy(
    x: FPtr,
    n: Int,
    order: Int,
    tolerance: Float64,
    metric: Int,
    workers: Int,
    counts: IPtr,
) -> Float64:
    if n < APP_ENTROPY_PARALLEL_MIN or workers <= 1:
        return approximate_phi(x, n, order, tolerance, metric) - approximate_phi(
            x, n, order + 1, tolerance, metric
        )
    return approximate_entropy_parallel(
        x, n, order, tolerance, metric, counts, workers
    )


# --- GPU path for approximate entropy -------------------------------------
#
# Approximate entropy is the only measure here that clears the arithmetic-
# intensity gate in bench/analysis.md. Its match loop is O(n^2) over an
# n-sample signal: at n = 10k that is ~5e8 comparisons over an 80 KB signal,
# thousands of flops per byte, against the 2 flops/byte gate. Every other
# entry point streams its input once or a few times per sample and stays on
# the CPU; bench/analysis.md records them as memory-bound.
#
# Shape of the launch. The CPU path evaluates two Phi values, each a full n^2
# pass. The GPU does the same in two launches: one block per template row,
# with the block_dim lanes of that block splitting the columns and each lane
# keeping its own partial match count. The signal is copied to the device
# once per call and the partials are copied back once; the reduction and the
# log sum then run on the host in the same order the CPU uses, so the two
# paths agree exactly rather than merely to rounding.
#
# Tail handling. The grid is exactly `vectors` blocks, so there is no ragged
# grid edge to pad; a block index at or past `vectors` returns immediately
# for the defensive case where a caller launches a wider grid. The column
# loop is `j = lane; j < vectors; j += Lanes`, so lane counts differ by at
# most one and no lane reads past `vectors - 1`, whatever the remainder when
# `vectors` is divided by the lane count. Both properties are covered by the
# GPU parity test at sizes that are not multiples of the block size.
comptime GPU_APP_ENTROPY_MIN = 6000
comptime GPU_LANES = 256


def app_entropy_phi_kernel(
    x: FPtr,
    partials: IPtr,
    vectors_value: Int64,
    order_value: Int64,
    tolerance: Float64,
    metric: Int64,
    lanes_value: Int64,
):
    var vectors = Int(vectors_value)
    var order = Int(order_value)
    var lanes = Int(lanes_value)
    var row = Int(block_idx.x)
    if row >= vectors:
        return
    var lane = Int(thread_idx.x)
    var matches = Int64(0)
    var radius2 = tolerance * tolerance
    var j = lane
    while j < vectors:
        if metric == 0:
            var distance = Float64(0)
            for k in range(order):
                var delta = x[row + k] - x[j + k]
                if delta < 0.0:
                    delta = -delta
                if delta > distance:
                    distance = delta
            if distance <= tolerance:
                matches += 1
        else:
            var squared = Float64(0)
            for k in range(order):
                var delta = x[row + k] - x[j + k]
                squared += delta * delta
            if squared <= radius2:
                matches += 1
        j += lanes
    partials[row * lanes + lane] = matches


def gpu_available() -> Bool:
    # Constructing a context is how this toolchain reports whether a device
    # exists; it raises on a CPU-only host.
    try:
        var ctx = DeviceContext()
        _ = ctx.name()
        return True
    except:
        return False


def app_entropy_gpu_phi(
    x: FPtr, n: Int, order: Int, tolerance: Float64, metric: Int
) raises -> Float64:
    var vectors = n - order + 1
    var lanes = GPU_LANES
    # 8 bytes per lane partial plus the 8-byte-per-sample signal. At the
    # largest n this path is called with in practice (a few 100k) that is
    # ~2 GB of partials, so the ceiling below keeps every allocation well
    # inside the 2 GB device budget by halving the lane count instead. The
    # host reduction below uses the value chosen here, so the grid and the
    # reduction can never disagree about the lane count.
    while Int64(vectors) * Int64(lanes) * 8 > 1_500_000_000:
        lanes //= 2
        if lanes < 1:
            lanes = 1
            break
    var ctx = DeviceContext()
    var device_x = ctx.enqueue_create_buffer[DType.float64](n)
    var device_partials = ctx.enqueue_create_buffer[DType.int64](vectors * lanes)
    var host_partials = ctx.enqueue_create_host_buffer[DType.int64](vectors * lanes)
    ctx.enqueue_copy(device_x, x)
    ctx.enqueue_function[app_entropy_phi_kernel](
        device_x,
        device_partials,
        Int64(vectors),
        Int64(order),
        tolerance,
        Int64(metric),
        Int64(lanes),
        grid_dim=vectors,
        block_dim=lanes,
    )
    ctx.enqueue_copy(host_partials, device_partials)
    ctx.synchronize()
    var total = Float64(0)
    for i in range(vectors):
        var matches = Int64(0)
        var base = i * lanes
        for lane in range(lanes):
            matches += host_partials[base + lane]
        total += log(Float64(matches) / Float64(vectors))
    return total / Float64(vectors)


def app_entropy_gpu(
    x: FPtr, n: Int, order: Int, tolerance: Float64, metric: Int
) raises -> Float64:
    # The signal is uploaded twice, once per Phi. Fusing the two orders into
    # one launch would need a per-row block that matches both order and
    # order+1 in a single pass, which changes the summation order; keeping two
    # launches preserves the CPU path's arithmetic exactly.
    return app_entropy_gpu_phi(x, n, order, tolerance, metric) - app_entropy_gpu_phi(
        x, n, order + 1, tolerance, metric
    )


def sample_entropy(
    x: FPtr, n: Int, order: Int, tolerance: Float64, metric: Int
) -> Float64:
    # Serial on purpose. A parallel split of the outer template index was
    # measured here and reverted: at n = 1500 it tied serial (6.1 ms both
    # ways) and it lost to serial below that, so it would only have slowed the
    # sizes real callers pass. See README "Parallelism" for the numbers.
    var vectors = n - order
    if order == 2:
        var inclusive = not (metric == 0 and n < 5000)
        var counts = sample_entropy_order2_range(
            x, vectors, tolerance, metric, inclusive, 0, 1
        )
        var denominator = counts[0]
        var numerator = counts[1]
        if denominator == 0:
            return nan_value()
        if numerator == 0:
            return infinity()
        return -log(Float64(numerator) / Float64(denominator))

    var denominator = Int64(0)
    var numerator = Int64(0)
    var inclusive = not (metric == 0 and n < 5000)
    for i in range(vectors):
        for j in range(i + 1, vectors):
            if vectors_match(x, i, j, order, tolerance, metric, inclusive):
                denominator += 1
                if vectors_match(
                    x, i, j, order + 1, tolerance, metric, inclusive
                ):
                    numerator += 1
    if denominator == 0:
        return nan_value()
    if numerator == 0:
        return infinity()
    return -log(Float64(numerator) / Float64(denominator))


def sample_entropy_order2_range(
    x: FPtr,
    vectors: Int,
    tolerance: Float64,
    metric: Int,
    inclusive: Bool,
    first_i: Int,
    i_stride: Int,
) -> Tuple[Int64, Int64]:
    comptime W = simdwidthof[DType.float64]()
    var denominator = Int64(0)
    var numerator = Int64(0)
    var ones = SIMD[DType.int64, W](1)
    var zeros = SIMD[DType.int64, W](0)
    var radius2 = tolerance * tolerance
    var i = first_i
    while i < vectors:
        var xi0 = SIMD[DType.float64, W](x[i])
        var xi1 = SIMD[DType.float64, W](x[i + 1])
        var xi2 = SIMD[DType.float64, W](x[i + 2])
        var j = i + 1
        while j + W <= vectors:
            var d0 = abs(x.unsafe_load[width=W](j) - xi0)
            var d1 = abs(x.unsafe_load[width=W](j + 1) - xi1)
            var base = (
                max(d0, d1).le(tolerance)
                if inclusive
                else max(d0, d1).lt(tolerance)
            )
            if metric != 0:
                base = (
                    (d0 * d0 + d1 * d1).le(radius2)
                    if inclusive
                    else (d0 * d0 + d1 * d1).lt(radius2)
                )
            var d2 = abs(x.unsafe_load[width=W](j + 2) - xi2)
            var extended = base & (
                max(max(d0, d1), d2).le(tolerance)
                if inclusive
                else max(max(d0, d1), d2).lt(tolerance)
            )
            if metric != 0:
                extended = base & (
                    (d0 * d0 + d1 * d1 + d2 * d2).le(radius2)
                    if inclusive
                    else (d0 * d0 + d1 * d1 + d2 * d2).lt(radius2)
                )
            denominator += base.select(ones, zeros).reduce_add()
            numerator += extended.select(ones, zeros).reduce_add()
            j += W
        while j < vectors:
            var d0 = abs(x[j] - x[i])
            var d1 = abs(x[j + 1] - x[i + 1])
            var base_match = (
                max(d0, d1) <= tolerance
                if inclusive
                else max(d0, d1) < tolerance
            )
            if metric != 0:
                base_match = (
                    d0 * d0 + d1 * d1 <= radius2
                    if inclusive
                    else d0 * d0 + d1 * d1 < radius2
                )
            if base_match:
                denominator += 1
                var d2 = abs(x[j + 2] - x[i + 2])
                var extended_match = (
                    max(max(d0, d1), d2) <= tolerance
                    if inclusive
                    else max(max(d0, d1), d2) < tolerance
                )
                if metric != 0:
                    extended_match = (
                        d0 * d0 + d1 * d1 + d2 * d2 <= radius2
                        if inclusive
                        else d0 * d0 + d1 * d1 + d2 * d2 < radius2
                    )
                if extended_match:
                    numerator += 1
            j += 1
        i += i_stride
    return (denominator, numerator)


def lz_complexity(sequence: UPtr, n: Int) -> Int:
    if n == 0:
        return 1
    var complexity = Int(1)
    var prefix_len = Int(1)
    var substring_len = Int(1)
    var max_substring_len = Int(1)
    var pointer = Int(0)
    while prefix_len + substring_len <= n:
        if (
            sequence[pointer + substring_len - 1]
            == sequence[prefix_len + substring_len - 1]
        ):
            substring_len += 1
        else:
            max_substring_len = max(substring_len, max_substring_len)
            pointer += 1
            if pointer == prefix_len:
                complexity += 1
                prefix_len += max_substring_len
                pointer = 0
                max_substring_len = 1
            substring_len = 1
    if substring_len != 1:
        complexity += 1
    return complexity


def packed_word(sequence: BPtr, bit: Int) -> UInt64:
    var byte = bit // 8
    var shift = bit % 8
    var words = (sequence + byte).bitcast[UInt64]()
    var low = words.unsafe_load[alignment=1]()
    if shift == 0:
        return low
    var high = (sequence + byte + 8).bitcast[UInt64]().unsafe_load[alignment=1]()
    return (low >> UInt64(shift)) | (high << UInt64(64 - shift))


def lz_complexity_packed_binary(
    sequence: BPtr, windows: U64Ptr, n: Int
) -> Int:
    if n == 0:
        return 1
    for bit in range(n):
        windows[bit] = packed_word(sequence, bit)
    var complexity = 1
    var prefix_len = 1
    while prefix_len < n:
        var remaining = n - prefix_len
        var phrase_len = 1
        for pointer in range(prefix_len):
            var matched = 0
            while matched < remaining:
                var different = (
                    windows[pointer + matched]
                    ^ windows[prefix_len + matched]
                )
                var available = remaining - matched
                if different == 0:
                    matched += min(64, available)
                    if available <= 64:
                        break
                else:
                    matched += min(
                        Int(count_trailing_zeros(different)), available
                    )
                    break
            phrase_len = max(phrase_len, min(matched + 1, remaining))
            if phrase_len == remaining:
                break
        prefix_len += phrase_len
        complexity += 1
    return complexity


def zero_crossings_row(x: FPtr, n: Int) -> Int64:
    var crossings = Int64(0)
    for i in range(1, n):
        if (x[i - 1] < 0.0) != (x[i] < 0.0):
            crossings += 1
    return crossings


def variance(x: FPtr, start: Int, length: Int, stride: Int = 1) -> Float64:
    var mean = Float64(0)
    for i in range(length):
        mean += x[start + i * stride]
    mean /= Float64(length)
    var moment = Float64(0)
    for i in range(length):
        var delta = x[start + i * stride] - mean
        moment += delta * delta
    return moment / Float64(length)


def hjorth_row(
    x: FPtr,
    n: Int,
    sf: Float64,
    mobility_values: FPtr,
    complexity_values: FPtr,
    row: Int,
):
    var mean_x = Float64(0)
    var mean_dx = Float64(0)
    var mean_ddx = Float64(0)
    for i in range(n):
        mean_x += x[i]
    for i in range(n - 1):
        mean_dx += x[i + 1] - x[i]
    for i in range(n - 2):
        mean_ddx += x[i + 2] - 2.0 * x[i + 1] + x[i]
    mean_x /= Float64(n)
    mean_dx /= Float64(n - 1)
    mean_ddx /= Float64(n - 2)

    var x_var = Float64(0)
    var dx_var = Float64(0)
    var ddx_var = Float64(0)
    for i in range(n):
        var delta = x[i] - mean_x
        x_var += delta * delta
    for i in range(n - 1):
        var delta = (x[i + 1] - x[i]) - mean_dx
        dx_var += delta * delta
    for i in range(n - 2):
        var delta = (x[i + 2] - 2.0 * x[i + 1] + x[i]) - mean_ddx
        ddx_var += delta * delta
    x_var /= Float64(n)
    dx_var /= Float64(n - 1)
    ddx_var /= Float64(n - 2)

    var mobility = sqrt(dx_var / x_var)
    mobility_values[row] = mobility * sf
    complexity_values[row] = sqrt(ddx_var / dx_var) / mobility


def petrosian_row(x: FPtr, n: Int) -> Float64:
    var crossings = Int64(0)
    if n > 2:
        var previous_negative = (x[1] - x[0]) < 0.0
        for i in range(2, n):
            var negative = (x[i] - x[i - 1]) < 0.0
            if negative != previous_negative:
                crossings += 1
            previous_negative = negative
    var length = Float64(n)
    return log(length) / (
        log(length) + log(length / (length + 0.4 * Float64(crossings)))
    )


def katz_row(x: FPtr, n: Int) -> Float64:
    var curve_length = Float64(0)
    var farthest = Float64(0)
    for i in range(1, n):
        curve_length += abs(x[i] - x[i - 1])
        farthest = max(farthest, abs(x[i] - x[0]))
    var average = curve_length / Float64(n - 1)
    return log(curve_length / average) / log(farthest / average)


def linear_slope_from_sums(
    count: Int, sx: Float64, sx2: Float64, sy: Float64, sxy: Float64
) -> Float64:
    return (
        (Float64(count) * sxy - sx * sy)
        / (Float64(count) * sx2 - sx * sx + REG_EPS)
    )


def higuchi(x: FPtr, n: Int, kmax: Int) -> Float64:
    comptime W = simdwidthof[DType.float64]()
    var sx = Float64(0)
    var sx2 = Float64(0)
    var sy = Float64(0)
    var sxy = Float64(0)
    for k in range(1, kmax + 1):
        var quotient = (n - 1) // k
        var remainder = (n - 1) % k
        var total_curve = Float64(0)
        var i = k
        while i + W <= n:
            var delta = abs(
                x.unsafe_load[width=W](i) - x.unsafe_load[width=W](i - k)
            )
            total_curve += delta.reduce_add()
            i += W
        while i < n:
            total_curve += abs(x[i] - x[i - k])
            i += 1

        var smaller_curve = Float64(0)
        var smaller_is_first = remainder + 1 <= k - remainder - 1
        if smaller_is_first:
            for m in range(remainder + 1):
                for j in range(1, quotient + 1):
                    smaller_curve += abs(
                        x[m + j * k] - x[m + (j - 1) * k]
                    )
        else:
            for m in range(remainder + 1, k):
                for j in range(1, quotient):
                    smaller_curve += abs(
                        x[m + j * k] - x[m + (j - 1) * k]
                    )

        var first_curve = (
            smaller_curve
            if smaller_is_first
            else total_curve - smaller_curve
        )
        var second_curve = (
            total_curve - smaller_curve
            if smaller_is_first
            else smaller_curve
        )
        var weighted_curve = first_curve / Float64(quotient)
        if remainder + 1 < k:
            weighted_curve += second_curve / Float64(quotient - 1)
        var mean_length = (
            Float64(n - 1) * weighted_curve
            / Float64(k * k * k)
        )
        var xr = log(1.0 / Float64(k))
        var yr = log(mean_length) if mean_length > 0.0 else -infinity()
        sx += xr
        sx2 += xr * xr
        sy += yr
        sxy += xr * yr
    return linear_slope_from_sums(kmax, sx, sx2, sy, sxy)


def dfa_fluctuation(walk: FPtr, n: Int, scale: Int) -> Float64:
    var windows = n // scale
    var sx = Float64(scale * (scale - 1)) * 0.5
    var sx2 = Float64(scale * (scale - 1) * (2 * scale - 1)) / 6.0
    var total_fluctuation = Float64(0)
    for window in range(windows):
        var base = window * scale
        var sy = Float64(0)
        var sxy = Float64(0)
        for j in range(scale):
            sy += walk[base + j]
            sxy += Float64(j) * walk[base + j]
        var slope = linear_slope_from_sums(scale, sx, sx2, sy, sxy)
        var intercept = sy / Float64(scale) - slope * sx / Float64(scale)
        var residual = Float64(0)
        for j in range(scale):
            var delta = walk[base + j] - (intercept + slope * Float64(j))
            residual += delta * delta
        total_fluctuation += residual / Float64(scale)
    return sqrt(total_fluctuation / Float64(windows))


# DFA parallel path: one level of fan-out for the whole run, not one per scale.
#
# The first attempt parallelised the window loop of each scale separately. On
# this host that was 220 ms against 15 ms serial at n = 100k: ~40 tiny
# parallelize calls in a row cost more than the work they split. Instead the
# work is flattened into (scale, start) items, each worker walks its items in
# order and accumulates one running residual sum per scale in slot
# `worker * num_scales + scale`, and the totals are added scale by scale in
# worker order. Each window computes exactly the expression the serial loop
# computes and the per-scale additions happen in the same order, so the result
# is bit-identical to the serial path.
def dfa_scales(n: Int) -> List[Int]:
    var result = List[Int]()
    var max_scale = 0.1 * Float64(n)
    if max_scale <= 4.0:
        return result^
    var max_i = Int(floor(log(max_scale / 4.0) / log(1.2)))
    var last_scale = Int64(4)
    var current = Float64(4)
    for i in range(max_i + 1):
        if i > 0:
            current *= 1.2
        var scale = Int(floor(current))
        if Int64(scale) <= last_scale and i > 0:
            continue
        last_scale = Int64(scale)
        result.append(scale)
    return result^


def dfa_fluctuations_parallel(
    walk: FPtr, n: Int, scales: List[Int], workers: Int
) -> Float64:
    var num_scales = len(scales)
    var work_items = workers * num_scales
    var partials = empty_partials(work_items)
    for i in range(work_items):
        partials.unsafe_store(i, 0.0)
    def work(item: Int) {imm}:
        var worker = item // num_scales
        var scale = scales.unsafe_get(item % num_scales)
        var windows = n // scale
        var sx = Float64(scale * (scale - 1)) * 0.5
        var sx2 = Float64(scale * (scale - 1) * (2 * scale - 1)) / 6.0
        var total = Float64(0)
        var window = worker
        while window < windows:
            var base = window * scale
            var sy = Float64(0)
            var sxy = Float64(0)
            for j in range(scale):
                sy += walk[base + j]
                sxy += Float64(j) * walk[base + j]
            var slope = linear_slope_from_sums(scale, sx, sx2, sy, sxy)
            var intercept = sy / Float64(scale) - slope * sx / Float64(scale)
            var residual = Float64(0)
            for j in range(scale):
                var delta = walk[base + j] - (intercept + slope * Float64(j))
                residual += delta * delta
            total += residual / Float64(scale)
            window += workers
        partials.unsafe_store(item, total)
    parallelize(work, work_items, workers)

    var count = Int(0)
    var sx = Float64(0)
    var sx2 = Float64(0)
    var sy = Float64(0)
    var sxy = Float64(0)
    for scale_index in range(num_scales):
        var scale = scales.unsafe_get(scale_index)
        var windows = n // scale
        var total_fluctuation = Float64(0)
        for worker in range(workers):
            total_fluctuation += partials.unsafe_load(worker * num_scales + scale_index)
        var fluctuation = sqrt(total_fluctuation / Float64(windows))
        if fluctuation > 0.0:
            var lx = log(Float64(scale))
            var ly = log(fluctuation)
            sx += lx
            sx2 += lx * lx
            sy += ly
            sxy += lx * ly
            count += 1
    if count == 0:
        return nan_value()
    return linear_slope_from_sums(count, sx, sx2, sy, sxy)


def detrended_fluctuation(
    x: FPtr, walk: FPtr, n: Int, workers: Int
) -> Float64:
    var mean = Float64(0)
    for i in range(n):
        mean += x[i]
    mean /= Float64(n)
    var cumulative = Float64(0)
    for i in range(n):
        cumulative += x[i] - mean
        walk[i] = cumulative

    var max_scale = 0.1 * Float64(n)
    if max_scale <= 4.0:
        return nan_value()
    var scales = dfa_scales(n)
    if len(scales) == 0:
        return nan_value()
    if workers > 1 and n >= DFA_PARALLEL_MIN:
        return dfa_fluctuations_parallel(walk, n, scales, workers)

    var count = Int(0)
    var sx = Float64(0)
    var sx2 = Float64(0)
    var sy = Float64(0)
    var sxy = Float64(0)
    for scale_index in range(len(scales)):
        var scale = scales.unsafe_get(scale_index)
        var fluctuation = dfa_fluctuation(walk, n, scale)
        if fluctuation > 0.0:
            var lx = log(Float64(scale))
            var ly = log(fluctuation)
            sx += lx
            sx2 += lx * lx
            sy += ly
            sxy += lx * ly
            count += 1
    if count == 0:
        return nan_value()
    return linear_slope_from_sums(count, sx, sx2, sy, sxy)


@export("ma_perm_entropy")
def ma_perm_entropy(
    x_addr: Int,
    result_addr: Int,
    rows: Int,
    n: Int,
    order: Int,
    delay: Int,
    normalize: Int,
    keys_addr: Int,
    counts_addr: Int,
    work_addr: Int,
    capacity: Int,
) abi("C"):
    if (
        x_addr == 0 or result_addr == 0 or keys_addr == 0
        or counts_addr == 0 or work_addr == 0 or rows <= 0 or n <= 0
        or order < 2 or delay < 1 or capacity <= 0
    ):
        return
    var x = fp(x_addr)
    var result = fp(result_addr)
    var keys = ip(keys_addr)
    var counts = ip(counts_addr)
    var work = ip(work_addr)
    # Serial: real callers pass one long signal rather than a batch of rows,
    # and a multi-row batch was not measured to win with workers, so no
    # parallel path is claimed here. Each call still clears its own table.
    for row in range(rows):
        result.unsafe_store(
            row,
            permutation_entropy_row(
                x + row * n,
                n,
                order,
                delay,
                normalize != 0,
                keys,
                counts,
                work,
                capacity,
            ),
        )


@export("ma_perm_entropy_order3")
def ma_perm_entropy_order3(
    x_addr: Int,
    result_addr: Int,
    rows: Int,
    n: Int,
    delay: Int,
    normalize: Int,
) abi("C"):
    if x_addr == 0 or result_addr == 0 or rows <= 0 or n <= 0 or delay < 1:
        return
    var x = fp(x_addr)
    var result = fp(result_addr)
    # Serial: same reasoning as ma_perm_entropy -- see the note there.
    for row in range(rows):
        result.unsafe_store(
            row,
            permutation_entropy_order3_row(
                x + row * n,
                n,
                delay,
                normalize != 0,
            ),
        )


@export("ma_app_entropy")
def ma_app_entropy(
    x_addr: Int,
    counts_addr: Int,
    n: Int,
    order: Int,
    tolerance: Float64,
    metric: Int,
    workers_addr: Int,
    use_gpu: Int = 0,
) abi("C") -> Float64:
    if x_addr == 0 or counts_addr == 0 or n <= order or order < 1:
        return nan_value()
    # The CPU path stays the default and the fallback. A device is only tried
    # when the caller opts in and the signal is large enough for the host
    # round-trip to be repaid; any device error falls back rather than
    # propagating, so a host without a usable GPU behaves exactly as before.
    if use_gpu != 0 and n >= GPU_APP_ENTROPY_MIN:
        try:
            return app_entropy_gpu(fp(x_addr), n, order, tolerance, metric)
        except:
            pass
    return approximate_entropy(
        fp(x_addr),
        n,
        order,
        tolerance,
        metric,
        workers_param(workers_addr),
        ip(counts_addr),
    )


@export("ma_gpu_available")
def ma_gpu_available() abi("C") -> Int:
    """1 when a device context can be created, 0 on a CPU-only host."""
    return 1 if gpu_available() else 0


@export("ma_gpu_app_entropy_min")
def ma_gpu_app_entropy_min() abi("C") -> Int:
    """Smallest signal size the GPU approximate-entropy path accepts."""
    return GPU_APP_ENTROPY_MIN


@export("ma_sample_entropy")
def ma_sample_entropy(
    x_addr: Int, n: Int, order: Int, tolerance: Float64, metric: Int
) abi("C") -> Float64:
    if x_addr == 0 or n <= order or order < 1:
        return nan_value()
    return sample_entropy(fp(x_addr), n, order, tolerance, metric)


@export("ma_sample_entropy_order2_counts")
def ma_sample_entropy_order2_counts(
    x_addr: Int,
    counts_addr: Int,
    n: Int,
    tolerance: Float64,
    metric: Int,
    first_i: Int,
    i_stride: Int,
) abi("C"):
    if (
        x_addr == 0 or counts_addr == 0 or n <= 2
        or first_i < 0 or i_stride < 1
    ):
        return
    var counts = sample_entropy_order2_range(
        fp(x_addr),
        n - 2,
        tolerance,
        metric,
        not (metric == 0 and n < 5000),
        first_i,
        i_stride,
    )
    var result = ip(counts_addr)
    result.unsafe_store(0, counts[0])
    result.unsafe_store(1, counts[1])


@export("ma_parallel_capacity")
def ma_parallel_capacity(size: Int) abi("C") -> Int:
    """Int64 slots the caller must provide for a parallel scratch buffer."""
    return max(size, 1) if size < 2_000_000_000 else 2_000_000_000


@export("ma_parallel_cap")
def ma_parallel_cap() abi("C") -> Int:
    """Worker ceiling for every parallel path (this host is shared)."""
    return WORKER_CAP


@export("ma_parallel_workers")
def ma_parallel_workers(slot_addr: Int) abi("C"):
    """Validate the caller's worker knob.

    There is no library-side state to set: the knob is the caller's own UInt64,
    whose address is passed to each kernel as ``workers_addr``.
    """
    _ = slot_addr


@export("ma_lziv_complexity")
def ma_lziv_complexity(sequence_addr: Int, n: Int) abi("C") -> Int:
    if n == 0:
        return 1
    if sequence_addr == 0 or n < 0:
        return 0
    return lz_complexity(up(sequence_addr), n)


@export("ma_lziv_complexity_packed_binary")
def ma_lziv_complexity_packed_binary(
    sequence_addr: Int, windows_addr: Int, n: Int
) abi("C") -> Int:
    if n == 0:
        return 1
    if sequence_addr == 0 or windows_addr == 0 or n < 0:
        return 0
    return lz_complexity_packed_binary(
        BPtr(unsafe_from_address=sequence_addr),
        U64Ptr(unsafe_from_address=windows_addr),
        n,
    )


@export("ma_num_zerocross")
def ma_num_zerocross(
    x_addr: Int, result_addr: Int, rows: Int, n: Int
) abi("C"):
    if x_addr == 0 or result_addr == 0 or rows <= 0 or n <= 0:
        return
    var x = fp(x_addr)
    var result = ip(result_addr)
    # Serial: this kernel is memory-bound and four workers measured slower than
    # one at 512-1000 rows on the benchmark host (see README).
    for row in range(rows):
        result.unsafe_store(row, zero_crossings_row(x + row * n, n))


@export("ma_hjorth_params")
def ma_hjorth_params(
    x_addr: Int,
    mobility_addr: Int,
    complexity_addr: Int,
    rows: Int,
    n: Int,
    sf: Float64,
    workers_addr: Int,
) abi("C"):
    if (
        x_addr == 0 or mobility_addr == 0 or complexity_addr == 0
        or rows <= 0 or n < 3
    ):
        return
    var x = fp(x_addr)
    var mobility = fp(mobility_addr)
    var complexity = fp(complexity_addr)
    var workers = workers_param(workers_addr)
    if rows < HJORTH_ROWS_PARALLEL_MIN:
        workers = 1
    var ranges = split_ranges(rows, workers)
    def run_range(index: Int) {imm}:
        var start = ranges[index].unsafe_get(0)
        var stop = ranges[index].unsafe_get(1)
        for row in range(start, stop):
            hjorth_row(x + row * n, n, sf, mobility, complexity, row)
    if workers > 1:
        parallelize(run_range, workers, workers)
    else:
        run_range(0)


@export("ma_petrosian_fd")
def ma_petrosian_fd(
    x_addr: Int, result_addr: Int, rows: Int, n: Int
) abi("C"):
    if x_addr == 0 or result_addr == 0 or rows <= 0 or n <= 0:
        return
    var x = fp(x_addr)
    var result = fp(result_addr)
    # Serial: this kernel is memory-bound and four workers measured slower than
    # one at 512-1000 rows on the benchmark host (see README).
    for row in range(rows):
        result.unsafe_store(row, petrosian_row(x + row * n, n))


@export("ma_katz_fd")
def ma_katz_fd(
    x_addr: Int, result_addr: Int, rows: Int, n: Int
) abi("C"):
    if x_addr == 0 or result_addr == 0 or rows <= 0 or n < 2:
        return
    var x = fp(x_addr)
    var result = fp(result_addr)
    # Serial: this kernel is memory-bound and four workers measured slower than
    # one at 512-1000 rows on the benchmark host (see README).
    for row in range(rows):
        result.unsafe_store(row, katz_row(x + row * n, n))


@export("ma_higuchi_fd")
def ma_higuchi_fd(x_addr: Int, n: Int, kmax: Int) abi("C") -> Float64:
    if x_addr == 0 or n <= 0 or kmax < 2 or kmax >= n:
        return nan_value()
    return higuchi(fp(x_addr), n, kmax)


@export("ma_detrended_fluctuation")
def ma_detrended_fluctuation(
    x_addr: Int, walk_addr: Int, n: Int, workers_addr: Int
) abi("C") -> Float64:
    if x_addr == 0 or walk_addr == 0 or n <= 0:
        return nan_value()
    return detrended_fluctuation(
        fp(x_addr), fp(walk_addr), n, workers_param(workers_addr)
    )
