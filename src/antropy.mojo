from std.math import abs, floor, log, sqrt
from std.memory import UnsafePointer
from std.sys import simd_width_of


comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime UPtr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime LN2 = 0.693147180559945309417232121458
comptime REG_EPS = 1.0e-9


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
    x: FPtr, n: Int, order: Int, tolerance: Float64, metric: Int
) -> Float64:
    return (
        approximate_phi(x, n, order, tolerance, metric)
        - approximate_phi(x, n, order + 1, tolerance, metric)
    )


def sample_entropy(
    x: FPtr, n: Int, order: Int, tolerance: Float64, metric: Int
) -> Float64:
    var vectors = n - order
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
    comptime W = simd_width_of[DType.float64]()
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
                x.load[width=W](i) - x.load[width=W](i - k)
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


def detrended_fluctuation(x: FPtr, walk: FPtr, n: Int) -> Float64:
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
    var max_i = Int(floor(log(max_scale / 4.0) / log(1.2)))
    var last_scale = Int64(4)
    var count = Int(0)
    var sx = Float64(0)
    var sx2 = Float64(0)
    var sy = Float64(0)
    var sxy = Float64(0)
    var current = Float64(4)
    for i in range(max_i + 1):
        if i > 0:
            current *= 1.2
        var scale = Int(floor(current))
        if Int64(scale) <= last_scale and i > 0:
            continue
        last_scale = Int64(scale)
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
    for row in range(rows):
        result[row] = permutation_entropy_row(
            x + row * n,
            n,
            order,
            delay,
            normalize != 0,
            keys,
            counts,
            work,
            capacity,
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
    for row in range(rows):
        result[row] = permutation_entropy_order3_row(
            x + row * n,
            n,
            delay,
            normalize != 0,
        )


@export("ma_app_entropy")
def ma_app_entropy(
    x_addr: Int, n: Int, order: Int, tolerance: Float64, metric: Int
) abi("C") -> Float64:
    if x_addr == 0 or n <= order or order < 1:
        return nan_value()
    return approximate_entropy(fp(x_addr), n, order, tolerance, metric)


@export("ma_sample_entropy")
def ma_sample_entropy(
    x_addr: Int, n: Int, order: Int, tolerance: Float64, metric: Int
) abi("C") -> Float64:
    if x_addr == 0 or n <= order or order < 1:
        return nan_value()
    return sample_entropy(fp(x_addr), n, order, tolerance, metric)


@export("ma_lziv_complexity")
def ma_lziv_complexity(sequence_addr: Int, n: Int) abi("C") -> Int:
    if n == 0:
        return 1
    if sequence_addr == 0 or n < 0:
        return 0
    return lz_complexity(up(sequence_addr), n)


@export("ma_num_zerocross")
def ma_num_zerocross(
    x_addr: Int, result_addr: Int, rows: Int, n: Int
) abi("C"):
    if x_addr == 0 or result_addr == 0 or rows <= 0 or n <= 0:
        return
    var x = fp(x_addr)
    var result = ip(result_addr)
    for row in range(rows):
        result[row] = zero_crossings_row(x + row * n, n)


@export("ma_hjorth_params")
def ma_hjorth_params(
    x_addr: Int,
    mobility_addr: Int,
    complexity_addr: Int,
    rows: Int,
    n: Int,
    sf: Float64,
) abi("C"):
    if (
        x_addr == 0 or mobility_addr == 0 or complexity_addr == 0
        or rows <= 0 or n < 3
    ):
        return
    var x = fp(x_addr)
    var mobility = fp(mobility_addr)
    var complexity = fp(complexity_addr)
    for row in range(rows):
        hjorth_row(x + row * n, n, sf, mobility, complexity, row)


@export("ma_petrosian_fd")
def ma_petrosian_fd(
    x_addr: Int, result_addr: Int, rows: Int, n: Int
) abi("C"):
    if x_addr == 0 or result_addr == 0 or rows <= 0 or n <= 0:
        return
    var x = fp(x_addr)
    var result = fp(result_addr)
    for row in range(rows):
        result[row] = petrosian_row(x + row * n, n)


@export("ma_katz_fd")
def ma_katz_fd(
    x_addr: Int, result_addr: Int, rows: Int, n: Int
) abi("C"):
    if x_addr == 0 or result_addr == 0 or rows <= 0 or n < 2:
        return
    var x = fp(x_addr)
    var result = fp(result_addr)
    for row in range(rows):
        result[row] = katz_row(x + row * n, n)


@export("ma_higuchi_fd")
def ma_higuchi_fd(x_addr: Int, n: Int, kmax: Int) abi("C") -> Float64:
    if x_addr == 0 or n <= 0 or kmax < 2 or kmax >= n:
        return nan_value()
    return higuchi(fp(x_addr), n, kmax)


@export("ma_detrended_fluctuation")
def ma_detrended_fluctuation(
    x_addr: Int, walk_addr: Int, n: Int
) abi("C") -> Float64:
    if x_addr == 0 or walk_addr == 0 or n <= 0:
        return nan_value()
    return detrended_fluctuation(fp(x_addr), fp(walk_addr), n)
