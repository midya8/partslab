"""Exact fixed-basis NNLS weak-duality certificates, not joint NMF certificates."""
from fractions import Fraction
import math
import re
from numbers import Real


def _problem(pixels, basis, coefficients):
    raw = list(pixels) + list(coefficients) + [v for row in basis for v in row]
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in raw):
        raise ValueError('numeric binary64 values required')
    x = [float(v) for v in pixels]
    b = [[float(v) for v in row] for row in basis]
    c = [float(v) for v in coefficients]
    if not 1 <= len(x) <= 256 or not 1 <= len(b) <= 64 or len(c) != len(b) or any(len(row) != len(x) for row in b):
        raise ValueError('inconsistent certificate dimensions')
    if any(not math.isfinite(v) or v < 0 for v in x + c + [v for row in b for v in row]):
        raise ValueError('finite nonnegative binary64 values required')
    return x, b, c


def _fraction(value):
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}


def decimal_bound(value, upper=False, digits=6):
    """Outward-rounded scientific decimal string for a nonnegative Fraction."""
    n, d = value.numerator, value.denominator
    if n < 0 or not 1 <= digits <= 20:
        raise ValueError('nonnegative fraction and 1..20 digits required')
    if n == 0:
        return '0'
    exponent = len(str(n)) - len(str(d))
    if (n < d * 10**exponent) if exponent >= 0 else (n * 10**(-exponent) < d):
        exponent -= 1
    power = digits - 1 - exponent
    a, b = (n * 10**power, d) if power >= 0 else (n, d * 10**(-power))
    q, rem = divmod(a, b)
    if upper and rem:
        q += 1
    if q >= 10**digits:
        q //= 10
        exponent += 1
    s = str(q).zfill(digits)
    return s[0] + ('.' + s[1:] if digits > 1 else '') + 'e' + str(exponent)


def certify(pixels, basis, coefficients):
    x, b, c = _problem(pixels, basis, coefficients)
    values = x + c + [v for row in b for v in row]
    scale = max(v.as_integer_ratio()[1] for v in values)
    integer = lambda v: v.as_integer_ratio()[0] * (scale // v.as_integer_ratio()[1])
    xi, bi, ci = list(map(integer, x)), [list(map(integer, row)) for row in b], list(map(integer, c))
    rscale = scale * scale
    xr = [v * scale for v in xi]
    residual = [xr[j] - sum(ci[k] * bi[k][j] for k in range(len(b))) for j in range(len(x))]
    # tau = N / (R * D). Every positive row sum makes B(r-tau*1) <= 0.
    n, d = 0, 1
    for row in bi:
        row_sum = sum(row)
        cross = sum(v * r for v, r in zip(row, residual))
        if row_sum and cross * d > n * row_sum:
            n, d = cross, row_sum
    y = [r * d - n for r in residual]
    yscale = rscale * d
    primal = Fraction(sum(r * r for r in residual), 2 * rscale * rscale)
    dual = sum((Fraction(v, scale) * Fraction(w, yscale) for v, w in zip(xi, y)), Fraction()) - sum((Fraction(w * w, 2 * yscale * yscale) for w in y), Fraction())
    source = 'shifted-residual'
    if dual < 0:
        y, yscale, dual, source = [0] * len(x), 1, Fraction(), 'zero-fallback'
    gap = primal - dual
    if gap < 0:
        raise ArithmeticError('weak duality violated')
    return {'schema': 1, 'scope': 'fixed-basis NNLS; exact supplied binary64 values',
            'problem': {'pixels': x, 'basis': b, 'coefficients': c},
            'dual_numerators': list(map(str, y)), 'dual_denominator': str(yscale), 'dual_source': source,
            'primal': _fraction(primal), 'dual': _fraction(dual), 'gap': _fraction(gap),
            'display': {'optimal_objective_lower': decimal_bound(dual), 'optimal_objective_upper': decimal_bound(primal, True),
                        'suboptimality_upper': decimal_bound(gap, True)}}


def _int(value):
    if not isinstance(value, str) or not re.fullmatch(r'0|-?[1-9][0-9]*', value) or len(value) > 10000:
        raise ValueError('canonical bounded integer string required')
    return int(value)


def _read_fraction(value):
    n, d = _int(value['numerator']), _int(value['denominator'])
    if d <= 0:
        raise ValueError('positive denominator required')
    return Fraction(n, d)


def verify(witness):
    """Check constraints and objectives directly with Fraction, no shift construction."""
    try:
        if type(witness['schema']) is not int or witness['schema'] != 1 or witness['scope'] != 'fixed-basis NNLS; exact supplied binary64 values':
            return False
        p = witness['problem']
        x, b, c = _problem(p['pixels'], p['basis'], p['coefficients'])
        x, b, c = list(map(Fraction, x)), [list(map(Fraction, row)) for row in b], list(map(Fraction, c))
        denominator = _int(witness['dual_denominator'])
        if denominator <= 0 or not isinstance(witness['dual_numerators'], list) or len(witness['dual_numerators']) != len(x):
            return False
        y = [Fraction(_int(v), denominator) for v in witness['dual_numerators']]
        if any(sum((a * v for a, v in zip(row, y)), Fraction()) > 0 for row in b):
            return False
        residual = [x[j] - sum((c[k] * b[k][j] for k in range(len(c))), Fraction()) for j in range(len(x))]
        primal = sum((r*r/2 for r in residual), Fraction())
        dual = sum((v*w - w*w/2 for v, w in zip(x, y)), Fraction())
        gap = primal - dual
        if dual < 0 or gap < 0 or any(_read_fraction(witness[k]) != v for k, v in [('primal', primal), ('dual', dual), ('gap', gap)]):
            return False
        return witness['display'] == {'optimal_objective_lower': decimal_bound(dual), 'optimal_objective_upper': decimal_bound(primal, True), 'suboptimality_upper': decimal_bound(gap, True)}
    except (ValueError, TypeError, KeyError, OverflowError):
        return False
