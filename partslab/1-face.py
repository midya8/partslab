"""Exact supplied-binary64 fixed-face NNLS response, not a full solution path."""
from fractions import Fraction as F
from .certificate import _problem, _fraction
from .io import dumps


def solve(matrix, right):
    """Exact Gauss-Jordan with two right-hand sides; singularity is not repaired."""
    n = len(matrix)
    a = [list(row) + list(rhs) for row, rhs in zip(matrix, right)]
    for k in range(n):
        pivot = next((i for i in range(k, n) if a[i][k]), None)
        if pivot is None:
            raise ValueError('singular selected Gram block')
        a[k], a[pivot] = a[pivot], a[k]
        div = a[k][k]
        a[k] = [v / div for v in a[k]]
        for i in range(n):
            if i != k:
                mul = a[i][k]
                a[i] = [u - mul * v for u, v in zip(a[i], a[k])]
    return [row[n:] for row in a]


def _input(pixels, basis, support, pixel):
    x, b, _ = _problem(pixels, basis, [0] * len(basis))
    if any(v > 1 for v in x):
        raise ValueError('pixels must be in [0,1]')
    if type(pixel) is not int or not 0 <= pixel < len(x):
        raise ValueError('integer pixel index required')
    if not isinstance(support, list) or any(type(k) is not int or not 0 <= k < len(b) for k in support) or support != sorted(set(support)):
        raise ValueError('sorted distinct integer support indices required')
    return x, b


def _forms(x, b, c0, c1, pixel, support):
    """Recompute exact coefficient and inactive-gradient half-lines."""
    r0 = [sum((c0[k] * b[k][j] for k in range(len(b))), F()) - x[j] for j in range(len(x))]
    r1 = [sum((c1[k] * b[k][j] for k in range(len(b))), F()) - int(j == pixel) for j in range(len(x))]
    gradient = [(sum((v * w for v, w in zip(row, r0)), F()), sum((v * w for v, w in zip(row, r1)), F())) for row in b]
    if any(gradient[k] != (0, 0) for k in support):
        raise ValueError('selected stationarity failure')
    if any(c0[k] != 0 or c1[k] != 0 for k in range(len(b)) if k not in support):
        raise ValueError('unselected coefficient failure')
    forms = [('coefficient', k, c0[k], c1[k]) for k in support]
    forms += [('gradient', k, *gradient[k]) for k in range(len(b)) if k not in support]
    return forms


def _interval(forms, x, pixel):
    lower, upper = -x[pixel], 1 - x[pixel]
    for _, _, a, b in forms:
        if a < 0:
            raise ValueError('candidate fails exact KKT at starting point')
        if b > 0:
            lower = max(lower, -a / b)
        elif b < 0:
            upper = min(upper, -a / b)
    # Zero is included after every validated KKT inequality; singleton is allowed.
    if lower > 0 or upper < 0:
        raise ArithmeticError('starting point excluded')
    def limits(t, domain):
        out = [{'kind': kind, 'index': k} for kind, k, a, b in forms if b and a + b * t == 0]
        if t == domain:
            out.append({'kind': 'pixel-domain', 'index': pixel})
        return out
    return lower, upper, limits(lower, -x[pixel]), limits(upper, 1 - x[pixel])


def response(pixels, basis, support, pixel):
    """Solve a nominated face and certify its closed candidate-validity interval.

    Support can be suggested numerically, but exact starting KKT is mandatory.
    Closed endpoints may have zero selected coefficients or zero unselected
    gradients. Thus support is a working face, not strict positive support.
    """
    xraw, braw = _input(pixels, basis, support, pixel)
    x, b = list(map(F, xraw)), [list(map(F, row)) for row in braw]
    gram = [[sum((u * v for u, v in zip(b[k], b[l])), F()) for l in support] for k in support]
    right = [[sum((u * v for u, v in zip(b[k], x)), F()), b[k][pixel]] for k in support]
    answer = solve(gram, right)
    c0, c1 = [F()] * len(b), [F()] * len(b)
    for k, (a, d) in zip(support, answer):
        c0[k], c1[k] = a, d
    forms = _forms(x, b, c0, c1, pixel, support)
    lo, hi, lolim, hilim = _interval(forms, x, pixel)
    return {'schema': 1, 'scope': 'exact supplied binary64; fixed-basis NNLS candidate face; pixel delta',
            'problem': {'pixels': xraw, 'basis': braw, 'support': list(support), 'pixel': pixel},
            'coefficients': [{'intercept': _fraction(a), 'slope': _fraction(d)} for a, d in zip(c0, c1)],
            'inequalities': [{'kind': kind, 'index': k, 'intercept': _fraction(a), 'slope': _fraction(d)} for kind, k, a, d in forms],
            'interval': {'lower': _fraction(lo), 'upper': _fraction(hi), 'lower_limits': lolim, 'upper_limits': hilim,
                         'singleton': lo == hi}}


def verify(witness):
    """Verify stationarity, half-lines, all endpoints and maximality without solving.

    A self-contained mathematical witness is not authenticated model provenance.
    """
    from .certificate import _read_fraction
    try:
        if type(witness['schema']) is not int or witness['schema'] != 1 or witness['scope'] != 'exact supplied binary64; fixed-basis NNLS candidate face; pixel delta':
            return False
        p = witness['problem']
        xr, br = _input(p['pixels'], p['basis'], p['support'], p['pixel'])
        x, b = list(map(F, xr)), [list(map(F, row)) for row in br]
        coefficients = witness['coefficients']
        if not isinstance(coefficients, list) or len(coefficients) != len(b):
            return False
        c0 = [_read_fraction(c['intercept']) for c in coefficients]
        c1 = [_read_fraction(c['slope']) for c in coefficients]
        # Fraction inputs must already be canonical, rather than merely equal.
        if coefficients != [{'intercept': _fraction(a), 'slope': _fraction(d)} for a, d in zip(c0, c1)]:
            return False
        forms = _forms(x, b, c0, c1, p['pixel'], p['support'])
        if dumps(witness['inequalities']) != dumps([{'kind': kind, 'index': k, 'intercept': _fraction(a), 'slope': _fraction(d)} for kind, k, a, d in forms]):
            return False
        lo, hi, lolim, hilim = _interval(forms, x, p['pixel'])
        return dumps(witness['interval']) == dumps({'lower': _fraction(lo), 'upper': _fraction(hi), 'lower_limits': lolim, 'upper_limits': hilim, 'singleton': lo == hi})
    except (ValueError, TypeError, KeyError, OverflowError, ArithmeticError):
        return False
