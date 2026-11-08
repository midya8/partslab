# Exact fixed-basis NNLS objective bounds

PartsLab v0.2 adds a separate primal/dual certificate. The numerical solver and its KKT stopping rule are unchanged. An exact certificate is computed for its candidate coefficients, including when the numerical solver stops unfinished.

This certifies the convex coefficient problem for a **fixed** basis. It does not certify joint NMF optimality, unique coefficients, semantic parts, model quality or the origin of an imported problem. All six joint fits still exhaust their original stationarity budget. All original models, scores, training protocol and frozen guards are unchanged.

## The bound

Let `B` have nonnegative rows, `x` be the input vector, and `c >= 0` the candidate coefficients. Use the half squared error, not per-pixel MSE:

```
P(c) = 1/2 ||x - cB||^2
D(y) = x dot y - 1/2 ||y||^2, provided B y <= 0
```

For every feasible `c` and `y`, writing `r = x-cB` gives the exact identity

```
P(c) - D(y) = 1/2 ||r-y||^2 - c dot (B y) >= 0.
```

Thus `D(y) <= P* <= P(c)` and `0 <= P(c)-P* <= P(c)-D(y)`. The proof needs weak duality only. It does not assume that the coefficient solver has converged or that the basis has full row rank. Zero and duplicate rows are allowed. Nonnegativity of `c` is checked, rather than inferred from a solver label.

## A feasible dual vector

For a nonzero nonnegative row `B_i`, its row sum `s_i` is strictly positive. Choose

```
tau = max(0, max_i (B_i dot r) / s_i)
y = r - tau * ones
```

Every row then satisfies `B_i dot y <= 0` exactly. A zero row imposes the vacuous constraint `0 <= 0` and is skipped in the ratio. If this shifted residual has negative dual value, choose `y=0`, whose dual value is zero. This can be a loose bound. We do not solve a second optimization problem or call this dual vector optimal.

The identity-basis candidate `x=[1,1], c=[1,0]` illustrates a loose bound: its primal value and certified surplus bound are both `1/2`, while the optimum is zero. At zero inference sweeps, a valid bound still exists; validity must not be confused with a useful tolerance. With duplicate rows `B=[[1,0],[1,0]]`, both `c=[1,0]` and `c=[0,1]` have zero gap for `x=[1,0]`. Zero gap does not identify unique coefficients.

## Exact arithmetic over the represented problem

The browser reads each finite nonnegative binary64 number's IEEE-754 bits with `DataView`. Python uses each float's exact integer ratio. Their common denominator is a power of two. Inputs, basis entries and candidate coefficients are represented exactly, without decimal quantization, a feasibility epsilon, or rounded matrix products. BigInt/integer operations construct the shifted residual, check dual feasibility and evaluate both objectives. Rational numerators and denominators are exported as decimal **strings**, never JSON numbers.

This is the mathematical NNLS problem whose coefficients are the exact represented binary64 values. It is not a certificate for hypothetical unrounded learned factors, image acquisition uncertainty, or the floating-point accumulation order used to draw a canvas. The certified primal value can differ in its last bits from a floating-point reconstructed MSE. The original holdout table remains unchanged.

The displayed optimum lower bound is rounded down. The displayed primal upper bound and suboptimality bound are rounded up to six significant decimal digits, using integer arithmetic. The underlying rational fractions are available in the downloaded JSON. An exponent in that display is not an estimated tolerance. The bounds are half squared error; to convert a gap to MSE units, multiply by `2 / number_of_pixels`.

## Portable verification

The downloadable inference JSON includes a `certificate` object with the problem, a rational dual vector, rational primal/dual/gap values and outward-rounded display values. Copy that object into the browser's "Check a portable certificate" panel. The checker validates the embedded problem only. It does not authenticate where its basis or input came from. The browser has no uploads, server or model calls.

The package includes a separate direct `fractions.Fraction` verifier. It does not reconstruct the residual-shift procedure. It checks every dual inequality and directly recomputes both objectives, their gap and the display rounding.

```bash
# Extract the certificate object from an inference download.
python -c 'import json; w=json.load(open("partslab-inference.json")); json.dump(w["certificate"],open("certificate.json","w"))'
python -m partslab verify-certificate --witness certificate.json

# Or generate a certificate from a JSON object containing pixels, basis, coefficients.
python -m partslab certify --problem problem.json --out certificate.json
```

Invalid proofs return a nonzero CLI exit status. Changing a feasible problem and recomputing a correct proof can produce another valid certificate; verification is not an authenticity check. A browser-generated candidate may differ slightly from NumPy's candidate because of floating-point update order. Each certificate applies to its own supplied coefficient values.

## Independent checks and coverage

- Eighty tiny NNLS problems are solved independently by enumerating active supports with exact rational Gaussian elimination and exact KKT feasibility. The optimum lies in every certified interval, and each actual candidate surplus is no larger than its certified gap.
- Another exact check verifies the sum-of-squares/slack identity above. Subnormals, ordinary inexact decimal floats, powers of two, zero rows, zero inputs, unfinished and zero-lock candidates, duplicate-row nonuniqueness, outward rounding and tampered witnesses are covered.
- All 120 already-public model/example pairs have native exact witness references in `partslab/artifacts/certificates.json`. A separate JavaScript BigInt implementation reproduces those witnesses exactly using the **same native candidate coefficients**. These are twenty fixed public examples for six models, not a gap audit of all 1,797 holdout rows.
- Real Chrome computes and verifies certificates for its own 120 numerical candidates. It tests the zero-sweep unfinished control, accepts a valid pasted proof, rejects a tampered gap, and retains the original contribution, input, keyboard, copy/download and responsive-layout checks.

The complete suite is 30 Python tests and eight Node/real-Chrome tests. The installed wheel imports eight modules and executes all seven CLI subcommands outside the source checkout, including all six refits and the original holdout reproduction.

## Sources and attribution

The weak-duality principle and quadratic dual construction are standard. The residual-shift implementation and exact witness format here are this project's implementation, not a claimed new optimization algorithm. The following sources informed the derivation and representation:

- Boyd & Vandenberghe, [Duality lecture](https://see.stanford.edu/materials/lsocoee364a/05Duality.pdf): dual feasibility, weak duality and lower bounds.
- Boyd & Vandenberghe, [Convex Optimization](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf): convex least-squares context and duality. The directly checked identity above is the proof used here.
- Diakonikolas, Li, Padmanabhan & Song, [A Fast Scale-Invariant Algorithm for Non-negative Least Squares with Non-negative Data](https://ar5iv.labs.arxiv.org/html/2203.03808): related primal/dual NNLS literature. Its algorithm and complexity guarantees are **not** implemented or claimed here.
- Python, [Floating-Point Arithmetic](https://docs.python.org/3/tutorial/floatingpoint.html): exact `float.as_integer_ratio()` representation.
- MDN, [DataView.getBigUint64](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/DataView/getBigUint64) and [DataView.getFloat64](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/DataView/getFloat64): reading the binary64 bit pattern.

Original data and NMF attribution remain in [METHOD.md](METHOD.md). No endorsement is implied.
