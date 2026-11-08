# PartsLab v0.3: one face, one pixel

The fixed-basis NNLS problem is convex. The fitted NMF problem is not jointly convex. This addition certifies a nominated affine NNLS candidate, never a jointly optimal NMF fit, a unique coefficient vector, a whole solution path or a robustness radius.

The original six bases, data/splits, coordinate-descent solver, numerical KKT stopping warnings, reconstruction results, original exact weak-duality witnesses and stronger signed PCA/SVD controls are unchanged. No new performance result is reported.

## Exact problem

Treat each finite binary64 input as its exact dyadic rational value, not its rounded decimal spelling. For a fixed nonnegative basis B (rank by pixel), original x in [0,1]^p, selected coordinate j, and a sorted distinct nominated face S, set x(t) = x + t e_j. The pixel domain is [-x_j, 1-x_j].

Write G = BB^T and h(t) = Bx(t). Construct c outside S equal to zero and solve

    G_SS c_S(t) = h_S(0) + t B_S,j.

Python uses Fraction Gauss-Jordan; the independently written scalar browser implementation uses integer fraction-free Bareiss elimination and BigInt rational back substitution. No epsilon, ridge, singular pseudoinverse, rounding repair or clipping is used. Construction refuses a singular selected Gram block. An empty S is allowed.

The stored numeric coefficients only nominate S by strictly positive entries. The exact re-solved candidate generally differs from those numeric coefficients. If its starting coefficients or inactive gradients have any negative exact sign, the candidate is refused. Numeric convergence alone does not prove an exact face certificate. The browser keeps its last applied witness on errors and says so.

For g(t) = (c(t)B - x(t))B^T, exact stationarity gives g_S(t) identically zero. KKT leaves one affine half-line c_k(t) >= 0 for k in S and g_k(t) >= 0 for k outside S. Intersect every half-line with the pixel domain. The result is the maximal closed interval for THIS affine candidate. Starting KKT makes zero belong to it. Singletons are valid; zero slopes and permanently tight forms are not discarded from the witness. Every nonconstant limiting inequality and any pixel-domain tie are listed at each endpoint.

Closed endpoint membership is weak KKT. Selected coefficients may be zero and unselected gradients may be zero. The nominated working face is not the strict positive support throughout the closed interval. The implementation does not cross an endpoint, continue to a neighboring face, or claim maximal support stability.

## Why KKT is enough here

For any feasible z >= 0,

    f(z) - f(c) = g(c) dot (z-c) + 1/2 ||(z-c)B||².

Every selected gradient is zero; every unselected coefficient is zero and its gradient is nonnegative. Thus the first term is nonnegative, as is the squared norm. The candidate is globally optimal for this fixed-basis NNLS problem wherever all its exact KKT conditions hold. This proof does not require a nonsingular full Gram matrix or unique coefficients. A valid affine candidate on a singular face can also be checked by the verifier, although this constructor refuses to solve that singular face.

## Witness and verifier

`face.py:verify` recomputes stationarity, unselected zeros, all affine forms, starting KKT, the full interval and all endpoint ties directly. It does not rerun the linear solve. `docs/face.js:verify` does the same with BigInt arithmetic. Reduced fractions use decimal-string numerator and positive denominator fields, preserving large integers across JSON. Both check metadata types, including Boolean-vs-integer distinctions.

This is a mathematical witness for its embedded problem, not a source signature, authenticity proof or hostile-input security boundary. Native export records the model and candidate digests and exact source contexts; tests independently bind the browser bundle's bases, pixels and nomination coefficients to those package sources. Imported witnesses can describe unrelated problems. Public JSON parsing rejects duplicate keys and nonfinite constants for the new CLI commands. Input sizes are bounded to 256 pixels and 64 components; rational integer strings in verifiers are limited to 10,000 characters. Extremely large rational operations can still be expensive. The browser performs construction/verification in a disposable Worker, discards stale results, and never silently falls back to floating point.

## Controls

- `x=(1/2,1/2)`, rows `(1,1),(1,0)`, S={0}, pixel0: interval [-1/2,0]. The unused gradient binds at zero. A valid endpoint need not have a positive inactive-gradient margin.
- `x=(0,0)`, row `(1,0)`, empty S, pixel0: interval {0}. No artificial widening.
- `x=(1/2)`, duplicate rows `(1),(1)`, S={0}: a valid optimum with nonunique coefficients. Nominating both duplicate rows makes construction refuse a singular Gram block. Verification need not infer uniqueness from either witness.
- `x=(1/2,1/2)`, identity basis, S={0,1}: lower endpoint simultaneously hits the pixel domain and coefficient0. Both ties are retained.

## Reproduce

```sh
python -m pip install .
python -m partslab face --problem examples/face.json --out /tmp/face.json
python -m partslab verify-face --witness /tmp/face.json
python -m partslab face-export --out /tmp/face-reference.json
python -m unittest discover -s tests -v
node --test tests/*.cjs
```

`face-reference.json` contains 54 complete responses: six frozen models x three test rows x three pixel indices (18,27,36), plus four controls. All complete fields are compared against native Fraction outputs by scalar BigInt tests and actual Chrome Worker/UI tests. Source-nomination tests additionally cover all 120 stored row/model contexts. The interactive page offers all 20 stored test rows, six bases and all 64 pixel indices, with the exact check deciding whether a nomination is valid. It does not claim that every possible edited nomination works.

The small-system oracle uses an independent determinant/Cramer enumeration, not the production elimination. Exhaustive 3-component, 2-pixel binary bases yield 200 accepted face/pixel candidates. Exact endpoint/interior objectives agree with exhaustive independent NNLS solutions, and just-outside points violate a candidate inequality whenever still inside the pixel domain. Tests include nonmutation, duplicate/zero bases, empty singleton faces, subnormals, supplied-binary64 vs decimal differences, altered coefficients/limits/metadata, invalid inputs, stale workers, copy denial/success, actual Blob contents, all pixel colors at endpoints, keyboard query and 320/390/1280 layout checks.

## Sources and scope differences

- [Boyd and Vandenberghe, Convex Optimization](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf), textbook background on convex least-squares. The fetched introduction is not used as evidence for unseen later sections.
- [Gould, Parametric quadratic programming, revisited](https://www.numerical.rl.ac.uk/media/reports/gNAGIR20042.pdf), section2: working-set affine solutions, limiting primal/dual conditions and degeneracies. We do not implement its full continuation algorithm.
- [Bemporad, QP via NNLS](https://cse.lab.imtlucca.it/~bemporad/publications/papers/ieeetac-qpnnls.pdf), active-set and numerical-tolerance context. We do not implement that solver or inherit its complexity guarantees.
- [Feller, Johansen and Olaru, combinatorial multi-parametric QP](https://torarnj.folk.ntnu.no/satMatPruning_final.pdf), section2.1: KKT, affine response and critical-region inequalities. Its strict-convexity/LICQ results do not grant uniqueness to our possibly singular full NNLS problem.
- [Stanford convex optimization introduction](https://web.stanford.edu/class/ee364a/lectures/intro.pdf), explicit nonnegative least-squares formulation.
- [SciPy nnls documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html), convex NNLS and numerical active-set KKT background. SciPy is not an implementation dependency.
