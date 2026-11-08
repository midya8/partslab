# Parts without identification

## Scope and fixed protocol

PartsLab v0.1 fits original UCI optdigits pixels (64 integer counts /16) without labels. It preserves the original 3,823 training and 1,797 test rows and source hashes. The loader verifies source hashes/shape/range and deliberately never returns labels. All model fitting reads only training pixels; a test-loader prohibition is exercised during the complete refit. The original files retain their labels for provenance, not fitting.

Fixed ranks 8,16,32; seeds 11,23; random positive initial factors; 100 outer iterations; three cyclic coefficient sweeps then three cyclic basis sweeps; unit-L2-norm nonzero basis rows after each outer iteration, with the inverse scaling applied to coefficients. The joint stopping threshold is `1e-5 * max(1, initial coefficient PG infinity norm, initial basis PG infinity norm)` in this normalization gauge. All six fits exhaust the 100-iteration budget. This is not a claim of stationarity or a preferred production budget.

After fitting, re-solve training coefficients for the frozen basis and select the lowest training MSE seed within each rank, lower seed breaking exact ties. No rank is selected: all three remain visible. Fixed choices are not tuned on the test rows. Before opening the test file, evaluation verifies hashes of the protocol and frozen model artifact. Evaluating changed models or protocol fails before test access. This prevents accidental refitting/selection during evaluation; it does not make the experiment a blinded independent replication.

## Objective and coordinate update

Rows of `X` are images, `A` is the nonnegative row-coefficient matrix, and rows of `B` are nonnegative basis vectors. Minimize

`F(A,B) = (1/2) ||X - A B||_F^2`, subject to `A >= 0, B >= 0`.

Gradients are `(AB-X) B.T` and `A.T (AB-X)`. For fixed B define `G=B B.T`, `C=X B.T`. Holding other coefficient coordinates fixed, the quadratic in coordinate j has curvature `G[j,j]`. Its exact real-arithmetic bounded minimizer is

`a_j <- max(0, a_j - ((a G)_j - C_j) / G[j,j])`.

After the change delta, update the working gradient by `g <- g + delta * G[j,:]`. Later coordinates use the already-updated values (Gauss-Seidel, not simultaneous Jacobi). Computation is binary64; tests bound numerical descent deviations, rather than claiming exact arithmetic. A zero diagonal means the corresponding basis row is exactly zero, its cross product is zero and the direction is flat. Set that coefficient to zero without an epsilon or division. Basis sweeps apply the same routine to `X.T`, with coefficients fixed. Subproblems are only improved by a fixed number of sweeps; we do not pretend to solve every training block exactly or claim a convergence theorem for our particular finite schedule.

For nonnegative variables z and gradient g, KKT requires `g>=0`, `z>=0`, `z*g=0`. Our projected gradient is g where z>0 or g<0, otherwise zero. Its infinity norm is a numerical stationarity residual. Joint residuals are measured after display normalization. Their magnitude depends on the factor scale. An additional complementarity infinity norm is saved. Stationarity alone would still not imply a local minimum or a global NMF solution, because the joint problem is nonconvex.

## Fixed-basis inference

With a frozen B, minimize `(1/2)||x-aB||^2` over `a>=0`. This is convex, although not necessarily strictly convex or coefficient-identifiable. Start at zero, perform cyclic coordinate minimization, and recompute the actual residual gradient `(aB-x) B.T` before each stopping decision. Stop a batch when every row's projected-gradient infinity norm is at most `1e-8 * max(1, max_j |(x B.T)_j|)`, or after 1,000 sweeps. The per-row mask, residual, threshold and total sweeps remain in the report. All saved test rows meet this numerical condition. Batch rows can continue updating until the slowest row finishes; single-input browser solves use the same criterion independently.

A residual threshold is not an exact dual bound or certified objective gap. Ill-conditioned dictionaries and arbitrary inputs can need more sweeps. An unfinished mask is not infeasibility. No regularization, clipping of the optimization target, hidden shrinkage, pretrained dictionary, or label enters inference.

## Nonuniqueness and a boundary control

For any positive diagonal D, `(A D) (D^-1 B) = A B`. A shared permutation also preserves the product. Unit-norm basis normalization fixes a scale convention without eliminating the other ambiguities. Two identical basis rows can represent `[1,0]` with coefficients `[1,0]` or `[0,1]`. Thus even fixed-basis optimal coefficients can be nonunique. We do not claim actual real-data optima are nonunique merely because this control exists.

For target `[1,1]`, identity basis and coefficients `[1,0]`, the plain multiplicative coefficient update `A <- A * (X B.T)/(A B B.T)` keeps the second coordinate zero. Our control leaves a zero-denominator coordinate unchanged, so loss stays 0.5 and its gradient is -1, violating stationarity. Cyclic coordinate minimization moves the coordinate to 1 and attains zero loss. This isolates an actual zero-lock mechanism on the same objective; it is not a performance ranking between tuned solvers. Modified multiplicative methods can address it. Neither the displayed basis nor its coefficient is a causal explanation of image pixels or a guarantee of semantic digit parts.

## Controls and measurements

Training-mean reconstruction is constant. Centered PCA computes an SVD of `X_train - mean_train`; reconstruction is `mean + (x-mean) V_k.T V_k`. Uncentered SVD uses `x V_k.T V_k` from the raw training SVD. Both signed controls use the same k as NMF. PCA has an extra mean vector; the comparison is rank-matched, not parameter matched or equal compute. No whitened coordinates. Test scores use the full raw reconstructed vectors, including negative or >1 entries, with no clamp. Display canvases alone clip. Each basis atlas independently rescales that vector by its maximum and says so; contribution images use a common [0,1] scale. Summing contributions uses the unscaled basis.

The official held-out mean MSE is 0.0735163327. Training-chosen NMF MSEs for ranks8/16/32 are 0.0309683010/0.0163995953/0.0046791296; corresponding PCA MSEs are 0.0251266658/0.0118775534/0.0027689070; SVD MSEs are 0.0257206087/0.0119639018/0.0027681384. Both signed controls are lower on this dataset/protocol. All six per-row scores and solver residuals remain public, including nonselected seeds. No confidence intervals, paired significance, recognition, perceptual quality or general superiority claim. Model and report hashes bind which fixed fits produced the numbers.

## Independent evidence

- Scalar objective and scalar residual-coordinate updates independently match vectorized code.
- Central finite differences check every coefficient/basis partial in a tiny dense problem.
- An independent support-enumeration NNLS oracle solves 35 tiny random problems using least-squares on independent supports, comparing optimal objectives to coordinate inference. It does not reuse Gram-coordinate updates. A conic projection has an independent-support representation, so singular supersets need not be inverted.
- Zero data, zero basis, duplicate bases, scale/permutation, boundary movement, invalid inputs, zero budgets, monotone histories and nonconvergence states are checked.
- Full six-fit reproduction forbids test-loader access. Frozen guards reject mutations before evaluation. All 1,797 holdout/control scores are independently reproduced. PCA orthogonality and residual projection are checked; negative raw reconstructions prove scoring is not silently clamped.
- 120 complete native/JavaScript decompositions agree within 2e-10, including coefficients, reconstructions, signed controls, numerical residuals and sweep counts. Chrome exercises every visible model/example combination, all 2,048 pixels in 32 contribution canvases, custom pixel edits, keyboard, basis controls, copy failures/success, download, expanded layouts and minimum44px targets at320/390/1280 widths.
- 21 Python and four Node/Chrome tests; installed wheel imports seven modules and executes five CLIs, refits all six models and reproduces the holdout outside the checkout. Refit comparisons allow 2e-9 absolute plus2e-8 relative binary64 variation; frozen-file hashes are exact. SVG generation from frozen data is byte-exact. No timing or throughput benchmark.

## Sources

Established algorithms and limits, not algorithmic novelty or a reproduction of a paper's optimized performance:

1. Lee & Seung, [Algorithms for Non-negative Matrix Factorization](https://proceedings.neurips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf), NeurIPS2000: multiplicative-update objective and nonincrease context.
2. Chih-Jen Lin, [Projected Gradient Methods for Non-negative Matrix Factorization](https://www.csie.ntu.edu.tw/~cjlin/papers/pgradnmf.pdf), Neural Computation19(2007), [DOI](https://doi.org/10.1162/neco.2007.19.10.2756): gradients, KKT, scaling, two-block issues and why stopping by objective change is inadequate. We use cyclic scalar-coordinate minimization, not Lin's specific projected-gradient implementation.
3. Chih-Jen Lin, [On the Convergence of Multiplicative Update Algorithms for Nonnegative Matrix Factorization](https://www.csie.ntu.edu.tw/~cjlin/papers/multconv.pdf), IEEE TNN18(2007), [DOI](https://doi.org/10.1109/tnn.2007.895831): original versus modified update convergence distinctions. No blanket statement that all MU methods fail.
4. [scikit-learn decomposition guide](https://scikit-learn.org/stable/modules/decomposition.html), section2.5.7: solver choices, zero-lock and nonconvexity caveats. Documentation reference only, not a runtime dependency.
5. [scikit-learn NMF API](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html): factor objective, normalization conventions and coordinate/multiplicative solver scope. This code does not wrap it.
6. [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), Alpaydin & Kaynak(1998): original source, official split and CC BY4.0 attribution. Original files/hash/license included. Derived bases and summaries are new transformations. MIT original code; no endorsement.
