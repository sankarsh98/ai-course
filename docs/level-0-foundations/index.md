# Level 0: Foundations

The math and tools that everything else uses: arrays and floating point, linear algebra, automatic differentiation, probability and statistics, and optimization. Each module is built around code you run, including a small autograd engine you write yourself.

| ID | Module | Prerequisites | Time | Scripts |
|---|---|---|---|---|
| L0-01 | Python and NumPy for ML | none | 5 to 6 hours | 6 |
| L0-02 | Linear Algebra for ML | L0-01 | 6 to 8 hours | 6 |
| L0-03 | Calculus and Automatic Differentiation | L0-01, L0-02 | 5 to 7 hours | 5 |
| L0-04 | Probability and Statistics | L0-01 | 6 to 8 hours | 7 |
| L0-05 | Optimization | L0-01, L0-02, L0-03 | 5 to 7 hours | 4 |

## What each module teaches

### L0-01: Python and NumPy for ML

- Predict the shape, dtype and memory cost of an array expression, and say whether a result is a view or a copy.
- Apply broadcasting rules by hand, and catch the shape bugs that broadcasting hides instead of reporting.
- Replace Python loops with vectorized NumPy, and measure the speedup honestly with `timeit` and `cProfile`.
- Make experiments reproducible with explicit random generators.

### L0-02: Linear Algebra for ML

- Compute and interpret dot products, norms, angles and projections, and explain how high-dimensional geometry differs from 2D intuition.
- Read a matrix as a linear map, predict the cost of a chain of products, and choose the cheaper evaluation order.
- Find rank and null space, derive least squares as an orthogonal projection, and explain why ill-conditioning makes weights unstable.
- Compute eigendecompositions and SVDs, implement power iteration, and use the Eckart-Young theorem to build and evaluate low-rank approximations.

### L0-03: Calculus and Automatic Differentiation

- Explain derivatives, gradients and Jacobians as local linear approximations, and compute them by hand for small functions.
- Apply the chain rule to compositions, including the multivariable case where it becomes a product of Jacobians.
- Estimate derivatives numerically, pick a sensible step size, and explain why a tiny step makes the answer worse.
- Run forward-mode and reverse-mode differentiation by hand on a computational graph, and say which is cheaper for a given problem.

### L0-04: Probability and Statistics

- Describe common distributions by their mean, variance and typical ML use, and check those properties by simulation.
- Apply Bayes' rule, and derive maximum likelihood and MAP estimates for simple models.
- Compute entropy, cross-entropy and KL divergence, and explain why the cross-entropy loss is maximum likelihood.
- Estimate expectations by Monte Carlo, predict how the error shrinks, and recognize when importance sampling fails.

### L0-05: Optimization

- Predict from a loss's curvature which learning rates converge, oscillate or diverge.
- Explain how the condition number sets the speed of gradient descent, and why momentum helps.
- Implement SGD, momentum, RMSProp, Adam and AdamW from scratch and test claims about them.
- Recognise convex problems, local minima and saddle points, and use a backtracking line search.
