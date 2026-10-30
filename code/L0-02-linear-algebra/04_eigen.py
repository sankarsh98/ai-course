"""Eigenvectors: directions a matrix only stretches. Power iteration and uses."""
import numpy as np

rng = np.random.default_rng(0)

# 1. A symmetric 2x2 example by hand: A = [[2,1],[1,2]].
A = np.array([[2.0, 1.0],
              [1.0, 2.0]])
vals, vecs = np.linalg.eigh(A)               # eigh: for symmetric matrices
print("eigenvalues:", vals)
print("eigenvectors (columns):\n", np.round(vecs, 4))
for lam, q in zip(vals, vecs.T):
    print(f"A q = {np.round(A @ q, 4)}  lambda q = {np.round(lam * q, 4)}")
Q, L = vecs, np.diag(vals)
print("A == Q diag(lambda) Q^T:", np.allclose(A, Q @ L @ Q.T),
      "| Q orthogonal:", np.allclose(Q.T @ Q, np.eye(2)))


# 2. Power iteration: repeatedly apply A and renormalize. Converges to the
#    top eigenvector at a rate set by |lambda_2 / lambda_1|.
def power_iteration(M, iters, rng):
    v = rng.normal(size=M.shape[0])
    history = []
    for _ in range(iters):
        v = M @ v
        v /= np.linalg.norm(v)
        history.append(v @ M @ v)            # Rayleigh quotient estimates lambda_1
    return v, history


C = rng.normal(size=(200, 200))
M = C @ C.T / 200                            # symmetric positive semidefinite
true_vals = np.linalg.eigvalsh(M)
v, hist = power_iteration(M, 200, rng)
print(f"\ntop eigenvalue true={true_vals[-1]:.6f}  "
      f"after 10 iters={hist[9]:.6f}  after 200 iters={hist[-1]:.6f}")
print(f"ratio lambda2/lambda1 = {true_vals[-2] / true_vals[-1]:.4f} (closer to 1 = slower)")

# 3. Covariance eigenvectors point along directions of greatest spread.
#    (Centering + this idea is PCA, covered in L1-04.)
cov_true = np.array([[3.0, 1.2],
                     [1.2, 1.0]])
pts = rng.multivariate_normal([0, 0], cov_true, size=5000)
S = np.cov(pts, rowvar=False)
lam, U = np.linalg.eigh(S)
print("\nsample covariance eigenvalues:", np.round(lam, 3))
print("top direction:", np.round(U[:, -1], 3),
      "| variance of data projected on it:", round(float(np.var(pts @ U[:, -1], ddof=1)), 3))

# 4. Markov chain: the stationary distribution is the eigenvector with eigenvalue 1.
T = np.array([[0.9, 0.1, 0.0],               # row-stochastic transition matrix
              [0.2, 0.7, 0.1],
              [0.1, 0.2, 0.7]])
w, V = np.linalg.eig(T.T)                    # left eigenvectors of T
pi = np.real(V[:, np.argmin(np.abs(w - 1))])
pi /= pi.sum()
print("\nstationary distribution:", np.round(pi, 4),
      "| pi T == pi:", np.allclose(pi @ T, pi))
print("eigenvalues of T (eig: complex dtype, unsorted):", np.round(w, 4))
