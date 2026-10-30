"""EM for a Gaussian mixture, from scratch, on stretched clusters where k-means fails.

Shows: log-likelihood rises every iteration, a full-covariance GMM recovers
elongated clusters that k-means cuts wrongly, and BIC picks the number of
components.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

rng = np.random.default_rng(0)


def make_stretched(n_per=300):
    # Three long, thin, parallel clusters: close in one direction, far in the other.
    base = rng.normal(size=(3 * n_per, 2)) * [4.0, 0.4]
    shift = np.repeat([[0, 0], [0, 2.5], [0, 5]], n_per, axis=0)
    y = np.repeat([0, 1, 2], n_per)
    return base + shift, y


def em_gmm(X, k, n_iter=200, tol=1e-6, seed=0):
    n, d = X.shape
    r = np.random.default_rng(seed)
    mu = X[r.choice(n, k, replace=False)]
    cov = np.array([np.cov(X.T) for _ in range(k)])
    pi = np.full(k, 1.0 / k)
    lls = []
    for _ in range(n_iter):
        # E step: log p(x, z=j) for each point and component, then normalize.
        log_joint = np.column_stack([
            np.log(pi[j]) + multivariate_normal(mu[j], cov[j]).logpdf(X)
            for j in range(k)])
        log_px = logsumexp(log_joint, axis=1)      # log-sum-exp avoids underflow
        resp = np.exp(log_joint - log_px[:, None])  # responsibilities, rows sum to 1
        lls.append(log_px.sum())
        if len(lls) > 1 and lls[-1] - lls[-2] < tol:
            break
        # M step: weighted means, covariances, mixing weights.
        Nk = resp.sum(0)
        pi = Nk / n
        mu = (resp.T @ X) / Nk[:, None]
        for j in range(k):
            diff = X - mu[j]
            cov[j] = (resp[:, j, None] * diff).T @ diff / Nk[j] + 1e-6 * np.eye(d)
    return resp.argmax(1), lls


X, y = make_stretched()
labels, lls = em_gmm(X, 3)
print(f"EM iterations: {len(lls)}")
print("log-likelihood, first 6:", ", ".join(f"{v:.1f}" for v in lls[:6]))
print(f"log-likelihood, final : {lls[-1]:.1f}")
print("monotone non-decreasing:", all(b >= a - 1e-8 for a, b in zip(lls, lls[1:])))

# EM finds a local optimum too, so use several restarts and keep the best.
best = max((em_gmm(X, 3, seed=s) for s in range(5)), key=lambda t: t[1][-1])
km = KMeans(3, n_init=10, random_state=0).fit(X)
print(f"\nARI, our GMM (best of 5 restarts): {adjusted_rand_score(y, best[0]):.3f}")
print(f"ARI, k-means                     : {adjusted_rand_score(y, km.labels_):.3f}")

print("\nChoosing the number of components by BIC (lower is better).")
print("scikit-learn's default init runs k-means first, which cuts these clusters")
print("the wrong way, so compare it with initializing from random data points.")
print("      init='kmeans'  init='random_from_data'")
for k in range(1, 7):
    bics = [GaussianMixture(k, n_init=10, init_params=init, random_state=0)
            .fit(X).bic(X) for init in ("kmeans", "random_from_data")]
    print(f"  k={k}  {bics[0]:8.1f}       {bics[1]:8.1f}")
