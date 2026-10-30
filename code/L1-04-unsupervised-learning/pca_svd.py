"""PCA via the SVD, from scratch, checked against scikit-learn.

Shows: explained variance, reconstruction error equals the discarded variance,
agreement with sklearn up to sign, and how unscaled features hijack PCA.
"""
import numpy as np
from sklearn.datasets import load_digits, load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_svd(X, k):
    mean = X.mean(0)
    Xc = X - mean                          # PCA needs centered data
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S**2 / (len(X) - 1)              # eigenvalues of the covariance matrix
    return mean, Vt[:k], var


X = load_digits().data                     # 1797 images, 8x8 pixels = 64 features
n, d = X.shape
mean, W, var = pca_svd(X, 64)
ratio = var / var.sum()
cum = np.cumsum(ratio)
print(f"digits: n={n}, d={d}")
print("explained variance ratio, first 5:", np.round(ratio[:5], 3))
for target in (0.5, 0.9, 0.95, 0.99):
    print(f"  components for {target:.0%} of variance: {np.searchsorted(cum, target) + 1}")

print("\nk   mean sq. reconstruction error   sum of discarded eigenvalues")
Xc = X - mean
for k in (2, 10, 20, 40):
    Z = Xc @ W[:k].T                       # project to k dims
    Xhat = Z @ W[:k]                       # map back to 64 dims
    err = ((Xc - Xhat) ** 2).sum(1).mean()
    print(f"{k:<3} {err:30.2f} {var[k:].sum() * (n - 1) / n:30.2f}")

sk = PCA(n_components=10).fit(X)
# Each component is defined only up to sign, so compare absolute cosines.
cos = np.abs(np.sum(sk.components_ * W[:10], axis=1))
print("\n|cos| between our components and sklearn's:", np.round(cos, 6))

wine_ds = load_wine()
wine, names = wine_ds.data, wine_ds.feature_names
print("\nwine feature std devs range from "
      f"{wine.std(0).min():.2f} to {wine.std(0).max():.1f}")
for label, data in [("raw", wine), ("standardized", StandardScaler().fit_transform(wine))]:
    _, W1, v = pca_svd(data, 1)
    top = np.argmax(np.abs(W1[0]))
    print(f"  {label:13s} PC1 explains {v[0] / v.sum():.1%}; "
          f"largest loading: {names[top]} ({np.abs(W1[0, top]):.3f})")
