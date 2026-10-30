"""Hierarchical clustering, what internal metrics reward, and anomaly detection.

Part 1: linkage choice on curved clusters, scored against known labels (ARI)
        and by the silhouette, which needs no labels.
Part 2: four anomaly detectors on data with planted anomalies.
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import make_blobs, make_moons
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, average_precision_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor

# Part 1 -------------------------------------------------------------------
X, y = make_moons(n_samples=600, noise=0.06, random_state=0)
print("Two moons, 2 clusters        ARI (vs truth)   silhouette (no labels)")
models = {f"agglomerative, {link:8s}": AgglomerativeClustering(2, linkage=link)
          for link in ("single", "complete", "average", "ward")}
models["k-means"] = KMeans(2, n_init=10, random_state=0)
for name, m in models.items():
    labels = m.fit_predict(X)
    print(f"  {name:26s} {adjusted_rand_score(y, labels):6.3f}  "
          f"{silhouette_score(X, labels):14.3f}")

# Single linkage chains through noise: add a few bridge points between moons.
bridge = np.column_stack([np.linspace(0.0, 1.0, 12), np.full(12, 0.25)])
Xb = np.vstack([X, bridge])
lab = AgglomerativeClustering(2, linkage="single").fit_predict(Xb)
print(f"  single linkage after adding 12 bridge points: cluster sizes "
      f"{np.bincount(lab).tolist()}")

# Part 2 -------------------------------------------------------------------
rng = np.random.default_rng(0)
normal, _ = make_blobs(n_samples=2000, centers=[[0, 0], [6, 6], [0, 8]],
                       cluster_std=[0.6, 1.5, 0.6], random_state=0)
outliers = rng.uniform(-4, 10, size=(40, 2))
Xa = np.vstack([normal, outliers])
is_anom = np.r_[np.zeros(len(normal)), np.ones(len(outliers))]

# Higher score = more anomalous, for every method.
scores = {
    "Mahalanobis (1 Gaussian)": EmpiricalCovariance().fit(Xa).mahalanobis(Xa),
    "GMM, 3 comps (-log p)": -GaussianMixture(3, random_state=0).fit(Xa).score_samples(Xa),
    "Isolation Forest": -IsolationForest(random_state=0).fit(Xa).score_samples(Xa),
    "Local Outlier Factor": -LocalOutlierFactor(n_neighbors=20).fit(Xa).negative_outlier_factor_,
}
k = len(outliers)
print(f"\nAnomaly detection: {len(normal)} normal points, {k} planted anomalies")
print("  (some uniform 'anomalies' land inside normal clusters, so 1.0 is not reachable)")
print("  method                      precision@40   average precision")
for name, s in scores.items():
    top = np.argsort(-s)[:k]
    print(f"  {name:27s} {is_anom[top].mean():10.2f}   {average_precision_score(is_anom, s):12.2f}")

# Why 1.0 is out of reach: how many "anomalies" sit right next to normal points?
near = (cdist(outliers, normal).min(1) < 0.5).sum()
print(f"  planted anomalies within 0.5 of a normal point: {near} of {k}")
