# Unsupervised Learning

> **Level 1 · Module 4** · Prerequisites: L0-02, L0-04, L1-01 · Time: 5 to 6 hours · Code: `code/L1-04-unsupervised-learning/`

## What you will be able to do

- Derive and implement k-means and EM for Gaussian mixtures, and explain why each improves monotonically yet gets stuck in local optima.
- Derive PCA from the SVD, compute the variance a projection keeps, and predict when unscaled features will dominate it.
- Choose a hierarchical linkage for the cluster shapes you expect, and recognize chaining.
- Read a t-SNE or UMAP plot without over-reading it.
- Build anomaly detectors and evaluate unsupervised results with internal scores, external labels and stability, knowing what each rewards.

## 1. What "unsupervised" means, and why it is slippery

You get a matrix $X \in \mathbb{R}^{n \times d}$ ($n$ rows, $d$ features) and no labels. The common jobs are **clustering** (group similar rows), **dimensionality reduction** (describe each row with $k \ll d$ numbers) and **density estimation and anomaly detection** (learn what typical rows look like and flag the rest).

The data doesn't define "structure". Each method brings its own definition. k-means says a cluster is a ball around a center. Single linkage says a cluster is anything you can walk through in small steps. PCA says important directions are high-variance ones. When two methods disagree, they are usually answering different questions. Most of the skill here is knowing which question each method answers.

## 2. k-means

Pick $k$ centers $\mu_1, \dots, \mu_k$ and assign each point $x_i$ to a cluster $z_i \in \{1,\dots,k\}$, minimizing the within-cluster sum of squares (scikit-learn calls it *inertia*):

$$J(z, \mu) = \sum_{i=1}^{n} \lVert x_i - \mu_{z_i} \rVert^2 .$$

The exact problem is NP-hard, as Arthur and Vassilvitskii note, so everyone uses a local search usually credited to Lloyd, who derived it for quantizing signals [Lloyd, 1982]. **Lloyd's algorithm** alternates:

1. **Assign**: with centers fixed, $z_i = \arg\min_j \lVert x_i - \mu_j \rVert^2$.
2. **Update**: with assignments fixed, set each $\mu_j$ to the mean of its points.

**Why it never gets worse.** Each step minimizes $J$ exactly over one block of variables. The nearest center is optimal for each point. The mean minimizes $\sum_i \lVert x_i - m\rVert^2$, since the gradient $-2\sum_i (x_i - m)$ vanishes there. So $J$ never increases, and since there are finitely many assignments, the algorithm stops, at a *local* minimum that can be far from the best.

**Seeding: k-means++.** If two initial centers land in one true cluster, Lloyd's algorithm rarely separates them again. k-means++ picks the first center uniformly, then each next center with probability proportional to $D(x)^2$, the squared distance to the nearest center chosen so far. Arthur and Vassilvitskii proved this gives an expected objective within $O(\log k)$ of optimal, and found it made k-means faster and more accurate in experiments [Arthur & Vassilvitskii, 2007]. scikit-learn uses it by default, with several restarts.

**Assumptions.** Nearest-center assignment cuts space into convex cells, so k-means suits round, similarly spread, well-separated clusters, and it depends on feature scale. The scikit-learn guide states the same: inertia assumes convex, isotropic clusters [scikit-learn clustering guide]. $J$ always falls as $k$ grows (it is zero at $k = n$), so it can't choose $k$. Section 8 covers what can.

## 3. Gaussian mixtures and EM

A **Gaussian mixture model** (GMM) generates a point by picking component $j$ with probability $\pi_j$, then drawing $x \sim \mathcal{N}(\mu_j, \Sigma_j)$:

$$p(x) = \sum_{j=1}^{k} \pi_j \, \mathcal{N}(x \mid \mu_j, \Sigma_j), \qquad \sum_j \pi_j = 1 .$$

Each cluster gets its own covariance $\Sigma_j$ (so it can be stretched and tilted) and weight $\pi_j$, and each point gets a soft assignment.

The log-likelihood $\sum_i \log p(x_i)$ has a sum inside the log and no closed-form maximizer. If we knew which component made each point, fitting would be easy. **Expectation-maximization** (EM) [Dempster et al., 1977] alternates between guessing that and refitting.

**E step.** The **responsibility** of component $j$ for point $i$ is its posterior probability (Bayes' rule):

$$r_{ij} = \frac{\pi_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}{\sum_{l=1}^{k} \pi_l \, \mathcal{N}(x_i \mid \mu_l, \Sigma_l)} .$$

**M step.** With $N_j = \sum_i r_{ij}$, refit weighted Gaussians:

$$\pi_j = \frac{N_j}{n}, \qquad \mu_j = \frac{1}{N_j}\sum_i r_{ij}\, x_i, \qquad \Sigma_j = \frac{1}{N_j}\sum_i r_{ij}\,(x_i-\mu_j)(x_i-\mu_j)^\top .$$

Dempster, Laird and Rubin showed an EM iteration never decreases the likelihood. Intuitively, the E step builds a lower bound that touches the log-likelihood at the current parameters, and the M step maximizes it. Like k-means, EM finds a local optimum. With every $\Sigma_j = \sigma^2 I$, equal weights and $\sigma \to 0$, responsibilities become hard nearest-center assignments and EM becomes k-means.

**Traps.** (1) *Collapse*: a component that shrinks onto one point has a likelihood going to infinity, so implementations add a small constant to each covariance diagonal (scikit-learn's `reg_covar` defaults to $10^{-6}$). (2) *Choosing $k$*: the likelihood always improves with more components. The **Bayesian information criterion**, $\text{BIC} = -2 \log \hat{L} + p \log n$ with $p$ free parameters, penalizes complexity, and lower is better. BIC inherits the local-optimum problem, as the hands-on shows.

## 4. Hierarchical clustering

**Agglomerative clustering** starts with each point alone and repeatedly merges the two closest clusters. The merge tree (*dendrogram*) can be cut at any height, so you needn't fix $k$ first. The **linkage** defines cluster distance:

| Linkage | Distance between clusters $A$, $B$ | Tends to find |
|---|---|---|
| single | closest pair | arbitrary shapes, but chains through noise |
| complete | farthest pair | compact clusters of similar diameter |
| average | mean over all pairs | a compromise |
| Ward | increase in within-cluster sum of squares | round clusters, like k-means (scikit-learn's default) |

Single linkage follows curved clusters because it only needs a chain of close neighbors. The same property means a thin trail of noise can merge two clusters (**chaining**). Storing all pairwise distances costs $O(n^2)$ memory.

## 5. Principal component analysis

PCA finds the directions of greatest variance and describes each point by its coordinates along the top $k$.

**Derivation.** Center the data, $\tilde{X} = X - \mathbf{1}\bar{x}^\top$, with covariance $S = \tilde{X}^\top \tilde{X}/(n-1)$. Projecting onto a unit vector $w$ gives variance $w^\top S w$. Maximizing subject to $\lVert w \rVert = 1$ with a Lagrange multiplier $\lambda$ gives

$$S w = \lambda w ,$$

so $w$ is an eigenvector and its captured variance is the eigenvalue $\lambda$. The next direction, orthogonal to the first, is the second eigenvector, and so on.

Compute it from the SVD (L0-02) without forming $S$. With $\tilde{X} = U \Sigma V^\top$,

$$S = \frac{V \Sigma U^\top U \Sigma V^\top}{n-1} = V \, \frac{\Sigma^2}{n-1} \, V^\top ,$$

so the columns of $V$ are the principal directions and $\lambda_j = \sigma_j^2/(n-1)$, where $\sigma_j$ are the singular values. This is numerically safer, because forming $\tilde{X}^\top \tilde{X}$ squares the condition number.

**What you lose.** Reconstruct with the top $k$ directions, $\hat{x} = \bar{x} + V_k V_k^\top (x - \bar{x})$. By Eckart-Young (L0-02), no rank-$k$ linear map does better, and the mean squared error per point is the sum of the discarded eigenvalues (times $(n-1)/n$). The **explained variance ratio** $\lambda_j / \sum_l \lambda_l$ is each direction's share.

**Watch for.** *Scale*: variance depends on units, so standardize unless shared units make scale meaningful. *Variance is not importance*: a low-variance direction can carry the signal a downstream task needs. *Linearity*: curved data needs more components than its intrinsic dimension. *Signs*: $w$ and $-w$ are both valid, so fix signs before comparing runs.

## 6. t-SNE and UMAP, and how they mislead

**t-SNE** [van der Maaten & Hinton, 2008] preserves *neighborhoods* rather than distances. In the original space, point $i$ picks neighbor $j$ with probability proportional to $\exp(-\lVert x_i - x_j\rVert^2 / 2\sigma_i^2)$. Each bandwidth $\sigma_i$ is set so the distribution has a chosen **perplexity**, roughly the effective number of neighbors. These are symmetrized into $p_{ij}$. In 2-D, similarities $q_{ij}$ use a heavy-tailed Student-t kernel $(1 + \lVert y_i - y_j \rVert^2)^{-1}$, and gradient descent moves the points $y_i$ to minimize

$$\mathrm{KL}(P \,\Vert\, Q) = \sum_{i \ne j} p_{ij} \log \frac{p_{ij}}{q_{ij}} .$$

The heavy tail lets moderately distant points sit far apart, reducing crowding. The KL divergence heavily punishes separating true neighbors and barely punishes placing non-neighbors close. So t-SNE keeps local neighborhoods and is largely indifferent to the rest.

**UMAP** [McInnes et al., 2018] builds a weighted nearest-neighbor graph, motivated by topology, and optimizes a 2-D layout to match it under a cross-entropy loss. Its authors describe it as competitive with t-SNE visually and "arguably" better at global structure, with faster run time. It is still a neighbor-graph layout with tunable knobs (`n_neighbors`, `min_dist`). This module's code uses only t-SNE. The reading rules below apply to both.

**Reading rules**, from Wattenberg, Viégas and Johnson's interactive article [Wattenberg et al., 2016], each reproduced with numbers in the hands-on:

1. Cluster sizes mean nothing: dense clusters expand and sparse ones contract.
2. Distances between clusters may mean nothing.
3. Random noise can look clustered, especially at low perplexity.
4. Hyperparameters really matter: check several perplexities and seeds.

Chari and Pachter go further for single-cell genomics [Chari & Pachter, 2023]. They show that the embeddings substantially distort neighborhoods and distances, and conclude they are unreliable even for exploration. Many practitioners still use them as a first look. A reasonable middle position is to use the plots for hypotheses, then test each hypothesis in the original space.

![Four t-SNE embeddings: (a) a tight and a diffuse cluster drawn at the same size; (b) three clusters where C is ten times farther from A than B is, drawn roughly evenly spaced; (c) pure Gaussian noise at perplexity 2 broken into small clumps; (d) the same noise at perplexity 30 as an even cloud.](../assets/L1-04-unsupervised-learning/tsne_pitfalls.png)

## 7. Anomaly detection

Detectors score each point, and you choose the threshold. There are three families:

- **Density**: fit $p(x)$ and score by $-\log p(x)$. A single Gaussian gives the Mahalanobis distance $\sqrt{(x-\mu)^\top \Sigma^{-1} (x-\mu)}$, which suits one elliptical blob and fails on multimodal data.
- **Neighbors**: the **local outlier factor** (LOF) compares a point's local density with its neighbors' densities.
- **Isolation**: **Isolation Forest** [Liu et al., 2008] grows random trees with random splits. Anomalies are few and different, so they isolate in fewer splits. It needs no distance metric and runs in linear time with little memory.

The threshold is an operational decision (how many alerts can someone review?). A `contamination` setting just turns a guessed anomaly rate into a threshold. And rare is not the same as interesting: many outliers are data errors or new legitimate behavior. A few labeled anomalies are worth a lot for evaluation.

## 8. Evaluating unsupervised results

No single number says a clustering is right. Combine kinds of evidence:

- **Internal indices** use only the data. The **silhouette** of a point [Rousseeuw, 1987] is $s = (b - a)/\max(a, b)$, where $a$ is its mean distance to its own cluster and $b$ to the nearest other cluster. It runs from -1 to 1 and is averaged over points. Silhouette, inertia and BIC all encode a definition of a good cluster (usually compact and separated) and favor methods that share it.
- **External indices** compare with known labels, for example on a small audited subset. The **adjusted Rand index** (ARI) scores pairwise agreement, corrected so that random labelings score about 0 and identical partitions 1, whatever the label names.
- **Stability**: rerun with new seeds or subsamples and measure agreement. Structure that vanishes under resampling probably wasn't there.
- **Usefulness**: do the segments behave differently, do PCA features help a properly cross-validated model (L1-01), do analysts find the flagged anomalies worth reviewing?

## Hands-on

Run each script from the repository root with `.venv/bin/python code/L1-04-unsupervised-learning/<script>.py`. Each takes a few seconds. The output below came from scikit-learn 1.9.1 and NumPy 2.5.4. Last digits may differ on other platforms or versions.

### Script 1: k-means from scratch

```python
"""k-means (Lloyd's algorithm) from scratch, with random and k-means++ seeding.

Shows: the objective never increases, different seeds land in different local
minima, k-means++ seeding helps, and our result matches scikit-learn.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


def sq_dists(X, C):
    # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2, computed for all pairs at once.
    # Rounding can make this slightly negative when x == c, so clip at 0.
    d2 = (X**2).sum(1)[:, None] - 2 * X @ C.T + (C**2).sum(1)[None, :]
    return np.maximum(d2, 0.0)


def init_random(X, k, rng):
    return X[rng.choice(len(X), size=k, replace=False)]


def init_kmeanspp(X, k, rng):
    # First center uniform; each next center sampled with prob. proportional
    # to squared distance from the nearest center chosen so far.
    centers = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = sq_dists(X, np.array(centers)).min(1)
        centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(centers)


def lloyd(X, C, max_iter=100):
    history = []
    for _ in range(max_iter):
        d2 = sq_dists(X, C)
        labels = d2.argmin(1)                     # assignment step
        history.append(d2[np.arange(len(X)), labels].sum())
        newC = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j]
                         for j in range(len(C))])  # update step
        if np.allclose(newC, C):
            break
        C = newC
    return C, labels, history


X, y = make_blobs(n_samples=1500, centers=10, cluster_std=0.6,
                  center_box=(-10, 10), random_state=3)
k = 10

_, _, hist = lloyd(X, init_random(X, k, np.random.default_rng(0)))
print("Objective per iteration (seed 0, random init):")
print("  " + ", ".join(f"{h:.0f}" for h in hist))
print("  never increases:", all(b <= a + 1e-9 for a, b in zip(hist, hist[1:])))

for name, init in [("random", init_random), ("k-means++", init_kmeanspp)]:
    finals, aris = [], []
    for seed in range(50):
        C, labels, hist = lloyd(X, init(X, k, np.random.default_rng(seed)))
        finals.append(hist[-1])
        aris.append(adjusted_rand_score(y, labels))
    finals = np.array(finals)
    best = finals.min()
    print(f"\n{name} init, 50 seeds:")
    print(f"  final objective  min {best:.0f}  median {np.median(finals):.0f}"
          f"  max {finals.max():.0f}")
    print(f"  runs within 1% of best: {(finals <= best * 1.01).sum()}/50")
    print(f"  ARI vs true labels: median {np.median(aris):.3f}  min {min(aris):.3f}")

sk = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
print(f"\nscikit-learn KMeans (k-means++, n_init=10): inertia {sk.inertia_:.0f}"
      f"  ARI {adjusted_rand_score(y, sk.labels_):.3f}")
```

```text
Objective per iteration (seed 0, random init):
  7677, 2676, 1879, 1844, 1840, 1835, 1832, 1831, 1829, 1827, 1823, 1820, 1817, 1814, 1810, 1808, 1808, 1807, 1807, 1807, 1806, 1806
  never increases: True

random init, 50 seeds:
  final objective  min 970  median 1807  max 14994
  runs within 1% of best: 2/50
  ARI vs true labels: median 0.727  min 0.499

k-means++ init, 50 seeds:
  final objective  min 969  median 1006  max 2096
  runs within 1% of best: 5/50
  ARI vs true labels: median 0.834  min 0.663

scikit-learn KMeans (k-means++, n_init=10): inertia 967  ARI 0.863
```

The objective falls fast and then crawls, never increasing. With random seeding, final objectives across 50 seeds span more than an order of magnitude, and only 2 runs come within 1% of the best. k-means++ narrows the spread a lot but guarantees nothing, which is why libraries restart. Even the best solution has ARI 0.863, not 1: some blobs overlap, so the lowest-inertia partition isn't the generating one. My first version crashed in k-means++ with "Probabilities are not non-negative". The fast distance expansion can produce tiny negative values from rounding (L0-01), hence the clip.

### Script 2: EM for a Gaussian mixture

```python
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
```

```text
EM iterations: 146
log-likelihood, first 6: -4602.4, -4423.3, -4400.8, -4365.8, -4299.6, -4252.6
log-likelihood, final : -3951.4
monotone non-decreasing: True

ARI, our GMM (best of 5 restarts): 0.997
ARI, k-means                     : 0.003

Choosing the number of components by BIC (lower is better).
scikit-learn's default init runs k-means first, which cuts these clusters
the wrong way, so compare it with initializing from random data points.
      init='kmeans'  init='random_from_data'
  k=1    8947.1         8947.1
  k=2    8987.1         8552.0
  k=3    9023.0         8018.4
  k=4    8056.5         8056.5
  k=5    8086.3         8089.7
  k=6    8124.7         8120.4
```

The log-likelihood rises every iteration. The clusters are long horizontally and stacked 2.5 units apart vertically. k-means cuts them into vertical slices (ARI 0.003, chance level), while the full-covariance GMM recovers them. The BIC table shows a surprise I didn't plan. With scikit-learn's default `init_params="kmeans"`, BIC picks $k = 4$ and the $k = 3$ fit scores worse than $k = 1$, because the k-means initialization drops EM into the wrong basin. Initializing from random data points finds the right solution, and BIC then picks 3. A selection criterion is only as good as the fits behind it.

### Script 3: PCA via the SVD

```python
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
```

```text
digits: n=1797, d=64
explained variance ratio, first 5: [0.149 0.136 0.118 0.084 0.058]
  components for 50% of variance: 5
  components for 90% of variance: 21
  components for 95% of variance: 29
  components for 99% of variance: 41

k   mean sq. reconstruction error   sum of discarded eigenvalues
2                           858.94                         858.94
10                          314.51                         314.51
20                          126.99                         126.99
40                           14.17                          14.17

|cos| between our components and sklearn's: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]

wine feature std devs range from 0.12 to 314.0
  raw           PC1 explains 99.8%; largest loading: proline (1.000)
  standardized  PC1 explains 36.2%; largest loading: flavanoids (0.423)
```

Digits need 21 of 64 components for 90% of the variance. Reconstruction error equals the discarded eigenvalue sum, confirming the derivation, and the components match scikit-learn's up to sign. On raw wine data, proline (std 314) *is* PC1 and claims 99.8% of the variance. After standardizing, PC1 explains 36.2% and mixes several measurements.

### Script 4: measuring how t-SNE misleads

```python
"""Measure how t-SNE distorts cluster sizes, between-cluster distances and noise.

Every effect is printed as a number, and a 2x2 figure is saved for the lesson.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

rng = np.random.default_rng(0)
D = 10                                    # dimension of the synthetic data
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
OUT = Path(__file__).resolve().parents[2] / "docs/assets/L1-04-unsupervised-learning"


def tsne(X, perplexity, seed=0):
    return TSNE(perplexity=perplexity, init="pca", random_state=seed).fit_transform(X)


def spread(P):
    # Root-mean-square distance of points from their cluster centroid.
    return np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean())


def knn_kept(X, E, k=10):
    # Fraction of each point's k nearest neighbors that stay neighbors in E.
    a = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[1][:, 1:]
    b = NearestNeighbors(n_neighbors=k + 1).fit(E).kneighbors(E)[1][:, 1:]
    return np.mean([len(set(r) & set(s)) / k for r, s in zip(a, b)])


# 1. Cluster sizes: one tight cluster (std 1) and one diffuse cluster (std 10).
tight = rng.normal(0, 1, (200, D))
wide = rng.normal(0, 10, (200, D)) + 100
E1 = tsne(np.vstack([tight, wide]), 30)
print("1. cluster size")
print(f"   spread ratio wide/tight, original 10-D: {spread(wide) / spread(tight):.1f}")
print(f"   spread ratio wide/tight, t-SNE 2-D    : {spread(E1[200:]) / spread(E1[:200]):.1f}")

# 2. Between-cluster distances: A and B are close, C is 10x farther from A.
centers = np.zeros((3, D))
centers[1, 0], centers[2, 0] = 10, 100
X2 = np.vstack([rng.normal(c, 1, (100, D)) for c in centers])
print("\n2. between-cluster distance, ratio dist(A,C)/dist(A,B); original = 10.0")
E2 = {}
for p in (5, 30, 100):
    E2[p] = tsne(X2, p)
    m = [E2[p][i * 100:(i + 1) * 100].mean(0) for i in range(3)]
    print(f"   perplexity {p:>3}: {np.linalg.norm(m[0] - m[2]) / np.linalg.norm(m[0] - m[1]):.1f}")

# 3. Pure noise: one Gaussian blob, no clusters at all.
noise = rng.normal(size=(300, D))
E3 = {p: tsne(noise, p) for p in (2, 30)}
print("\n3. pure Gaussian noise: plots saved (see figure, panels c and d)")

# 4. Local vs global structure on real data (digits, 64-D).
X4 = load_digits().data
dist_hi = pdist(X4)
print("\n4. digits: neighbors kept and global distance correlation")
for name, E in [("PCA 2-D", PCA(2).fit_transform(X4)), ("t-SNE 2-D", tsne(X4, 30))]:
    rho = spearmanr(dist_hi, pdist(E))[0]
    print(f"   {name:9s}: 10-NN kept {knn_kept(X4, E):.2f}, "
          f"Spearman corr. of all pairwise distances {rho:.2f}")

# Figure: what a reader would see, side by side.
fig, ax = plt.subplots(2, 2, figsize=(9, 8), facecolor="#fcfcfb")
panels = [
    (E1, [0, 200, 400], "(a) std 1 vs std 10: look the same size"),
    (E2[30], [0, 100, 200, 300], "(b) A to C is 10x A to B in 10-D; perplexity 30"),
    (E3[2], [0, 300], "(c) pure noise, perplexity 2: fake clumps"),
    (E3[30], [0, 300], "(d) pure noise, perplexity 30"),
]
for a, (E, cuts, title) in zip(ax.flat, panels):
    for i in range(len(cuts) - 1):
        color = "#8a8985" if len(cuts) == 2 else COLORS[i]
        a.scatter(*E[cuts[i]:cuts[i + 1]].T, s=8, color=color, linewidths=0)
    a.set_title(title, fontsize=10, color="#0b0b0b", loc="left")
    a.set_xticks([]), a.set_yticks([])
    a.set_facecolor("#fcfcfb")
    for s in a.spines.values():
        s.set_color("#d0cfca")
# Direct labels so identity never depends on color alone.
for E, n, labels, a in [(E1, 200, ["std 1", "std 10"], ax[0, 0]),
                        (E2[30], 100, ["A", "B", "C"], ax[0, 1])]:
    for i, label in enumerate(labels):
        a.annotate(label, E[i * n:(i + 1) * n].mean(0), fontsize=11,
                   color="#0b0b0b", weight="bold", ha="center")
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "tsne_pitfalls.png", dpi=110)
print(f"\nsaved {OUT.name}/tsne_pitfalls.png")
```

```text
1. cluster size
   spread ratio wide/tight, original 10-D: 10.0
   spread ratio wide/tight, t-SNE 2-D    : 1.0

2. between-cluster distance, ratio dist(A,C)/dist(A,B); original = 10.0
   perplexity   5: 1.0
   perplexity  30: 1.0
   perplexity 100: 3.9

3. pure Gaussian noise: plots saved (see figure, panels c and d)

4. digits: neighbors kept and global distance correlation
   PCA 2-D  : 10-NN kept 0.12, Spearman corr. of all pairwise distances 0.58
   t-SNE 2-D: 10-NN kept 0.59, Spearman corr. of all pairwise distances 0.51

saved L1-04-unsupervised-learning/tsne_pitfalls.png
```

These numbers match the figure in Section 6. A cluster with 10 times the spread is drawn at the same size. C is 10 times farther from A than B is, yet at perplexities 5 and 30 the map spaces them equally. At perplexity 100 (a third of the data per neighborhood) the ratio only recovers to 3.9. Pure noise at perplexity 2 breaks into clumps. On digits, t-SNE keeps 59% of 10-nearest neighbors against PCA's 12%, but its rank correlation with all original distances is lower (0.51 vs 0.58). It keeps local structure better and global geometry worse, which is the trade its objective makes.

### Script 5: linkage, internal metrics and anomaly detectors

```python
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
```

```text
Two moons, 2 clusters        ARI (vs truth)   silhouette (no labels)
  agglomerative, single       1.000           0.331
  agglomerative, complete     0.383           0.474
  agglomerative, average      0.196           0.412
  agglomerative, ward         0.547           0.433
  k-means                     0.255           0.486
  single linkage after adding 12 bridge points: cluster sizes [611, 1]

Anomaly detection: 2000 normal points, 40 planted anomalies
  (some uniform 'anomalies' land inside normal clusters, so 1.0 is not reachable)
  method                      precision@40   average precision
  Mahalanobis (1 Gaussian)          0.28           0.26
  GMM, 3 comps (-log p)             0.68           0.75
  Isolation Forest                  0.45           0.52
  Local Outlier Factor              0.62           0.67
  planted anomalies within 0.5 of a normal point: 16 of 40
```

Only single linkage recovers the moons (ARI 1.000), yet the silhouette ranks it *last* and ranks the mostly wrong k-means *first*. Twelve bridge points make single linkage chain everything into one cluster. For anomalies, one big Gaussian scores worst because points between the three clusters look central to it. The 3-component GMM wins because it matches how the data was generated, an advantage you rarely have. With 16 of 40 planted "anomalies" sitting next to normal points, no detector can be perfect. Real benchmarks share this problem.

## Common mistakes

- **Unscaled features.** k-means, hierarchical clustering, LOF and PCA use distances or variances, so large-unit features decide the result. Check per-feature standard deviations and rerun standardized.
- **Trusting one run.** k-means, EM and t-SNE depend on initialization. Restart, and compare solutions, not just best objectives.
- **Reading t-SNE or UMAP quantitatively.** Change perplexity or `n_neighbors` and the seed. Anything that moves isn't a finding.
- **Choosing $k$ by the objective, or the algorithm by an internal index.** The first always prefers more clusters, and the second prefers compact convex clusters.
- **Fitting PCA on all data before splitting.** It is a learned transformation. Put it inside the cross-validation pipeline (L1-01).
- **Treating clusters as natural kinds.** k-means always returns $k$ clusters. Compare scores against the same data with each column independently shuffled.

## Exercises

1. [warm-up] Rerun Script 1 with `cluster_std=1.5`. What happens to the seeding gap and to the best ARI?
2. [warm-up] In Script 3, check that the PCA scores $\tilde{X} V_k$ are uncorrelated, with variances equal to the top $k$ eigenvalues.
3. [warm-up] By hand: the silhouette of point 0 in $\{0, 1, 5, 6\}$ with clusters $\{0, 1\}$ and $\{5, 6\}$.
4. [core] For Script 1's blobs, plot inertia and mean silhouette for $k = 1..15$. Do they point to $k = 10$?
5. [core] Make EM collapse (duplicate one point 50 times and start a component there), then add a guard that re-seeds any component with $N_j < 2$.
6. [core] Turn `em_gmm` into k-means: fixed covariance $\sigma^2 I$ with $\sigma = 0.01$ and equal weights. Compare its labels with Lloyd's from the same start.
7. [core] Repeat Script 4 part 2 with five seeds per perplexity. Is "distances between clusters mean nothing" too strong, too weak or about right?
8. [core] Vary Isolation Forest's `max_samples` (64, 256, 1024) and LOF's `n_neighbors` (5, 20, 100) in Script 5. Which detector is more sensitive here?
9. [stretch] Stability: for $k = 2..8$, cluster 20 bootstrap samples of Script 1's blobs and average the pairwise ARI on shared points. Compare the most stable $k$ with BIC's choice.
10. [stretch] Put PCA before logistic regression on digits in a `Pipeline` and choose the number of components by cross-validation. Is the best number near 21? Why or why not?

### Hints and answers

1. Wider blobs overlap more. Seeding still helps, but good and bad local minima get closer, and the best ARI drops because overlapping points go to the nearer center.
2. `np.cov((Xc @ W[:k].T).T)` is diagonal (off-diagonals about $10^{-13}$) with diagonal `var[:k]`, because $V^\top S V = \Sigma^2/(n-1)$.
3. $a = 1$, $b = 5.5$, so $s = 4.5/5.5 \approx 0.82$.
4. Overlapping blobs blur the elbow, and silhouette may peak below 10 because merging overlapping blobs can raise it. They measure compactness and separation, not the number of generating blobs.
5. The component's covariance shrinks to about `reg_covar` and the log-likelihood jumps. Re-seeding avoids the degenerate optimum.
6. Responsibilities become 0 or 1, the M-step mean equals the Lloyd update, and labels match except for exact ties.
7. Expect ratios far below 10 at low perplexity on every seed, and larger, more variable values at high perplexity. "Unreliable and perplexity-dependent" is the fair summary.
8. Compare how much precision@40 moves. Tiny `n_neighbors` makes LOF react to local noise, and large values blur small clusters.
9. Stability favors $k$ values where the cut is unambiguous, which can be below the generating number when blobs overlap.
10. Usually not exactly: variance kept is not label information kept. With regularization, extra components rarely hurt much.

## Check your understanding

1. **Why does Lloyd's algorithm terminate, and why might it terminate badly?** Each step minimizes the objective over one block of variables, so it never increases, and there are finitely many assignments. It stops where neither step helps, a local minimum that poor seeds can make much worse than the global one.

2. **How does k-means relate to EM's E step?** The E step computes soft responsibilities by Bayes' rule. k-means uses hard nearest-center assignments, which are the limit of EM for equal spherical components with variance going to zero.

3. **PCA on income (dollars) and age (years) gives PC1 with 99% of the variance. Conclusion?** Income has a far larger numeric variance in its units. Customers don't vary along one meaningful axis. Standardize and rerun.

4. **A t-SNE plot shows two groups, one twice as large and far from a third group. What can it support?** At most, that points within each group are neighbors in the original space. Relative size and the distance to the third group are not supported. Even the split needs checking at other perplexities and seeds, and in the original space.

5. **Why can silhouette rank a wrong clustering above a right one?** It rewards compact, separated clusters in Euclidean distance. A correct clustering of curved shapes has points far from much of their own cluster and scores low. A convex but wrong split scores high.

## Further reading

- **Stuart P. Lloyd, "Least Squares Quantization in PCM"** (1982, IEEE Transactions on Information Theory 28(2), 129-137). <https://www.stat.cmu.edu/~rnugent/PCMI2016/papers/LloydKMeans.pdf>
  *Read for:* the origin of Lloyd's algorithm in signal quantization (course copy; the IEEE version is paywalled). *Level:* advanced.
- **David Arthur and Sergei Vassilvitskii, "k-means++: The Advantages of Careful Seeding"** (2007, ACM-SIAM Symposium on Discrete Algorithms, SODA). <https://www.cs.columbia.edu/~verma/classes/uml/ref/clustering_kmean_plusplus_arthur_vassilvitskii.pdf>
  *Read for:* the $D^2$ seeding rule, its $O(\log k)$ guarantee, and experiments (course copy). *Level:* intermediate.
- **A. P. Dempster, N. M. Laird and D. B. Rubin, "Maximum Likelihood from Incomplete Data via the EM Algorithm"** (1977, Journal of the Royal Statistical Society Series B 39(1)). <https://academic.oup.com/jrsssb/article/39/1/1/7027539>
  *Read for:* the general EM framework and its monotone-likelihood property (publisher page, may be paywalled). *Level:* advanced.
- **Laurens van der Maaten and Geoffrey Hinton, "Visualizing Data using t-SNE"** (2008, Journal of Machine Learning Research 9, 2579-2605). <https://jmlr.org/papers/v9/vandermaaten08a.html>
  *Read for:* the t-SNE objective and why the Student-t kernel reduces crowding (open access). *Level:* intermediate.
- **Leland McInnes, John Healy and James Melville, "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction"** (2018, arXiv:1802.03426). <https://arxiv.org/abs/1802.03426>
  *Read for:* UMAP's construction and its authors' comparison with t-SNE. *Level:* advanced.
- **Martin Wattenberg, Fernanda Viégas and Ian Johnson, "How to Use t-SNE Effectively"** (2016, Distill). <https://distill.pub/2016/misread-tsne/>
  *Read for:* interactive versions of every t-SNE reading rule here; the best first read. *Level:* intro.
- **Tara Chari and Lior Pachter, "The specious art of single-cell genomics"** (2023, PLOS Computational Biology 19(8): e1011288). <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011288>
  *Read for:* a quantitative critique of t-SNE and UMAP as scientific evidence (open access; a correction notice was added in November 2025). *Level:* intermediate.
- **Fei Tony Liu, Kai Ming Ting and Zhi-Hua Zhou, "Isolation Forest"** (2008, IEEE International Conference on Data Mining, 413-422). <https://doi.org/10.1109/ICDM.2008.17>
  *Read for:* the isolation idea and why it scales (IEEE, may be paywalled). *Level:* intermediate.
- **Peter J. Rousseeuw, "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis"** (1987, Journal of Computational and Applied Mathematics 20, 53-65). <https://www.sciencedirect.com/science/article/pii/0377042787901257>
  *Read for:* the silhouette's definition and intended use (publisher page). *Level:* intermediate.
- **Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani and Jonathan Taylor, "An Introduction to Statistical Learning with Applications in Python"** (2023, Springer; free PDF from the authors). <https://www.statlearning.com/>
  *Read for:* Chapter 12 (Unsupervised Learning), a gentle treatment of PCA and clustering with Python labs. *Level:* intro.
- **Trevor Hastie, Robert Tibshirani and Jerome Friedman, "The Elements of Statistical Learning"** (2nd edition, 2009, Springer; free PDF from the authors). <https://hastie.su.domains/ElemStatLearn/>
  *Read for:* Chapter 14 (Unsupervised Learning), deeper on mixtures and clustering. *Level:* advanced.
- **scikit-learn developers, "Clustering", User Guide (version 1.9.1)**. <https://scikit-learn.org/stable/modules/clustering.html>
  *Read for:* algorithm comparisons, assumptions, and the ARI and silhouette metrics used here. *Level:* intro.
