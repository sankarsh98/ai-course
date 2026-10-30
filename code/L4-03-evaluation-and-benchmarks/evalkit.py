"""Small statistics toolkit for evaluations (imported by the scripts; prints nothing).

Conventions: `scores` is a 1-D array with one number per eval item (1/0 for
correct/incorrect, or a mean over several samples of the same item).
`clusters` is an optional array of the same length giving each item's group
(for example, several questions written about the same passage).
"""
import numpy as np
from scipy.stats import norm


def mean_se(scores):
    """Standard error of the mean from the central limit theorem."""
    s = np.asarray(scores, float)
    return s.std(ddof=1) / np.sqrt(len(s))


def clustered_se(scores, clusters):
    """Cluster-robust standard error of the mean: residuals are summed within
    each cluster before squaring, so correlated items count once, not n times."""
    s = np.asarray(scores, float)
    resid = s - s.mean()
    sums = np.array([resid[clusters == c].sum() for c in np.unique(clusters)])
    G = len(sums)
    return np.sqrt(G / (G - 1) * (sums ** 2).sum()) / len(s)


def bootstrap_ci(scores, clusters=None, n_boot=4000, level=0.95, seed=0):
    """Percentile bootstrap interval for the mean. With clusters, resample
    whole clusters (the unit that was sampled), not individual items."""
    rng = np.random.default_rng(seed)
    s = np.asarray(scores, float)
    if clusters is None:
        idx = rng.integers(len(s), size=(n_boot, len(s)))
        stats = s[idx].mean(1)
    else:
        groups = [s[clusters == c] for c in np.unique(clusters)]
        sums = np.array([g.sum() for g in groups])
        sizes = np.array([len(g) for g in groups])
        pick = rng.integers(len(groups), size=(n_boot, len(groups)))
        stats = sums[pick].sum(1) / sizes[pick].sum(1)
    a = (1 - level) / 2
    return np.quantile(stats, [a, 1 - a])


def paired_permutation_p(a, b, clusters=None, n_perm=10000, seed=0):
    """Two-sided p-value for 'A and B have the same mean score' on the same
    items. Under that null, swapping A's and B's result on an item (or on a
    whole cluster) is equally likely, so we flip the sign of differences."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    if clusters is not None:
        d = np.array([d[clusters == c].sum() for c in np.unique(clusters)])
    obs = abs(d.sum())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    return (np.abs((signs * d).sum(1)) >= obs - 1e-12).mean()


def power_paired(sd_diff, delta, n, alpha=0.05):
    """Approximate power of a two-sided paired z-test to detect a true mean
    difference `delta` with `n` items, given the per-item sd of differences."""
    z = norm.ppf(1 - alpha / 2)
    return norm.cdf(abs(delta) * np.sqrt(n) / sd_diff - z)


def n_for_power(sd_diff, delta, power=0.8, alpha=0.05):
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return int(np.ceil((z * sd_diff / delta) ** 2))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order]).clip(max=1)
    out = np.empty(m)
    out[order] = adj
    return out


def benjamini_hochberg(pvals):
    """BH adjusted p-values (controls the false discovery rate)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = (p[order] * m / np.arange(1, m + 1))[::-1]
    adj = np.minimum.accumulate(adj)[::-1].clip(max=1)
    out = np.empty(m)
    out[order] = adj
    return out


def cohen_kappa(r1, r2):
    """Agreement between two raters beyond what their label frequencies
    would produce by chance: (p_o - p_e) / (1 - p_e)."""
    r1, r2 = np.asarray(r1), np.asarray(r2)
    cats = np.union1d(r1, r2)
    p_o = (r1 == r2).mean()
    p_e = sum((r1 == c).mean() * (r2 == c).mean() for c in cats)
    return (p_o - p_e) / (1 - p_e)


def krippendorff_alpha_nominal(ratings):
    """Krippendorff's alpha for nominal labels. `ratings` is (raters, items)
    with np.nan for missing ratings. alpha = 1 - observed / expected disagreement."""
    R = np.asarray(ratings, float)
    cats = np.unique(R[~np.isnan(R)])
    # Coincidence matrix: within each item, every ordered pair of ratings from
    # different raters, weighted by 1 / (m_u - 1).
    o = np.zeros((len(cats), len(cats)))
    for u in range(R.shape[1]):
        vals = R[~np.isnan(R[:, u]), u]
        m = len(vals)
        if m < 2:
            continue
        counts = np.array([(vals == c).sum() for c in cats])
        o += (np.outer(counts, counts) - np.diag(counts)) / (m - 1)
    n_c = o.sum(1)
    n = n_c.sum()
    d_o = (o.sum() - np.trace(o)) / n
    d_e = (n ** 2 - (n_c ** 2).sum()) / (n * (n - 1))
    return 1 - d_o / d_e
