"""A synthetic recommendation world with known ground truth, plus shared helpers.

Because we generate the data, we know every user's true preference for every item,
including the ones they never saw. Real logs never give you that; here it lets us
check what offline metrics get right and wrong.
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_world(n_users=600, n_items=400, k=8, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1, (n_users, k)) / np.sqrt(k) * 1.6   # true user factors
    Q = rng.normal(0, 1, (n_items, k))                      # true item factors
    b_user = rng.normal(0, 0.3, n_users)
    b_item = rng.normal(0, 0.5, n_items)                    # item "quality"
    affinity = P @ Q.T + b_user[:, None] + b_item[None, :]
    # Exposure propensity: a power law over items, only loosely tied to quality.
    order = np.argsort(-(b_item + rng.normal(0, 1.0, n_items)))
    pop = np.empty(n_items)
    pop[order] = 1.0 / np.arange(1, n_items + 1) ** 1.0
    pop /= pop.sum()
    # Side features that a two-tower model can use: noisy linear views of the truth.
    user_feat = P @ rng.normal(0, 1, (k, 12)) + rng.normal(0, 0.5, (n_users, 12))
    item_feat = np.hstack([Q, b_item[:, None]]) @ rng.normal(0, 1, (k + 1, 12)) \
        + rng.normal(0, 0.5, (n_items, 12))
    return dict(P=P, Q=Q, affinity=affinity, pop=pop, b_item=b_item,
                user_feat=user_feat.astype(np.float32), item_feat=item_feat.astype(np.float32),
                click_prob=sigmoid(2.0 * (affinity - 1.5)))


def organic_clicks(world, exposures=60, seed=1):
    """Each user is shown `exposures` items drawn by popularity, and clicks some."""
    rng = np.random.default_rng(seed)
    n_users, n_items = world["click_prob"].shape
    X = np.zeros((n_users, n_items), np.float32)
    for u in range(n_users):
        shown = rng.choice(n_items, exposures, replace=False, p=world["pop"])
        clicks = shown[rng.random(exposures) < world["click_prob"][u, shown]]
        X[u, clicks] = 1
    return X


def split_holdout(X, frac=0.2, seed=2):
    """Hold out a fraction of each user's positives for testing."""
    rng = np.random.default_rng(seed)
    train, test = X.copy(), np.zeros_like(X)
    for u in range(X.shape[0]):
        pos = np.flatnonzero(X[u])
        if len(pos) >= 2:
            held = rng.choice(pos, max(1, int(frac * len(pos))), replace=False)
            train[u, held], test[u, held] = 0, 1
    return train, test


# ---------------------------------------------------------------- models
def ials(X, k=16, alpha=10.0, reg=1.0, iters=10, seed=0):
    """Implicit ALS (Hu, Koren and Volinsky, 2008): every cell is a 0/1 preference
    with confidence 1 + alpha * x. Dense solves are fine at this size."""
    rng = np.random.default_rng(seed)
    n_u, n_i = X.shape
    U, V = rng.normal(0, 0.1, (n_u, k)), rng.normal(0, 0.1, (n_i, k))
    C = 1.0 + alpha * X                              # confidence
    Pref = (X > 0).astype(float)                     # preference
    I = reg * np.eye(k)
    for _ in range(iters):
        for A, B, Cm, Pm in ((U, V, C, Pref), (V, U, C.T, Pref.T)):
            for r in range(A.shape[0]):
                Bc = B * Cm[r][:, None]              # rows of B weighted by confidence
                A[r] = np.linalg.solve(B.T @ Bc + I, Bc.T @ Pm[r])
    return U, V


def item_knn_scores(X, shrink=10.0):
    """Item-item cosine similarity; a user's score for i sums similarities to their items."""
    norms = np.sqrt(X.sum(0))
    S = (X.T @ X) / (np.outer(norms, norms) + shrink)
    np.fill_diagonal(S, 0)
    return X @ S


# ---------------------------------------------------------------- metrics
def topk_metrics(scores, train, test, k=10):
    """Full-catalogue Recall@k and NDCG@k, never recommending training items."""
    s = np.where(train > 0, -np.inf, scores)
    top = np.argsort(-s, axis=1)[:, :k]
    users = np.flatnonzero(test.sum(1) > 0)
    rec, ndcg = [], []
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for u in users:
        hits = test[u, top[u]] > 0
        n_rel = int(test[u].sum())
        rec.append(hits.sum() / min(n_rel, k))
        ndcg.append((hits * discounts).sum() / discounts[:min(n_rel, k)].sum())
    return float(np.mean(rec)), float(np.mean(ndcg))


def sampled_hit_rate(scores, train, test, n_neg=100, k=10, seed=3):
    """The common shortcut: rank each test item against n_neg random negatives only."""
    rng = np.random.default_rng(seed)
    hits = []
    for u, i in zip(*np.nonzero(test)):
        cand = np.flatnonzero((train[u] == 0) & (test[u] == 0))
        neg = rng.choice(cand, n_neg, replace=False)
        hits.append((scores[u, neg] >= scores[u, i]).sum() < k)
    return float(np.mean(hits))


def gini(x):
    """Gini coefficient of exposure counts: 0 = equal, near 1 = concentrated."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))
