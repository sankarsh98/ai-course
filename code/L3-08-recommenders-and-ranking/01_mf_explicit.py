"""Explicit-feedback matrix factorization (biased MF trained by SGD) on synthetic
ratings, compared with baselines and with SVD of an imputed matrix. Then: what the
observed-ratings test set hides when ratings are missing not at random.
"""
import numpy as np

from recdata import make_world

world = make_world()
A = world["affinity"]
n_u, n_i = A.shape
rng = np.random.default_rng(0)

# True expected rating on a 1-5 scale; an observed rating adds noise and rounds.
true_rating = np.clip(3.0 + 0.7 * (A - A.mean()), 1, 5)
# Missing not at random: people see popular items and rate what they like.
w = (world["pop"][None, :] ** 0.5) * np.exp(0.8 * A)
cells = rng.choice(n_u * n_i, 15_000, replace=False, p=(w / w.sum()).ravel())
u_all, i_all = np.divmod(cells, n_i)
r_all = np.clip(np.round(true_rating[u_all, i_all] + rng.normal(0, 0.5, len(cells))), 1, 5)
perm = rng.permutation(len(cells))
tr, va, te = np.split(perm, [int(0.8 * len(perm)), int(0.9 * len(perm))])
print(f"{len(cells)} ratings, density {len(cells) / (n_u * n_i):.1%}, "
      f"mean observed rating {r_all.mean():.2f} vs mean true rating {true_rating.mean():.2f}")


def rmse(pred, idx):
    return np.sqrt(np.mean((pred(u_all[idx], i_all[idx]) - r_all[idx]) ** 2))


def fit_mf(k, lam, epochs=40, lr=0.01, seed=0):
    """r_hat = mu + b_u + b_i + p_u . q_i, squared error + L2, plain SGD."""
    g = np.random.default_rng(seed)
    mu = r_all[tr].mean()
    bu, bi = np.zeros(n_u), np.zeros(n_i)
    Pu, Qi = g.normal(0, 0.1, (n_u, k)), g.normal(0, 0.1, (n_i, k))
    for _ in range(epochs):
        for t in g.permutation(tr):
            u, i, r = u_all[t], i_all[t], r_all[t]
            e = r - (mu + bu[u] + bi[i] + Pu[u] @ Qi[i])
            bu[u] += lr * (e - lam * bu[u])
            bi[i] += lr * (e - lam * bi[i])
            Pu[u], Qi[i] = (Pu[u] + lr * (e * Qi[i] - lam * Pu[u]),
                            Qi[i] + lr * (e * Pu[u] - lam * Qi[i]))
    return lambda u, i: np.clip(mu + bu[u] + bi[i] + np.sum(Pu[u] * Qi[i], axis=-1), 1, 5)


# ---- 1. Baselines and MF ------------------------------------------------------
mu = r_all[tr].mean()
print("\n1. RMSE on observed ratings              train    valid    test")
models = {"global mean": lambda u, i: np.full(len(np.atleast_1d(u)), mu),
          "oracle (true expectation)": lambda u, i: true_rating[u, i],
          "biases only (k=0)": fit_mf(0, 0.05)}
for k, lam in [(8, 0.0), (8, 0.05), (32, 0.0), (32, 0.05), (32, 0.2)]:
    models[f"MF k={k:<2d} lambda={lam}"] = fit_mf(k, lam)
for name, f in models.items():
    print(f"   {name:26s}       {rmse(f, tr):.3f}    {rmse(f, va):.3f}    {rmse(f, te):.3f}")

# ---- 2. Why not just take the SVD? -----------------------------------------------
# Plain SVD needs a full matrix. Filling the gaps with the mean treats "not rated"
# as "rated average", which is false for 94% of the cells.
print("\n2. Truncated SVD after filling missing cells with the mean")
M = np.full((n_u, n_i), mu)
M[u_all[tr], i_all[tr]] = r_all[tr]
Uu, s, Vt = np.linalg.svd(M - mu, full_matrices=False)
for k in (8, 32):
    Mk = mu + (Uu[:, :k] * s[:k]) @ Vt[:k]
    f = lambda u, i, Mk=Mk: np.clip(Mk[u, i], 1, 5)
    print(f"   {'SVD, mean-imputed, k=' + str(k):26s}       {rmse(f, tr):.3f}    "
          f"{rmse(f, va):.3f}    {rmse(f, te):.3f}")

# ---- 3. What the observed test set hides -------------------------------------
# Score every model on 20,000 random (user, item) cells against the TRUE expected
# rating. A real system can only approximate this with a randomized experiment.
ru, ri = rng.integers(0, n_u, 20_000), rng.integers(0, n_i, 20_000)
print("\n3. Observed test cells vs random cells (against the true expected rating)")
print("   model                         RMSE test   RMSE random   mean pred random")
for name in ("oracle (true expectation)", "global mean", "biases only (k=0)",
             "MF k=8  lambda=0.05", "MF k=32 lambda=0.2"):
    f = models[name]
    rand_err = np.sqrt(np.mean((f(ru, ri) - true_rating[ru, ri]) ** 2))
    print(f"   {name:26s}      {rmse(f, te):.3f}       {rand_err:.3f}         "
          f"{f(ru, ri).mean():.2f}")
print(f"   (mean true rating over random cells: {true_rating[ru, ri].mean():.2f})")
