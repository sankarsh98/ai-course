"""Embedding geometry: cosine vs dot product, nearest neighbors, and pitfalls.

The embeddings are synthetic so the script needs no downloads: each "item"
is a topic direction plus noise, plus a shared offset that mimics the
common-direction (anisotropy) effect reported for real embeddings.
"""
import numpy as np

rng = np.random.default_rng(0)
d, n_topics, per_topic = 64, 4, 50

# 1. Build embeddings: topic centers + noise + random norms + a shared offset.
centers = rng.normal(size=(n_topics, d))
centers /= np.linalg.norm(centers, axis=1, keepdims=True)
labels = np.repeat(np.arange(n_topics), per_topic)
E = centers[labels] + 2.0 * rng.normal(size=(len(labels), d)) / np.sqrt(d)
E *= rng.uniform(0.5, 3.0, size=(len(labels), 1))       # norms vary (e.g. frequency)
shared = 4.0 * rng.normal(size=d) / np.sqrt(d)          # common direction
E_aniso = E + shared


def normalize(M):
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def knn_topic_accuracy(S, labels, k=5):
    """Fraction of each item's k nearest neighbors (by score S) in the same topic."""
    S = S.copy()
    np.fill_diagonal(S, -np.inf)                         # exclude self
    nn = np.argsort(-S, axis=1)[:, :k]
    return np.mean(labels[nn] == labels[:, None])


def mean_offdiag_cos(M):
    C = normalize(M) @ normalize(M).T
    return C[~np.eye(len(M), dtype=bool)].mean()


# 2. Dot product rewards large norms; cosine ignores them.
print(f"{'embeddings':<22}{'score':<8}{'kNN same-topic':>15}{'mean cos':>10}")
for name, M in [("isotropic", E), ("with shared offset", E_aniso),
                ("offset, mean-removed", E_aniso - E_aniso.mean(0))]:
    for score, S in [("dot", M @ M.T), ("cosine", normalize(M) @ normalize(M).T)]:
        print(f"{name:<22}{score:<8}{knn_topic_accuracy(S, labels):>15.3f}"
              f"{mean_offdiag_cos(M):>10.3f}")

# 3. Nearest-neighbor search is one matrix multiply on normalized vectors.
query = E[0] + 0.05 * rng.normal(size=d)
scores = normalize(E) @ (query / np.linalg.norm(query))
top = np.argsort(-scores)[:5]
print("\nquery is a noisy copy of item 0 (topic 0)")
print("top-5 items:", top, "topics:", labels[top], "cos:", np.round(scores[top], 3))

# 4. Cosine similarity is not invariant to per-dimension rescaling.
#    If a model only uses dot products u.v, then (u D) . (v D^-1) = u.v for
#    any positive diagonal D: the model's predictions are unchanged, but the
#    cosine similarities among the u's change. This is the core of the
#    argument in Steck et al. (2024).
D = np.exp(rng.normal(scale=1.0, size=d))
U_items = E[:5]
V_users = rng.normal(size=(3, d))
print("\nuser-item dot products identical after rescaling:",
      np.allclose(U_items @ V_users.T, (U_items * D) @ (V_users / D).T))
c_before = normalize(U_items) @ normalize(U_items).T
c_after = normalize(U_items * D) @ normalize(U_items * D).T
print("item-item cosine before:", np.round(c_before[0, 1:], 3))
print("item-item cosine after: ", np.round(c_after[0, 1:], 3))

# 5. Analogy arithmetic works when relations are consistent vector offsets.
#    We plant one: "plural" = a fixed offset vector.
base = normalize(rng.normal(size=(5, d)))
plural = 0.8 * normalize(rng.normal(size=(1, d)))[0]
vocab = np.vstack([base, base + plural + 0.05 * rng.normal(size=(5, d))])
names = ["cat", "dog", "car", "tree", "cup", "cats", "dogs", "cars", "trees", "cups"]
target = vocab[names.index("dogs")] - vocab[names.index("dog")] + vocab[names.index("car")]
sims = normalize(vocab) @ (target / np.linalg.norm(target))
for w in ("dogs", "dog", "car"):             # standard practice: exclude inputs
    sims[names.index(w)] = -np.inf
print("\ndogs - dog + car ->", names[int(np.argmax(sims))])
