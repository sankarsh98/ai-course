"""Broadcasting and indexing patterns that show up in every ML codebase."""
import numpy as np

rng = np.random.default_rng(0)

# 1. Standardize features: (n, d) minus (d,) broadcasts over rows.
X = rng.normal(loc=[0.0, 5.0, -3.0], scale=[1.0, 10.0, 0.1], size=(1000, 3))
mu = X.mean(axis=0)                      # shape (3,)
sd = X.std(axis=0)                       # shape (3,)
Z = (X - mu) / sd                        # (1000,3) op (3,) -> (1000,3)
print("column means after standardizing:", np.round(Z.mean(axis=0), 6))
print("column stds  after standardizing:", np.round(Z.std(axis=0), 6))

# 2. The classic bug: subtracting per-ROW statistics needs keepdims.
row_mean = X.mean(axis=1)                # shape (1000,)
try:
    X - row_mean                         # (1000,3) vs (1000,) -> trailing dims 3 vs 1000
except ValueError as e:
    print("\nwithout keepdims:", e)
centered = X - X.mean(axis=1, keepdims=True)   # (1000,3) - (1000,1) works
print("with keepdims, row means are now ~0:", np.abs(centered.mean(axis=1)).max() < 1e-12)

# 3. Pairwise squared distances between n queries and m keys with no loop.
#    ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2, each term broadcast to (n, m).
A = rng.normal(size=(4, 3))
B = rng.normal(size=(5, 3))
d2_bcast = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=-1)     # (4,1,3)-(1,5,3)
d2_expand = (A**2).sum(1)[:, None] - 2 * A @ B.T + (B**2).sum(1)[None, :]
print("\npairwise distance shape:", d2_bcast.shape,
      "| two methods agree:", np.allclose(d2_bcast, d2_expand))

# 4. A silent broadcasting bug: (n,) vs (n,1) gives an (n,n) matrix, not an error.
y_true = np.array([1.0, 2.0, 3.0])          # shape (3,)
y_pred = np.array([[1.1], [1.9], [3.2]])    # shape (3,1), e.g. output of a model
wrong = ((y_true - y_pred) ** 2).mean()
right = ((y_true - y_pred.ravel()) ** 2).mean()
print(f"\nMSE with shape bug: {wrong:.4f}  (diff shape {(y_true - y_pred).shape})")
print(f"MSE correct:        {right:.4f}")

# 5. Boolean masks and integer ("fancy") indexing.
scores = np.array([0.2, 0.9, 0.4, 0.75, 0.1])
print("\nmask scores > 0.5:", scores > 0.5, "->", scores[scores > 0.5])

# 6. One-hot encoding via fancy indexing, and gathering per-row entries.
labels = np.array([2, 0, 1, 2])
onehot = np.zeros((labels.size, 3))
onehot[np.arange(labels.size), labels] = 1.0
print("one-hot:\n", onehot)
probs = rng.dirichlet(np.ones(3), size=4)    # each row sums to 1
p_correct = probs[np.arange(4), labels]      # probability of the true class per row
print("prob of true class:", np.round(p_correct, 3))

# 7. Repeated indices: += applies once per unique index, np.add.at accumulates.
counts = np.zeros(3)
idx = np.array([0, 0, 0, 2])
counts[idx] += 1
print("\ncounts[idx] += 1  ->", counts)
counts = np.zeros(3)
np.add.at(counts, idx, 1)
print("np.add.at          ->", counts)
print("np.bincount        ->", np.bincount(idx, minlength=3))
