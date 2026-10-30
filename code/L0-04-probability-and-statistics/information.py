"""Entropy, cross-entropy, and KL divergence, and why minimizing cross-entropy is maximum likelihood."""
import numpy as np
from scipy import stats


def entropy(p):
    p = np.asarray(p, float)
    nz = p > 0  # 0 * log 0 is taken to be 0
    return -np.sum(p[nz] * np.log2(p[nz])) + 0.0  # + 0.0 turns -0.0 into 0.0


def cross_entropy(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    nz = p > 0
    return -np.sum(p[nz] * np.log2(q[nz]))  # infinite if q = 0 where p > 0


def kl(p, q):
    return cross_entropy(p, q) - entropy(p)


print("Entropy in bits")
for name, p in [("fair coin", [0.5, 0.5]), ("90/10 coin", [0.9, 0.1]),
                ("99/1 coin", [0.99, 0.01]), ("fair 8-sided die", [1 / 8] * 8),
                ("certain outcome", [1.0, 0.0])]:
    print(f"  {name:<17} H = {entropy(p):.4f}")

p = np.array([0.7, 0.2, 0.1])
q = np.array([0.4, 0.4, 0.2])
print(f"\np = {p}, q = {q}")
print(f"  H(p) = {entropy(p):.4f}  H(p, q) = {cross_entropy(p, q):.4f}  "
      f"KL(p||q) = {kl(p, q):.4f}  KL(q||p) = {kl(q, p):.4f}")

# Cross-entropy loss of a classifier is the average negative log-likelihood of the labels.
rng = np.random.default_rng(0)
labels = rng.integers(0, 3, size=8)
logits = rng.normal(size=(8, 3))
probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  # softmax
nll = -np.mean(np.log(probs[np.arange(8), labels]))
one_hot = np.eye(3)[labels]
ce = np.mean([-np.sum(one_hot[i] * np.log(probs[i])) for i in range(8)])
print(f"\nclassifier: mean NLL of true labels = {nll:.6f}, mean cross-entropy vs one-hot = {ce:.6f} (nats)")

# Cross-entropy H(p_hat, q) = H(p_hat) + KL(p_hat || q) is smallest when q equals the data frequencies.
data = rng.choice(3, size=1000, p=[0.5, 0.3, 0.2])
p_hat = np.bincount(data, minlength=3) / len(data)
print("\n1000 draws from [0.5, 0.3, 0.2]; empirical frequencies p_hat =", np.round(p_hat, 3))
for q_try in [p_hat, [0.5, 0.3, 0.2], [1 / 3] * 3, [0.8, 0.1, 0.1]]:
    print(f"  q = {np.round(q_try, 3)}: cross-entropy with data = {cross_entropy(p_hat, q_try):.4f} bits")

# Forward vs reverse KL: fit one Gaussian q to a two-bump mixture p.
x = np.linspace(-10, 10, 4001)
dx = x[1] - x[0]
p_x = 0.5 * stats.norm.pdf(x, -3, 1) + 0.5 * stats.norm.pdf(x, 3, 1)


def kl_cont(a, b):
    mask = a > 1e-300
    return np.sum(a[mask] * (np.log(a[mask]) - np.log(b[mask] + 1e-300))) * dx  # nats


grid = [(mu, sigma) for mu in np.linspace(-5, 5, 101) for sigma in np.linspace(0.5, 5, 91)]
print("\nOne Gaussian q fitted by grid search to p = 50/50 mixture of N(-3,1) and N(3,1):")
for name, div in [("forward KL(p||q)", lambda q: kl_cont(p_x, q)), ("reverse KL(q||p)", lambda q: kl_cont(q, p_x))]:
    vals = [div(stats.norm.pdf(x, mu, sigma)) for mu, sigma in grid]
    mu, sigma = grid[int(np.argmin(vals))]
    print(f"  minimizing {name}: mu = {mu:+.2f}, sigma = {sigma:.2f}, divergence = {min(vals):.3f} nats")
