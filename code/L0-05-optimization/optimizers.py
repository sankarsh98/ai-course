"""SGD, momentum, RMSProp, Adam and AdamW written from scratch, then compared.

Each optimizer is a function step(w, g, state, lr) -> new w that keeps its own state.
"""
import numpy as np


def sgd(w, g, s, lr):
    return w - lr * g

def momentum(w, g, s, lr, beta=0.9):
    s["v"] = beta * s.get("v", 0.0) + g          # running sum of past gradients
    return w - lr * s["v"]

def rmsprop(w, g, s, lr, rho=0.99, eps=1e-8):
    s["r"] = rho * s.get("r", 0.0) + (1 - rho) * g * g
    return w - lr * g / (np.sqrt(s["r"]) + eps)  # per-coordinate scaling

def adam(w, g, s, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, bias_correct=True):
    t = s["t"] = s.get("t", 0) + 1
    s["m"] = b1 * s.get("m", 0.0) + (1 - b1) * g
    s["v"] = b2 * s.get("v", 0.0) + (1 - b2) * g * g
    m_hat = s["m"] / (1 - b1 ** t) if bias_correct else s["m"]
    v_hat = s["v"] / (1 - b2 ** t) if bias_correct else s["v"]
    # AdamW: weight decay applied directly to w, outside the adaptive scaling.
    return w - lr * m_hat / (np.sqrt(v_hat) + eps) - lr * wd * w

OPTS = {"sgd": sgd, "momentum": momentum, "rmsprop": rmsprop, "adam": adam}


def steps_to_tol(opt, f, grad, w0, lr, tol=1e-8, max_steps=5000):
    """Steps until f(w) < tol * f(w0); None if it diverges or runs out of steps."""
    w, s, f0 = w0.copy(), {}, f(w0)
    for t in range(1, max_steps + 1):
        w = opt(w, grad(w), s, lr)
        if not np.all(np.isfinite(w)) or np.abs(w).max() > 1e6:
            return None
        if f(w) < tol * f0:
            return t
    return None


def best_lr(opt, f, grad, w0, lrs=np.logspace(-5, 0, 11)):
    """Tune the learning rate on a log grid so each method gets a fair chance."""
    runs = [(steps_to_tol(opt, f, grad, w0, lr), lr) for lr in lrs]
    runs = [r for r in runs if r[0] is not None]
    return min(runs) if runs else (None, None)


def quadratic(R, kappa=1e4):
    """f(w) = 0.5 w^T A w with eigenvalues 1 and kappa, eigenvectors = columns of R."""
    A = R @ np.diag([1.0, kappa]) @ R.T
    return (lambda w: 0.5 * w @ A @ w), (lambda w: A @ w)

c = np.cos(np.pi / 4)
rotated = np.array([[c, -c], [c, c]])
w_start = np.array([1.0, 0.3])
problems = {
    "quadratic, axis-aligned": (*quadratic(np.eye(2)), w_start),
    "quadratic, rotated 45deg": (*quadratic(rotated), w_start),
    "rosenbrock": (
        lambda w: (1 - w[0]) ** 2 + 100 * (w[1] - w[0] ** 2) ** 2,
        lambda w: np.array([-2 * (1 - w[0]) - 400 * w[0] * (w[1] - w[0] ** 2),
                            200 * (w[1] - w[0] ** 2)]),
        np.array([-1.5, 2.0]),
    ),
}

print("Steps to reduce f by 1e8 (max 5000), best lr from an 11-point log grid")
print(f"{'problem':>25} " + " ".join(f"{k:>17}" for k in OPTS))
for name, (f, g, w0) in problems.items():
    cells = []
    for opt in OPTS.values():
        n, lr = best_lr(opt, f, g, w0)
        cells.append(">5000 (all lrs)" if n is None else f"{n} (lr {lr:.0e})")
    print(f"{name:>25} " + " ".join(f"{cell:>17}" for cell in cells))

print("\nAdam bias correction: size of the first 3 steps, lr = 0.001, g = 0.01 each step")
for bc in [True, False]:
    w, s, sizes = np.zeros(1), {}, []
    for _ in range(3):
        w_new = adam(w, np.array([0.01]), s, 0.001, bias_correct=bc)
        sizes.append(abs(w_new - w)[0])
        w = w_new
    print(f"bias_correct={bc!s:5}: " + ", ".join(f"{x:.2e}" for x in sizes))

print("\nL2-in-the-gradient Adam vs AdamW, 3000 steps, lr 1e-3, strength 0.1")
print("Loss gradient is pure noise (no signal), so only the decay moves w on average.")
rng = np.random.default_rng(0)
noise_scale = np.repeat([10.0, 0.01], 500)      # 500 noisy and 500 quiet weights
for label, l2, wd in [("Adam + L2", 0.1, 0.0), ("AdamW", 0.0, 0.1)]:
    w, s = np.ones(1000), {}
    for _ in range(3000):
        g = noise_scale * rng.normal(size=1000) + l2 * w
        w = adam(w, g, s, 1e-3, wd=wd)
    print(f"{label:>10}: mean w, noisy weights = {w[:500].mean():.3f}, "
          f"quiet weights = {w[500:].mean():.3f}")
