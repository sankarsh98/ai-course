"""Forward vs reverse mode on f(x) = W2 @ tanh(W1 @ x), f: R^n -> R^m.

A JVP pushes a tangent v forward and returns J @ v; a VJP pulls a cotangent u
backward and returns u @ J. The full m x n Jacobian takes n JVPs or m VJPs.
"""
import time

import numpy as np

HIDDEN = 300


def jvp(W1, W2, x, v):
    a = W1 @ x
    return W2 @ ((1 - np.tanh(a) ** 2) * (W1 @ v))  # tangents flow with the values


def vjp(W1, W2, h, u):
    return W1.T @ ((1 - h ** 2) * (W2.T @ u))  # uses h = tanh(W1 x) saved from the forward pass


def jacobian_forward(W1, W2, x):
    return np.stack([jvp(W1, W2, x, e) for e in np.eye(x.size)], axis=1)  # one pass per input


def jacobian_reverse(W1, W2, x):
    h = np.tanh(W1 @ x)  # forward pass, stored for the backward passes
    return np.stack([vjp(W1, W2, h, e) for e in np.eye(W2.shape[0])])  # one pass per output


def best_time(fn, *args, repeats=5):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        times.append(time.perf_counter() - t0)
    return out, 1e3 * min(times)


print(f"{'n':>5} {'m':>5} | {'fwd passes':>10} {'fwd ms':>7} | {'rev passes':>10} {'rev ms':>7} | max diff")
for n, m in [(1000, 1), (1000, 10), (100, 100), (10, 1000), (1, 1000)]:
    rng = np.random.default_rng(0)
    W1 = rng.normal(size=(HIDDEN, n)) / np.sqrt(n)
    W2 = rng.normal(size=(m, HIDDEN)) / np.sqrt(HIDDEN)
    x = rng.normal(size=n)
    J_f, t_f = best_time(jacobian_forward, W1, W2, x)
    J_r, t_r = best_time(jacobian_reverse, W1, W2, x)
    print(f"{n:>5} {m:>5} | {n:>10} {t_f:>7.2f} | {m:>10} {t_r:>7.2f} | {np.abs(J_f - J_r).max():.1e}")
