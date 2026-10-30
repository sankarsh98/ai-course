"""Numerical derivatives: the step size h trades truncation error against round-off error."""
import numpy as np


def f(x):
    return np.sin(x) * np.exp(x)


def df_exact(x):
    return (np.cos(x) + np.sin(x)) * np.exp(x)  # product rule


x0 = 1.0
exact = df_exact(x0)
print(f"exact f'({x0}) = {exact:.15f}\n")
print(f"{'h':>8} | {'forward error':>14} | {'central error':>14}")
print("-" * 42)
for k in range(1, 15):
    h = 10.0 ** (-k)
    e_fwd = abs((f(x0 + h) - f(x0)) / h - exact)
    e_cen = abs((f(x0 + h) - f(x0 - h)) / (2 * h) - exact)
    print(f"{h:8.0e} | {e_fwd:14.3e} | {e_cen:14.3e}")
