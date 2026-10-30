"""Shared helpers: intervals for a proportion, and a two-proportion z-test."""
import numpy as np
from scipy import stats

Z = 1.96  # two-sided 95%


def wald(k, n):
    p = k / n
    half = Z * np.sqrt(p * (1 - p) / n)
    return p - half, p + half


def wilson(k, n):
    p = k / n
    center = (p + Z ** 2 / (2 * n)) / (1 + Z ** 2 / n)
    half = Z * np.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / (1 + Z ** 2 / n)
    return center - half, center + half


def two_proportion_p(ka, kb, n):
    """Two-sided p-value for 'equal accuracy' with independent test sets of size n."""
    pooled = (ka + kb) / (2 * n)
    z = (kb - ka) / n / np.sqrt(pooled * (1 - pooled) * 2 / n)
    return 2 * stats.norm.sf(np.abs(z))
