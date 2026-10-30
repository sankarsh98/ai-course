"""Exponential smoothing (Holt-Winters, additive, damped trend) and a seasonal
ARIMA-style model, both from scratch, backtested against seasonal naive.
Then: are the 80% prediction intervals honest?"""
import os

# Many small fits: extra threads mostly add overhead, and on a shared machine
# they can slow things down a lot. Must be set before numpy is imported.
for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(var, "1")

import numpy as np
from scipy.optimize import minimize

M, H = 7, 14


def make_series(n, rng):                      # same generator as script 1
    t = np.arange(n)
    weekly = np.array([0, 2, 3, 2, 1, -3, -5])
    noise, eps = np.zeros(n), rng.normal(0, 3, n)
    for i in range(1, n):
        noise[i] = 0.6 * noise[i - 1] + eps[i]
    return 100 + 0.03 * t + 3 * weekly[t % 7] + 8 * np.sin(2 * np.pi * t / 365.25) + noise


# ---------------------------------------------------------------- ETS -------
def hw_filter(y, alpha, beta, gamma, phi):
    """Additive Holt-Winters in error-correction form. Returns one-step errors
    and the final (level, trend, seasonal) state."""
    level = y[:M].mean()
    trend = (y[M:2 * M].mean() - y[:M].mean()) / M
    season = list(y[:M] - level)                  # season[-M] is s_{t-m}
    errs = np.empty(len(y) - M)
    for i, obs in enumerate(y[M:]):
        e = obs - (level + phi * trend + season[-M])
        errs[i] = e
        level = level + phi * trend + alpha * e
        trend = phi * trend + beta * e
        season.append(season[-M] + gamma * e)
    return errs, level, trend, season[-M:]


def hw_fit(y):
    """Choose (alpha, beta, gamma, phi) by minimizing in-sample one-step SSE."""
    sse = lambda p: np.sum(hw_filter(y, *p)[0] ** 2)
    res = minimize(sse, x0=[0.3, 0.01, 0.1, 0.95], method="L-BFGS-B",
                   bounds=[(1e-3, 1), (0, 0.3), (0, 0.5), (0.8, 0.98)])
    return res.x


def hw_forecast(y, params, h):
    errs, level, trend, season = hw_filter(y, *params)
    phi = params[3]
    damp = np.cumsum(phi ** np.arange(1, h + 1))          # phi + phi^2 + ... + phi^k
    return level + damp * trend + np.array(season)[np.arange(h) % M], errs


def hw_simulate(y, params, h, n_paths, rng):
    """Sample future paths by feeding bootstrapped one-step errors back through
    the state equations. Uncertainty compounds, so intervals widen with h."""
    alpha, beta, gamma, phi = params
    errs, level0, trend0, season0 = hw_filter(y, *params)
    paths = np.empty((n_paths, h))
    for p in range(n_paths):
        level, trend, season = level0, trend0, list(season0)
        for k in range(h):
            e = rng.choice(errs)
            yhat = level + phi * trend + season[-M]
            paths[p, k] = yhat + e
            level = level + phi * trend + alpha * e
            trend = phi * trend + beta * e
            season.append(season[-M] + gamma * e)
    return paths


# --------------------------------------------------------- ARIMA-style -------
def ar_design(z, p):
    """Rows [1, z_{t-1}, ..., z_{t-p}] with target z_t."""
    X = np.column_stack([np.ones(len(z) - p)] + [z[p - j:len(z) - j] for j in range(1, p + 1)])
    return X, z[p:]


def sarima_fit(y, max_p=10):
    """Seasonal difference (lag M), then AR(p) by least squares, p chosen by AIC.
    This is ARIMA(p,0,0)(0,1,0)_M with a constant."""
    z = y[M:] - y[:-M]
    best = None
    for p in range(1, max_p + 1):
        X, target = ar_design(z[max_p - p:], p)        # same rows for every p
        coef, *_ = np.linalg.lstsq(X, target, rcond=None)
        sigma2 = np.mean((target - X @ coef) ** 2)
        aic = len(target) * np.log(sigma2) + 2 * (p + 1)
        if best is None or aic < best[0]:
            best = (aic, p, coef)
    return best[1], best[2]


def sarima_forecast(y, p, coef, h):
    z = list(y[M:] - y[:-M])
    hist = list(y)
    for _ in range(h):
        z_next = coef[0] + sum(coef[j] * z[-j] for j in range(1, p + 1))
        z.append(z_next)
        hist.append(hist[-M] + z_next)                   # undo the differencing
    return np.array(hist[-h:])


def acf(x, lags):
    x = x - x.mean()
    return [np.dot(x[:-k], x[k:]) / np.dot(x, x) for k in lags]


rng = np.random.default_rng(0)
y = make_series(3 * 365, rng)
lags = [1, 2, 7, 14]
print("autocorrelation at lags", lags)
print("  raw series:           ", np.round(acf(y, lags), 2))
d = y[M:] - y[:-M]
print("  after lag-7 difference", np.round(acf(d, lags), 2))

origins = np.arange(len(y) - 364, len(y) - H + 1, 5)        # every 5 days for a year
results = {"seasonal naive": [], "Holt-Winters (damped)": [], "seasonal AR": []}
coverage, naive_cov, widths = [], [], []
for o in origins:
    hist, actual = y[:o], y[o:o + H]
    scale = np.mean(np.abs(hist[M:] - hist[:-M]))
    results["seasonal naive"].append(np.abs(actual - hist[-M:][np.arange(H) % M]) / scale)
    params = hw_fit(hist)
    fc, errs = hw_forecast(hist, params, H)
    results["Holt-Winters (damped)"].append(np.abs(actual - fc) / scale)
    p, coef = sarima_fit(hist)
    results["seasonal AR"].append(np.abs(actual - sarima_forecast(hist, p, coef, H)) / scale)
    # 80% intervals: simulated paths vs. a constant +-1.28 one-step sigma band
    paths = hw_simulate(hist, params, H, 400, rng)
    lo, hi = np.quantile(paths, [0.1, 0.9], axis=0)
    coverage.append((actual >= lo) & (actual <= hi))
    widths.append(hi - lo)
    s1 = errs[-365:].std()
    naive_cov.append(np.abs(actual - fc) <= 1.2816 * s1)

print(f"\nlast fit: alpha={params[0]:.2f} beta={params[1]:.3f} gamma={params[2]:.2f} "
      f"phi={params[3]:.2f}; AR order chosen by AIC p={p}")
print(f"\nMASE over {len(origins)} origins x {H} steps (lower is better):")
for name, v in results.items():
    v = np.array(v)
    print(f"  {name:<22} mean {v.mean():.3f}   steps 1-7 {v[:, :7].mean():.3f}   steps 8-14 {v[:, 7:].mean():.3f}")
coverage, naive_cov, widths = map(np.array, (coverage, naive_cov, widths))
print("\nempirical coverage of nominal 80% intervals:")
print(f"  simulated paths, steps 1-3: {coverage[:, :3].mean():.0%}, steps 12-14: {coverage[:, -3:].mean():.0%}"
      f"   (mean width {widths[:, 0].mean():.1f} at step 1, {widths[:, -1].mean():.1f} at step 14)")
print(f"  constant one-step band,  steps 1-3: {naive_cov[:, :3].mean():.0%}, steps 12-14: {naive_cov[:, -3:].mean():.0%}")
