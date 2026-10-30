"""Forecasting baselines, rolling-origin evaluation with MASE, and how a
shuffled split flatters a model that cannot forecast."""
import os

# Many small fits: extra threads mostly add overhead, and on a shared machine
# they can slow things down a lot. Must be set before numpy is imported.
for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(var, "4")

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, TimeSeriesSplit

M, H = 7, 14          # seasonal period (weekly, daily data) and forecast horizon


def make_series(n, rng):
    """Daily demand-like series: trend + weekly + yearly seasonality + AR(1) noise."""
    t = np.arange(n)
    weekly = np.array([0, 2, 3, 2, 1, -3, -5])            # shape over Mon..Sun
    noise, eps = np.zeros(n), rng.normal(0, 3, n)
    for i in range(1, n):
        noise[i] = 0.6 * noise[i - 1] + eps[i]
    return 100 + 0.03 * t + 3 * weekly[t % 7] + 8 * np.sin(2 * np.pi * t / 365.25) + noise


# Each forecaster sees only the history before the origin.
def mean_fc(hist, h):
    return np.full(h, hist.mean())


def naive_fc(hist, h):
    return np.full(h, hist[-1])


def snaive_fc(hist, h):
    return hist[-M:][np.arange(h) % M]                     # repeat the last week


def drift_fc(hist, h):
    slope = (hist[-1] - hist[0]) / (len(hist) - 1)
    return hist[-1] + slope * np.arange(1, h + 1)


def mase_scale(hist):
    """In-sample MAE of the one-season-back forecast (Hyndman and Koehler, 2006)."""
    return np.mean(np.abs(hist[M:] - hist[:-M]))


def backtest(y, forecaster, origins):
    """Rolling origin: refit at each origin, forecast H steps, score, move on."""
    abs_err, sq_err, scaled = [], [], []
    for o in origins:
        hist, actual = y[:o], y[o:o + H]
        e = actual - forecaster(hist, H)
        abs_err.append(np.abs(e)); sq_err.append(e ** 2)
        scaled.append(np.abs(e) / mase_scale(hist))
    abs_err = np.array(abs_err)
    return (abs_err.mean(), np.sqrt(np.mean(sq_err)), np.mean(scaled),
            abs_err.mean(axis=0))                           # last: MAE per step


rng = np.random.default_rng(0)
y = make_series(3 * 365, rng)
origins = np.arange(len(y) - 365, len(y) - H + 1)            # every day of the last year
print(f"series length {len(y)}, {len(origins)} forecast origins, horizon {H} days\n")
print(f"{'method':>15} {'MAE':>6} {'RMSE':>6} {'MASE':>6}")
per_step = {}
for name, f in [("mean", mean_fc), ("naive", naive_fc),
                ("seasonal naive", snaive_fc), ("drift", drift_fc)]:
    mae, rmse, mase, steps = backtest(y, f, origins)
    per_step[name] = steps
    print(f"{name:>15} {mae:6.2f} {rmse:6.2f} {mase:6.2f}")
print("\nMAE by step ahead:   1     3     5     7    14")
for name in ["naive", "seasonal naive"]:
    print(f"{name:>15}: " + " ".join(f"{per_step[name][k - 1]:5.2f}" for k in [1, 3, 5, 7, 14]))

# ---- Part 2: a model of the calendar, scored three ways ----------------------
# Random walks with a weekly pattern. Features: time index and day of week.
# Trees can interpolate between neighbouring days but cannot extrapolate.
n, cut = 1000, 900
t = np.arange(n)
X = np.column_stack([t, t % 7])
model = HistGradientBoostingRegressor(max_iter=100, random_state=0)


def cv_mae(splitter, yy):
    errs = []
    for tr, te in splitter.split(X):
        p = model.fit(X[tr], yy[tr]).predict(X[te])
        errs.append(np.mean(np.abs(p - yy[te])))
    return np.mean(errs)


scores = []
for rep in range(10):                                   # 10 independent walks
    yy = 50 + np.cumsum(rng.normal(0, 1, n)) + 3 * np.array([0, 2, 3, 2, 1, -3, -5])[t % 7]
    future = model.fit(X[:cut], yy[:cut]).predict(X[cut:])
    scores.append([cv_mae(KFold(5, shuffle=True, random_state=0), yy),
                   cv_mae(TimeSeriesSplit(5, test_size=100), yy),
                   np.mean(np.abs(future - yy[cut:])),
                   np.mean(np.abs(yy[cut - 7:cut][np.arange(n - cut) % 7] - yy[cut:]))])
scores = np.array(scores)
print("\nGradient boosting on (time index, day of week), 10 random walks, mean MAE:")
for j, name in enumerate(["shuffled 5-fold CV", "TimeSeriesSplit, 5 folds",
                          "true future, last 100 days", "seasonal naive, same 100"]):
    print(f"  {name:<27} {scores[:, j].mean():6.2f}   (range {scores[:, j].min():5.2f} to {scores[:, j].max():5.2f})")
