"""What linear-model coefficients do and do not tell you: units, collinearity,
and omitted variables, on synthetic data where we know the truth.
"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

rng = np.random.default_rng(1)
n = 300

# 1. Units: the same model, distance in metres vs kilometres.
dist_m = rng.uniform(100, 5000, n)
price = 50 - 0.004 * dist_m + rng.normal(0, 2, n)
c_m = LinearRegression().fit(dist_m[:, None], price).coef_[0]
c_km = LinearRegression().fit((dist_m / 1000)[:, None], price).coef_[0]
print(f"1. coefficient per metre {c_m:.5f}, per km {c_km:.3f} (same model, same fit)")

# 2. Collinearity: x2 is almost a copy of x1; y depends only on x1.
x1 = rng.normal(size=n)
x2 = x1 + rng.normal(0, 0.05, n)
y = 3 * x1 + rng.normal(0, 1, n)
X = np.column_stack([x1, x2])
print("2. OLS on the full sample: (w1, w2) =",
      np.round(LinearRegression().fit(X, y).coef_, 2))
ols, rid = [], []
for _ in range(200):                     # refit on bootstrap resamples
    i = rng.integers(0, n, n)
    ols.append(LinearRegression().fit(X[i], y[i]).coef_)
    rid.append(Ridge(alpha=10.0).fit(X[i], y[i]).coef_)
ols, rid = np.array(ols), np.array(rid)
print("   truth (w1, w2) = (3, 0); mean +/- sd over 200 bootstrap refits:")
print(f"   OLS   w1 {ols[:, 0].mean():5.2f} +/- {ols[:, 0].std():.2f}   "
      f"w2 {ols[:, 1].mean():5.2f} +/- {ols[:, 1].std():.2f}   "
      f"w1+w2 {ols.sum(1).mean():.2f} +/- {ols.sum(1).std():.2f}")
print(f"   ridge w1 {rid[:, 0].mean():5.2f} +/- {rid[:, 0].std():.2f}   "
      f"w2 {rid[:, 1].mean():5.2f} +/- {rid[:, 1].std():.2f}   "
      f"w1+w2 {rid.sum(1).mean():.2f} +/- {rid.sum(1).std():.2f}")
print(f"   correlation of OLS w1 and w2 across resamples: {np.corrcoef(ols.T)[0, 1]:.3f}")

# 3. Omitted variable: z drives both x and y. Truly, more x lowers y.
z = rng.normal(size=n)
x = z + rng.normal(0, 0.5, n)
y = -1.0 * x + 3.0 * z + rng.normal(0, 1, n)
c_alone = LinearRegression().fit(x[:, None], y).coef_[0]
c_both = LinearRegression().fit(np.column_stack([x, z]), y).coef_
print(f"3. true effect of x is -1.0; y ~ x alone gives {c_alone:+.2f}; "
      f"y ~ x + z gives {c_both[0]:+.2f} (z: {c_both[1]:+.2f})")
