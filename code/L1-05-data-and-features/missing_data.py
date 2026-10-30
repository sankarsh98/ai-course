"""Missing data: MCAR vs MAR vs MNAR, and what each imputation strategy does.

x (always observed) predicts v. We hide about 40% of v under three mechanisms
and compare estimated mean, std and corr(x, v) with the simulated truth, then
test whether a missingness indicator helps prediction.
"""
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
n = 20000
x = rng.normal(size=n)
v = 2.0 * x + rng.normal(size=n)          # true mean 0, std sqrt(5) = 2.24


def hide(mechanism):
    if mechanism == "MCAR":                # unrelated to anything
        p = np.full(n, 0.4)
    elif mechanism == "MAR":               # depends on the observed x only
        p = 1 / (1 + np.exp(-(2 * x - 0.5)))
    else:                                  # MNAR: depends on v itself
        p = 1 / (1 + np.exp(-(v - 0.5)))
    return rng.random(n) < p


def impute(method, X):
    if method == "complete cases":
        return X[~np.isnan(X[:, 1])]
    if method == "mean impute":
        return SimpleImputer(strategy="mean").fit_transform(X)
    # Regress v on x. Plain regression fills in points exactly on the line;
    # sample_posterior=True adds noise drawn from the fitted model instead.
    stochastic = method == "regression + noise"
    return IterativeImputer(sample_posterior=stochastic, random_state=0).fit_transform(X)


print(f"truth: mean {v.mean():.2f}  std {v.std():.2f}  corr(x,v) {np.corrcoef(x, v)[0, 1]:.2f}\n")
print("mechanism  method              %missing   mean    std   corr(x,v)")
for mech in ("MCAR", "MAR", "MNAR"):
    miss = hide(mech)
    X = np.column_stack([x, np.where(miss, np.nan, v)])
    for method in ("complete cases", "mean impute", "regression impute", "regression + noise"):
        Z = impute(method, X)
        print(f"{mech:9s}  {method:18s} {miss.mean():7.0%}  {Z[:, 1].mean():6.2f} "
              f"{Z[:, 1].std():6.2f}  {np.corrcoef(Z[:, 0], Z[:, 1])[0, 1]:8.2f}")
    print()

# Prediction: target depends on v. Under MNAR, *whether* v is missing carries
# information about v, so a missingness indicator column can help.
target = v + rng.normal(scale=0.5, size=n)
print("Predicting a target that depends on v (5-fold CV R^2):")
for mech in ("MCAR", "MNAR"):
    X = np.column_stack([x, np.where(hide(mech), np.nan, v)])
    for flag in (False, True):
        model = make_pipeline(SimpleImputer(add_indicator=flag), LinearRegression())
        r2 = cross_val_score(model, X, target, cv=5, scoring="r2").mean()
        print(f"  {mech:5s} mean impute{' + indicator' if flag else '            '}  R^2 = {r2:.3f}")
