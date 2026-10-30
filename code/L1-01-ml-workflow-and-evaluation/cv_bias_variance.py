"""Under/overfitting, a measured bias-variance decomposition, and nested CV."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

rng = np.random.default_rng(0)
truth = lambda x: np.sin(2 * np.pi * x)
SIGMA = 0.3                                      # noise std, so irreducible MSE = 0.09

def sample(n):
    x = rng.uniform(0, 1, n)
    return x[:, None], truth(x) + SIGMA * rng.normal(size=n)

def poly(deg):
    # Centre x before taking powers: raw powers of x in [0, 1] are nearly collinear,
    # which makes high-degree least-squares fits numerically fragile.
    return make_pipeline(StandardScaler(), PolynomialFeatures(deg), LinearRegression())

# --- 1. Training error vs cross-validated error vs fresh test error ---
X, y = sample(30)
X_test, y_test = sample(10_000)
print("1. 30 training points from sin(2 pi x) + noise (noise MSE = 0.09)")
print(f"   {'degree':>6} {'train MSE':>10} {'5-fold CV MSE':>14} {'test MSE':>9}")
for deg in [1, 3, 5, 9, 15]:
    m = poly(deg).fit(X, y)
    train = np.mean((m.predict(X) - y) ** 2)
    cv = -cross_val_score(poly(deg), X, y, cv=KFold(5, shuffle=True, random_state=0),
                          scoring="neg_mean_squared_error").mean()
    test = np.mean((m.predict(X_test) - y_test) ** 2)
    print(f"   {deg:6d} {train:10.3f} {cv:14.3f} {test:9.3f}")

# --- 2. Bias^2 and variance, measured by refitting on 300 fresh training sets ---
x_grid = np.linspace(0.05, 0.95, 50)[:, None]
print("\n2. bias-variance at 50 fixed x values, 300 training sets of size 30")
print(f"   {'degree':>6} {'bias^2':>8} {'variance':>9} {'sum + noise':>12}")
for deg in [1, 3, 5, 9]:
    preds = np.array([poly(deg).fit(*sample(30)).predict(x_grid) for _ in range(300)])
    bias2 = np.mean((preds.mean(axis=0) - truth(x_grid[:, 0])) ** 2)
    var = np.mean(preds.var(axis=0))
    print(f"   {deg:6d} {bias2:8.3f} {var:9.3f} {bias2 + var + SIGMA ** 2:12.3f}")

# --- 3. Tuning and evaluating on the same folds is optimistic ---
# Random labels: no model can beat 50% on new data.
grid = {"C": np.logspace(-2, 3, 6), "gamma": np.logspace(-4, 1, 6)}
non_nested, nested = [], []
for trial in range(20):
    Xr = rng.normal(size=(60, 10))
    yr = rng.integers(0, 2, size=60)
    inner = KFold(5, shuffle=True, random_state=trial)
    outer = KFold(5, shuffle=True, random_state=100 + trial)
    search = GridSearchCV(SVC(), grid, cv=inner).fit(Xr, yr)
    non_nested.append(search.best_score_)               # best of 36 noisy scores
    nested.append(cross_val_score(GridSearchCV(SVC(), grid, cv=inner), Xr, yr,
                                  cv=outer).mean())     # tuning inside each outer fold
print("\n3. random labels, 60 samples, 36 SVM settings, 20 repeats")
print(f"   non-nested CV (best grid score): mean {np.mean(non_nested):.3f}")
print(f"   nested CV:                       mean {np.mean(nested):.3f}")
