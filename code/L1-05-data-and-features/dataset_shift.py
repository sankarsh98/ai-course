"""Dataset shift: covariate shift, detecting shift, and correcting label shift.

Part 1: covariate shift breaks a model that was only right where it was trained.
Part 2: a domain classifier tries to tell training rows from new rows.
Part 3: label shift. Estimate the new positive rate with black box shift
        estimation [Lipton et al., 2018] and re-weight the probabilities.
"""
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_predict

rng = np.random.default_rng(0)

# Part 1 -------------------------------------------------------------------
def f(x):
    return np.sin(x) + 0.3 * x            # the true relationship


x_tr = rng.normal(0, 0.8, 3000)
x_new = rng.normal(2.5, 0.8, 3000)        # same f, inputs moved right
y_tr = f(x_tr) + rng.normal(0, 0.1, 3000)
y_new = f(x_new) + rng.normal(0, 0.1, 3000)
lin = LinearRegression().fit(x_tr[:, None], y_tr)
print("Part 1: covariate shift, linear model fit on x ~ N(0, 0.8)")
print(f"  MSE on held-out training-like data: "
      f"{mean_squared_error(f(x_tr) + rng.normal(0, 0.1, 3000), lin.predict(x_tr[:, None])):.3f}")
print(f"  MSE on shifted data x ~ N(2.5, 0.8): {mean_squared_error(y_new, lin.predict(x_new[:, None])):.3f}")


# Part 2 -------------------------------------------------------------------
def domain_auc(A, B):
    # Label rows by origin and ask a classifier to tell them apart (out-of-fold).
    X = np.vstack([A, B])
    d = np.r_[np.zeros(len(A)), np.ones(len(B))]
    p = cross_val_predict(LogisticRegression(max_iter=1000), X, d, cv=5,
                          method="predict_proba")[:, 1]
    return roc_auc_score(d, p)


base = rng.normal(size=(2000, 5))
print("\nPart 2: domain-classifier AUC (0.5 = indistinguishable)")
print(f"  same distribution          : {domain_auc(base, rng.normal(size=(2000, 5))):.3f}")
moved = rng.normal(size=(2000, 5))
moved[:, 3] += 0.3                        # one feature's mean moves by 0.3 std
print(f"  one feature shifted 0.3 std: {domain_auc(base, moved):.3f}")
print(f"  Part 1 inputs              : {domain_auc(x_tr[:, None], x_new[:, None]):.3f}")


# Part 3 -------------------------------------------------------------------
def sample(n, prior):
    # Fixed class-conditional distributions p(x|y); only the prior changes.
    y = (rng.random(n) < prior).astype(int)
    x = rng.normal(size=(n, 2)) + np.where(y[:, None] == 1, 1.2, 0.0)
    return x, y


Xs, ys = sample(10000, 0.5)               # source: balanced
Xv, yv = sample(5000, 0.5)                # source validation set
Xt, yt = sample(10000, 0.1)               # target: only 10% positive, labels unseen
clf = LogisticRegression().fit(Xs, ys)

# BBSE: C[i, j] = P(pred = i, y = j) on source validation data;
# mu[i] = P(pred = i) on unlabeled target data. Solve C w = mu for
# w[j] = p_target(y = j) / p_source(y = j).
pred_v, pred_t = clf.predict(Xv), clf.predict(Xt)
C = np.array([[np.mean((pred_v == i) & (yv == j)) for j in (0, 1)] for i in (0, 1)])
mu = np.array([np.mean(pred_t == i) for i in (0, 1)])
w = np.linalg.solve(C, mu)
prior_hat = w[1] * yv.mean()
print("\nPart 3: label shift, source positive rate 0.5, target 0.1")
print(f"  mean predicted probability on target, unadjusted: {clf.predict_proba(Xt)[:, 1].mean():.3f}")
print(f"  BBSE estimate of target positive rate           : {prior_hat:.3f} (true {yt.mean():.3f})")

# Bayes' rule: p_t(y|x) is proportional to p_s(y|x) * w[y].
ps = clf.predict_proba(Xt)
pt = ps * w
pt = pt[:, 1] / pt.sum(1)
print(f"  Brier score on target, unadjusted: {brier_score_loss(yt, ps[:, 1]):.4f}")
print(f"  Brier score on target, adjusted  : {brier_score_loss(yt, pt):.4f}")
