"""Three kinds of leakage, each producing a score that looks better than the truth."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)

# 1. Feature selection before cross-validation. Labels are coin flips, so the
#    true accuracy of any model is 50%.
X = rng.normal(size=(100, 5000))
y = rng.integers(0, 2, size=100)
cv = KFold(5, shuffle=True, random_state=0)
X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)   # sees ALL labels
leaky = cross_val_score(LogisticRegression(max_iter=1000), X_sel, y, cv=cv)
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
honest = cross_val_score(pipe, X, y, cv=cv)                 # selects inside each fold
print("1. Random labels, 5000 noise features, pick 20 'best' features")
print(f"   select-then-CV accuracy: {leaky.mean():.3f}")
print(f"   select-inside-CV (pipeline): {honest.mean():.3f}")

# 2. Group leakage: 60 patients, 10 scans each. The label is a property of the
#    patient, and each patient's scans share a patient-specific "signature".
n_pat, per = 60, 10
pat_label = rng.integers(0, 2, size=n_pat)
signature = rng.normal(size=(n_pat, 20))
groups = np.repeat(np.arange(n_pat), per)
Xg = signature[groups] + 0.3 * rng.normal(size=(n_pat * per, 20))
Xg[:, 0] += 0.5 * pat_label[groups]          # a weak real signal in one feature
yg = pat_label[groups]
rf = RandomForestClassifier(n_estimators=200, random_state=0)
kf = cross_val_score(rf, Xg, yg, cv=KFold(5, shuffle=True, random_state=0))
gkf = cross_val_score(rf, Xg, yg, cv=GroupKFold(5), groups=groups)
print("\n2. 60 patients x 10 scans; does the model work on NEW patients?")
print(f"   KFold (patients split across folds): {kf.mean():.3f}")
print(f"   GroupKFold (each patient in one fold): {gkf.mean():.3f}")

# 3. Target leakage: a feature recorded after the outcome (here, 'was treated',
#    which is almost always 1 for positives) sneaks into the inputs.
n = 2000
Xt = rng.normal(size=(n, 5))
yt = (Xt[:, 0] + rng.normal(scale=1.5, size=n) > 0).astype(int)
treated = np.where(rng.random(n) < 0.95, yt, 1 - yt)        # 95% agrees with label
lr = LogisticRegression(max_iter=1000)
cv5 = KFold(5, shuffle=True, random_state=0)
print("\n3. Target leakage via a post-outcome feature")
print(f"   without 'treated': {cross_val_score(lr, Xt, yt, cv=cv5).mean():.3f}")
print(f"   with 'treated':    "
      f"{cross_val_score(lr, np.column_stack([Xt, treated]), yt, cv=cv5).mean():.3f}")
