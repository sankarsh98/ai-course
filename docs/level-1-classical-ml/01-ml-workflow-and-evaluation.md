# The ML workflow and evaluation

> **Level 1 · Module 1** · Prerequisites: L0-01, L0-04 · Time: 5 to 7 hours · Code: `code/L1-01-ml-workflow-and-evaluation/`

## What you will be able to do

- Frame a prediction problem as a decision with a baseline and a metric that reflects the cost of errors.
- Split data so the evaluation answers the deployment question, including grouped and time-ordered data.
- Detect and prevent the common forms of data leakage.
- Choose and interpret classification metrics (precision, recall, ROC and PR curves, calibration) and regression metrics.
- Use cross-validation correctly, including nested CV for tuning, and explain bias and variance with measured numbers.
- Report results with honest uncertainty and compare two models fairly.

## 1. Framing the problem

Before any model, write down four things.

1. **The decision.** What will someone do differently because of the prediction, such as flagging a transaction or ordering stock?
2. **The target, and the inputs available at prediction time.** A feature recorded after the event cannot be used, however predictive it looks (section 4).
3. **The cost of each error.** A missed fraud and a false alarm cost different amounts. That asymmetry decides the metric and the threshold.
4. **The unit of generalization.** Will the model meet new rows, new customers, new hospitals or next month? This decides how you split (section 3).

## 2. Baselines

A **baseline** is the simplest reasonable predictor. The trivial ones are the majority class or the training mean, and a plain linear model is the next step up. Baselines show whether a complex model learns anything. On scikit-learn's breast cancer data (569 tumours, 30 features), always predicting "benign" scores 62% accuracy (script 1). On the diabetes regression data, predicting the mean gives $R^2 \approx 0$, and linear regression reaches 0.36. "96% accuracy" means little until you know the baseline.

Baselines matter in research too. In a civil war prediction case study, Kapoor and Narayanan [2023] found that published claims of complex models beating logistic regression failed to reproduce because of leakage. Once it was fixed, the complex models were not substantively better than the old logistic regression.

## 3. Train, validation and test sets

Training error is optimistic, because the model was fitted to those points. So hold data out:

- the **training set** fits parameters,
- the **validation set** chooses among models and hyperparameters (settings fixed before training, like regularization strength),
- the **test set** is used **once**, at the end, to estimate the performance of the final choice.

If you look at the test score, change something and look again, the test set has become a validation set and its score is optimistic. After choosing hyperparameters, you may refit on training plus validation data and then evaluate once on test, as script 1 does.

**The split must mimic deployment.**

- **Random splits** suit independent rows from a stable population. **Stratify** classification splits to keep the class proportions.
- **Group splits** keep every row of a patient, user or device on one side. Without them, the model can recognise a group it has already seen.
- **Time splits** train on the past and test on the future. Random splits of time-ordered data leak the future (L3-09).

**One split is a noisy measurement.** In script 1 the same model scored anywhere from 0.956 to 1.000 across 20 random 80/20 splits. With 114 test examples, one mistake is worth almost 0.9 points.

## 4. Data leakage

**Leakage** is information that would not be available at prediction time reaching the model during training or evaluation, which makes the estimate better than reality. Kaufman et al. [2012] formalise it and propose **learn-predict separation**: build everything the model uses only from data that would exist at the moment of prediction. Surveying earlier reviews, Kapoor and Narayanan [2023] found leakage affecting at least 294 papers in 17 fields. Their eight-type taxonomy includes:

- no real test set, or preprocessing or feature selection fitted on training and test data together,
- duplicates across the split,
- illegitimate features (target proxies, or information recorded after the outcome),
- temporal leakage, non-independence between train and test rows, and test sets with sampling bias.

The first kind has a mechanical fix. Every step that learns from data (scaling, imputation, feature selection, target encoding) must be fitted on the training portion of each split only. scikit-learn's `Pipeline` does this inside cross-validation [scikit-learn, Common pitfalls]. ESL section 7.10.2, "The Wrong and Right Way to Do Cross-validation", makes the same point with pure noise [Hastie et al., 2009].

Script 2 shows three leaks. Feature selection on all labels turns coin-flip labels into 85% "accuracy". Random splits of patients' scans give 99.7% where new patients get 57%. A post-diagnosis "treated" feature lifts 68% to 96%. When a result looks too good, find the feature doing the work and ask when it becomes known.

## 5. Classification metrics

A classifier outputs a score $\hat p$, and you predict positive when $\hat p \ge t$ for a threshold $t$. The **confusion matrix** counts true positives (TP), false positives (FP), true negatives (TN) and false negatives (FN). From it:

- **Accuracy** $= (TP + TN)/N$, which is misleading under imbalance.
- **Precision** $= TP/(TP + FP)$: what fraction of the alarms are real?
- **Recall** (true positive rate, TPR) $= TP/(TP + FN)$: what fraction of the real cases are caught?
- **False positive rate** (FPR) $= FP/(FP + TN)$.
- **F1**, the harmonic mean of precision and recall. It is one number, but it hides the trade-off.

In script 3, with 2.5% positives, "always negative" scores 97.5% accuracy. Logistic regression at $t = 0.5$ scores 97.7% but catches only 46 of 499 positives (recall 0.09). At $t = 0.05$ it catches 297, at the price of 2,034 false alarms, and accuracy *falls* to 88.8%. Accuracy rewarded the useless behaviour. **Choose the threshold from error costs.** If a miss costs 20 false alarms, pick the $t$ that minimises $20 \cdot FN + FP$ on validation data.

**ROC curve and ROC-AUC.** Plot TPR against FPR as $t$ sweeps. The area under the curve equals the probability that a random positive scores above a random negative [Fawcett, 2006], so it is 0.5 for random scores and 1 for perfect ranking. TPR and FPR are each computed within one class, so ROC-AUC does not depend on the class ratio.

**Precision-recall curve and average precision (AP).** Plot precision against recall. Random scoring gives AP equal to the positive rate. Precision mixes the classes, so it reacts to imbalance. In script 3, one fixed random forest keeps ROC-AUC at 0.988 as test positives are thinned from 50% to 0.5%, while AP falls from 0.98 to 0.35. The model is unchanged, but at 0.5% prevalence most of its alarms are false. Saito and Rehmsmeier [2015] use simulations to argue that PR plots are more informative than ROC plots on imbalanced data. Report both, with the prevalence.

## 6. Calibration

A model is **calibrated** if, among cases where it says 0.7, about 70% are positive. AUC ignores calibration, but decisions that weigh probabilities against costs need it. A **reliability diagram** bins predictions and plots each bin's mean prediction against its observed positive rate, and a calibrated model lies on the diagonal. The usual summaries are:

- the **Brier score** $\frac1N \sum_i (\hat p_i - y_i)^2$,
- **log loss** $-\frac1N \sum_i [y_i \log \hat p_i + (1 - y_i)\log(1 - \hat p_i)]$ (the cross-entropy of L0-04), which punishes confident mistakes hard,
- **expected calibration error (ECE)**, the bin-weighted gap between confidence and frequency, which depends on the binning.

Models err in characteristic ways. Niculescu-Mizil and Caruana [2005] found that boosted trees and SVMs squeeze probabilities toward the middle (a sigmoid-shaped distortion), that naive Bayes pushes them toward 0 and 1, and that recalibration fixes much of this. Script 3 agrees. Naive Bayes says 0.95 where the truth is 0.84, and the random forest says 0.30 where the truth is 0.12, despite the best AUC. Guo et al. [2017] found modern deep networks overconfident and proposed **temperature scaling**, which divides the logits by one learned constant.

**Recalibration** fits a monotone map from scores to probabilities on held-out data. **Platt scaling** fits a logistic curve, and **isotonic regression** fits any non-decreasing step function. scikit-learn's guide notes that isotonic overfits more easily and needs roughly more than 1,000 calibration samples to beat the sigmoid [scikit-learn, Probability calibration]. In script 3, isotonic calibration cut naive Bayes's ECE from 0.065 to 0.007 without changing AUC.

## 7. Regression metrics

For targets $y_i$ and predictions $\hat y_i$:

- **MAE** $= \frac1N\sum|y_i - \hat y_i|$ is in target units and robust to outliers.
- **RMSE** $= \sqrt{\frac1N\sum(y_i - \hat y_i)^2}$ is also in target units, but weights big misses more.
- **$R^2$** $= 1 - \sum(y_i - \hat y_i)^2 / \sum(y_i - \bar y)^2$ is the variance explained relative to predicting the mean. It is 0 for the mean predictor and negative for anything worse.

Pick the one that matches cost: RMSE if a 20-unit miss is four times as bad as a 10-unit miss, MAE if it is twice as bad.

## 8. Cross-validation

**$k$-fold cross-validation (CV)** splits the data into $k$ folds, trains on $k - 1$ and scores the held-out fold, rotating through all $k$. You report the mean and spread. $k = 5$ or 10 is typical. Stratified, grouped and time-series variants exist for the reasons in section 3.

**What CV estimates.** Bates, Hastie and Tibshirani [2024] show for linear regression that CV estimates the average error of models trained on *other* datasets of the same size, not the error of your particular fitted model. They also show that the naive interval (mean $\pm 1.96$ standard errors across folds) can cover far less than 95% of the time, because folds share training data. They propose a nested-CV variance estimate that fixes coverage in their examples. ESL section 7.12 discusses the same conditional-versus-expected error question [Hastie et al., 2009].

**Tuning inside CV.** The best of many CV scores is biased upward, because it was selected for being high. Varma and Simon [2006] showed the bias can be large on small data and recommended **nested CV**, where an inner loop tunes using only the outer training fold and the outer loop scores the whole procedure. In script 4, on random labels, the best of 36 SVM settings averaged 59.6%, and nested CV gave 51.1%.

## 9. Bias, variance, under- and overfitting

For squared error, the expected error at $x$, averaged over training sets and noise, decomposes as [Hastie et al., 2009, ch. 7]

$$\mathbb{E}\big[(y - \hat f(x))^2\big] = \big(\mathbb{E}[\hat f(x)] - f(x)\big)^2 + \mathrm{Var}[\hat f(x)] + \sigma^2,$$

where $f$ is the true function, $\hat f$ the fitted model (random because the training set is), and $\sigma^2$ the irreducible noise. The first term is **bias²**, systematic error from a model too simple for $f$. The second is **variance**, sensitivity to the particular sample. More flexibility usually trades bias for variance. **Underfitting** (high bias) shows as training and validation errors both high and close. **Overfitting** (high variance) shows as low training error and much higher validation error.

Script 4 fits polynomials to 30 points from $\sin 2\pi x$ plus noise ($\sigma^2 = 0.09$). Degree 1 underfits (bias² 0.154). Degree 3 is close to ideal (test MSE 0.105). At degree 9, the variance averaged over 300 training sets is 0.71. Training error keeps falling as degree grows, so it cannot choose complexity. Also, with only 30 points, CV ranked degree 5 best (0.064, below the noise floor, which is luck), while degree 3 did better on fresh data. CV is an estimate with its own variance.

This U-shaped picture fits classical models. Very large models such as deep networks can interpolate their training data and still generalize, which L0-05 discusses briefly.

## 10. Reporting uncertainty

- **One accuracy** on $n$ test examples has standard error $\sqrt{a(1-a)/n}$. In script 5 the 95% interval is about $\pm 8$ points at $n = 100$ and $\pm 0.7$ at $n = 10{,}000$. The **bootstrap** (resample the test set with replacement, recompute, take quantiles) agrees and works for any metric. The $n = 100$ draw came out at 0.77 against a true 0.85: small test sets really do miss by that much.
- **Two models, one test set**: analyse the *paired* per-example differences. In script 5 the separate intervals overlap ([0.868, 0.922] and [0.888, 0.938]), yet the paired difference is $+0.018$ with interval $[+0.004, +0.034]$, because the models mostly succeed and fail on the same examples. **McNemar's test** uses only the disagreements (3 against 12, $p = 0.039$). Overlapping intervals do not mean "no difference" (L0-04 covers the testing basics).
- **Training randomness and splits**: rerun with several seeds and splits, and report the spread. Remember that fold-to-fold spread understates uncertainty [Bates et al., 2024].
- **Write down** the data version, split method, metric and why, baseline scores, how many configurations you tried, intervals and seeds. Kapoor and Narayanan [2023] propose "model info sheets" for this. LLM and benchmark evaluation are covered in L4-03.

## Hands-on

The scripts use bundled or synthetic data (no downloads). Run them from `code/L1-01-ml-workflow-and-evaluation/` with the course Python (scikit-learn 1.9.1 here). Each takes under about 35 seconds on a laptop CPU. Trailing digits may differ with other platforms or versions.

### Script 1: baselines and splits

```python
"""Baselines first, then a train/validation/test workflow, and how much one split can lie."""
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
print(f"breast cancer: {X.shape[0]} samples, {X.shape[1]} features, "
      f"{y.mean():.1%} labelled benign (class 1)")

# 60/20/20 split: test is locked away; validation is for choosing C.
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=0)

base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
print(f"\nmajority-class baseline, validation accuracy = {base.score(X_val, y_val):.3f}")

print("choosing C for logistic regression on the validation set:")
best = None
for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    acc = model.fit(X_tr, y_tr).score(X_val, y_val)
    print(f"  C = {C:<6} validation accuracy = {acc:.3f}")
    if best is None or acc > best[0]:
        best = (acc, C)
C = best[1]
# Refit on train+validation with the chosen C, then touch the test set exactly once.
final = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X_tmp, y_tmp)
print(f"chosen C = {C}; test accuracy (reported once) = {final.score(X_test, y_test):.3f}")

print("\nsame model, 20 different random 80/20 splits:")
accs = []
for seed in range(20):
    a, b, c, d = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(a, c)
    accs.append(m.score(b, d))
print(f"  test accuracy min {min(accs):.3f}, max {max(accs):.3f}, "
      f"mean {np.mean(accs):.3f}, std {np.std(accs, ddof=1):.3f}")

Xr, yr = load_diabetes(return_X_y=True)
a, b, c, d = train_test_split(Xr, yr, test_size=0.25, random_state=0)
print(f"\ndiabetes regression: {Xr.shape[0]} samples, target range {yr.min():.0f} to {yr.max():.0f}")
for name, reg in [("predict-the-mean", DummyRegressor()), ("linear regression", LinearRegression())]:
    p = reg.fit(a, c).predict(b)
    print(f"  {name:>17}: MAE {mean_absolute_error(d, p):6.1f}  "
          f"RMSE {root_mean_squared_error(d, p):6.1f}  R^2 {r2_score(d, p):6.3f}")
```

```text
breast cancer: 569 samples, 30 features, 62.7% labelled benign (class 1)

majority-class baseline, validation accuracy = 0.623
choosing C for logistic regression on the validation set:
  C = 0.001  validation accuracy = 0.868
  C = 0.01   validation accuracy = 0.965
  C = 0.1    validation accuracy = 0.974
  C = 1      validation accuracy = 0.991
  C = 10     validation accuracy = 0.974
  C = 100    validation accuracy = 0.974
chosen C = 1; test accuracy (reported once) = 0.982

same model, 20 different random 80/20 splits:
  test accuracy min 0.956, max 1.000, mean 0.980, std 0.012

diabetes regression: 442 samples, target range 25 to 346
   predict-the-mean: MAE   58.3  RMSE   70.5  R^2 -0.000
  linear regression: MAE   45.1  RMSE   56.4  R^2  0.359
```

Three values of C tie at 0.974, and the winner leads by two validation examples out of 114, so even the choice of C is noisy. The single test score of 0.982 is one draw from the 0.956 to 1.000 range below it. The mean predictor's $R^2$ is slightly negative because it applies the training mean to test data.

### Script 2: three leaks

```python
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
```

```text
1. Random labels, 5000 noise features, pick 20 'best' features
   select-then-CV accuracy: 0.850
   select-inside-CV (pipeline): 0.460

2. 60 patients x 10 scans; does the model work on NEW patients?
   KFold (patients split across folds): 0.997
   GroupKFold (each patient in one fold): 0.568

3. Target leakage via a post-outcome feature
   without 'treated': 0.681
   with 'treated':    0.960
```

The 0.460 is below 0.5 by chance, since with 100 examples fold noise is several points.

### Script 3: metrics and calibration

```python
"""Accuracy vs precision/recall, ROC-AUC vs PR-AUC under imbalance, and calibration."""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def data(pos_rate, n=40_000, seed=0):
    X, y = make_classification(n_samples=n, n_features=20, n_informative=5,
                               weights=[1 - pos_rate], flip_y=0.01, random_state=seed)
    return train_test_split(X, y, test_size=0.5, stratify=y, random_state=seed)

# --- 1. Thresholds on an imbalanced problem (about 2% positives) ---
X_tr, X_te, y_tr, y_te = data(0.02)
p = LogisticRegression(max_iter=2000).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
print(f"1. test set: {len(y_te)} examples, {y_te.sum()} positives ({y_te.mean():.1%})")
print(f"   always-negative accuracy: {1 - y_te.mean():.3f}")
print(f"   {'threshold':>9} {'accuracy':>9} {'precision':>10} {'recall':>7}  TN FP FN TP")
for t in [0.5, 0.2, 0.1, 0.05]:
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    print(f"   {t:9.2f} {(tn + tp) / len(y_te):9.3f} "
          f"{precision_score(y_te, pred, zero_division=0):10.3f} "
          f"{recall_score(y_te, pred):7.3f}  {tn} {fp} {fn} {tp}")

# --- 2. Same model, same score distributions, different prevalence ---
# Train once, then build test sets with fewer positives by dropping some of them.
# ROC-AUC depends only on how positive and negative scores compare, so it stays put.
a, b, c, d = data(0.5)
s = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0).fit(a, c).predict_proba(b)[:, 1]
rng = np.random.default_rng(0)
pos, neg = np.flatnonzero(d == 1), np.flatnonzero(d == 0)
print("\n2. one random forest, test sets with fewer and fewer positives")
print(f"   {'pos rate':>8} {'ROC-AUC':>8} {'avg precision':>14}")
for rate in [0.5, 0.1, 0.02, 0.005]:
    k = int(rate / (1 - rate) * len(neg))          # positives to keep for this rate
    idx = np.concatenate([neg, rng.choice(pos, size=min(k, len(pos)), replace=False)])
    # A random scorer has ROC-AUC 0.5 and average precision equal to the positive rate.
    print(f"   {d[idx].mean():8.3f} {roc_auc_score(d[idx], s[idx]):8.3f} "
          f"{average_precision_score(d[idx], s[idx]):14.3f}")

# --- 3. Calibration: do predicted probabilities match observed frequencies? ---
X_tr, X_te, y_tr, y_te = data(0.3)
models = {
    "logistic": LogisticRegression(max_iter=2000),
    "naive Bayes": GaussianNB(),
    "NB + isotonic": CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5),
    "random forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0),
}
print("\n3. calibration on a 30%-positive problem")
print(f"   {'model':>14} {'ROC-AUC':>8} {'Brier':>7} {'log loss':>9} {'ECE':>6}")
reliability = {}
for name, m in models.items():
    q = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    # Expected calibration error: bin-weighted gap between confidence and frequency.
    bins = np.minimum((q * 10).astype(int), 9)
    ece = sum(abs(q[bins == k].mean() - y_te[bins == k].mean()) * (bins == k).mean()
              for k in range(10) if (bins == k).any())
    print(f"   {name:>14} {roc_auc_score(y_te, q):8.3f} {brier_score_loss(y_te, q):7.4f} "
          f"{log_loss(y_te, q):9.4f} {ece:6.3f}")
    reliability[name] = calibration_curve(y_te, q, n_bins=5)
print("\n   reliability table (mean predicted -> observed fraction positive, 5 bins)")
for name, (frac, mean_pred) in reliability.items():
    pairs = "  ".join(f"{mp:.2f}->{fr:.2f}" for fr, mp in zip(frac, mean_pred))
    print(f"   {name:>14}: {pairs}")
```

```text
1. test set: 20000 examples, 499 positives (2.5%)
   always-negative accuracy: 0.975
   threshold  accuracy  precision  recall  TN FP FN TP
        0.50     0.977      0.767   0.092  19487 14 453 46
        0.20     0.973      0.428   0.273  19319 182 363 136
        0.10     0.950      0.236   0.449  18777 724 275 224
        0.05     0.888      0.127   0.595  17467 2034 202 297

2. one random forest, test sets with fewer and fewer positives
   pos rate  ROC-AUC  avg precision
      0.500    0.987          0.984
      0.100    0.988          0.896
      0.020    0.989          0.636
      0.005    0.988          0.347

3. calibration on a 30%-positive problem
            model  ROC-AUC   Brier  log loss    ECE
         logistic    0.906  0.1119    0.3549  0.017
      naive Bayes    0.901  0.1213    0.3969  0.065
    NB + isotonic    0.901  0.1143    0.3624  0.007
    random forest    0.986  0.0386    0.1686  0.050

   reliability table (mean predicted -> observed fraction positive, 5 bins)
         logistic: 0.06->0.05  0.29->0.33  0.50->0.53  0.70->0.70  0.90->0.86
      naive Bayes: 0.04->0.08  0.29->0.35  0.50->0.45  0.70->0.55  0.95->0.84
    NB + isotonic: 0.05->0.05  0.30->0.31  0.49->0.47  0.70->0.69  0.90->0.90
    random forest: 0.03->0.01  0.30->0.12  0.50->0.49  0.71->0.86  0.92->0.99
```

In part 3 the forest's reliability row (0.30 goes to 0.12, 0.71 goes to 0.86) is the under-confident S-shape, and naive Bayes is over-confident at the top. Isotonic calibration left AUC unchanged because it only remaps scores monotonically.

### Script 4: overfitting, bias-variance and nested CV

```python
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
```

```text
1. 30 training points from sin(2 pi x) + noise (noise MSE = 0.09)
   degree  train MSE  5-fold CV MSE  test MSE
        1      0.370          0.415     0.321
        3      0.099          0.149     0.105
        5      0.041          0.064     0.141
        9      0.040          0.113     0.142
       15      0.038          2.846     0.147

2. bias-variance at 50 fixed x values, 300 training sets of size 30
   degree   bias^2  variance  sum + noise
        1    0.154     0.022        0.266
        3    0.003     0.013        0.106
        5    0.000     0.032        0.122
        9    0.003     0.707        0.800

3. random labels, 60 samples, 36 SVM settings, 20 repeats
   non-nested CV (best grid score): mean 0.596
   nested CV:                       mean 0.511
```

The CV error of 2.846 at degree 15 comes from folds where the polynomial extrapolates wildly at the data's edge, so fold scores can have heavy tails. Part 2's "sum + noise" is the predicted test error, and its minimum at degree 3 (0.106) matches part 1's best test MSE. Part 1's single training set was lucky for degree 9: its test MSE of 0.142 is far below the 0.80 expected on average. That gap is the variance term.

### Script 5: uncertainty and paired comparison

```python
"""Putting error bars on a test score, and comparing two models on the same test set."""
import numpy as np
from scipy.stats import chi2 as chi2_dist
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)

def bootstrap_ci(values, n_boot=10_000, level=0.95):
    """Percentile bootstrap CI for the mean of per-example values (e.g. 0/1 correct)."""
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return values.mean(), lo, hi

# --- 1. How wide is the interval for a test-set accuracy? ---
print("1. 95% intervals for an accuracy of about 0.85")
print(f"   {'n test':>7} {'observed':>9} {'normal approx':>16} {'bootstrap':>16}")
for n in [100, 1000, 10_000]:
    correct = (rng.random(n) < 0.85).astype(float)
    acc = correct.mean()
    half = 1.96 * np.sqrt(acc * (1 - acc) / n)          # binomial standard error
    _, lo, hi = bootstrap_ci(correct)
    print(f"   {n:7d} {acc:9.3f}   [{acc - half:.3f}, {acc + half:.3f}]   [{lo:.3f}, {hi:.3f}]")

# --- 2. Two models, one test set: use paired differences ---
X, y = make_classification(n_samples=3000, n_features=20, n_informative=6,
                           flip_y=0.05, random_state=1)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=500, random_state=1)
a = LogisticRegression(max_iter=1000).fit(X_tr, y_tr).predict(X_te) == y_te
b = RandomForestClassifier(n_estimators=300, random_state=0).fit(X_tr, y_tr).predict(X_te) == y_te
print(f"\n2. test n = {len(y_te)}: logistic acc {a.mean():.3f}, random forest acc {b.mean():.3f}")
for name, v in [("logistic", a), ("forest", b)]:
    m, lo, hi = bootstrap_ci(v.astype(float))
    print(f"   {name:>8} alone:      {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
m, lo, hi = bootstrap_ci(b.astype(float) - a.astype(float))
print(f"   paired difference: {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
# McNemar's test uses only the examples where the two models disagree.
chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
print(f"   disagreements: logistic-only right {n01}, forest-only right {n10}, "
      f"McNemar chi2 = {chi2:.2f}, p = {chi2_dist.sf(chi2, df=1):.3f}")

# --- 3. Is the gap stable across splits? (Splits overlap, so these are not
#        independent replications; treat the spread as a rough guide only.) ---
gaps = []
for seed in range(10):
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=500, random_state=seed)
    la = LogisticRegression(max_iter=1000).fit(Xa, ya).score(Xb, yb)
    fa = RandomForestClassifier(n_estimators=300, random_state=seed).fit(Xa, ya).score(Xb, yb)
    gaps.append(fa - la)
print(f"\n3. forest minus logistic over 10 random splits: "
      f"mean {np.mean(gaps):+.3f}, min {min(gaps):+.3f}, max {max(gaps):+.3f}")
```

```text
1. 95% intervals for an accuracy of about 0.85
    n test  observed    normal approx        bootstrap
       100     0.770   [0.688, 0.852]   [0.690, 0.850]
      1000     0.852   [0.830, 0.874]   [0.830, 0.873]
     10000     0.851   [0.844, 0.858]   [0.844, 0.858]

2. test n = 500: logistic acc 0.896, random forest acc 0.914
   logistic alone:      0.896  95% CI [0.868, 0.922]
     forest alone:      0.914  95% CI [0.888, 0.938]
   paired difference: +0.018  95% CI [+0.004, +0.034]
   disagreements: logistic-only right 3, forest-only right 12, McNemar chi2 = 4.27, p = 0.039

3. forest minus logistic over 10 random splits: mean +0.028, min +0.016, max +0.040
```

The forest led on all 10 splits, consistent with the paired analysis. Because the splits overlap, treat this as support, not 10 independent replications.

## Common mistakes

- **Tuning on the test set**, even informally. Use validation data or CV for every decision.
- **Preprocessing before splitting.** Scalers, imputers, selectors and target encoders fitted on all data leak. Put them in a `Pipeline`.
- **Random splits for grouped or temporal data.** Split the way "new" works in deployment.
- **Accuracy on imbalanced data** reported without the baseline, precision and recall.
- **Defaulting to threshold 0.5.** The right threshold depends on costs and prevalence.
- **Treating AUC as the whole story.** A model can rank well yet be miscalibrated, or have poor precision at deployment prevalence (L1-05 covers shift).
- **Reporting the best of many CV scores.** Use nested CV or a fresh test set.
- **No intervals, no seeds.** A one-point gain on 500 examples may be noise.

## Exercises

1. [warm-up] A test set has 990 negatives and 10 positives. A classifier flags 20 cases, 8 correctly. Compute accuracy, precision and recall.
2. [warm-up] In script 1, use `DummyClassifier(strategy="stratified")`. Predict its accuracy first.
3. [core] In script 2 case 1, use 50 samples and `k=200`. How does the leaky estimate change?
4. [core] In script 2 case 2, raise the scan noise from 0.3 to 3. What happens to the KFold versus GroupKFold gap, and why?
5. [core] For script 3's logistic model, choose the threshold minimising $20 \cdot FN + FP$ on a validation split, then report the test cost.
6. [core] Add Platt scaling (`method="sigmoid"`) for the random forest in script 3. Compare ECE and AUC.
7. [core] Repeat script 4 part 1 with five seeds. How often does CV pick the degree with the best test MSE?
8. [core] Bootstrap a 95% interval for ROC-AUC on script 3's 2.5%-positive test set, and compare its width with a balanced test set.
9. [stretch] Implement nested CV by hand with two `KFold` loops for script 4's SVM, and check that it matches `cross_val_score(GridSearchCV(...))`.
10. [stretch] Build a time leak: a random-walk feature and a target depending on its future change. Compare `KFold(shuffle=True)` with `TimeSeriesSplit`.

### Hints and answers

1. TP = 8, FP = 12, FN = 2, TN = 978. Accuracy is 0.986, precision 0.4, recall 0.8.
2. It guesses class 1 with probability 0.627, so expected accuracy is $0.627^2 + 0.373^2 \approx 0.53$, below the majority baseline.
3. The leaky estimate rises further toward 1, while the pipeline stays near 0.5.
4. Noisier scans make patients harder to recognise, so the gap shrinks.
5. The best threshold will be far below 0.5, likely a few percent. Choose it on validation data and test once.
6. Expect a lower ECE, since the forest's distortion is S-shaped, and essentially unchanged AUC, since the sigmoid map is strictly monotonic.
7. Expect degree 3 or 5 most often, with some misses. Small data makes selection noisy.
8. With about 500 positives versus about 10,000, the interval is several times wider. AUC uncertainty is driven by the smaller class.
9. Run `GridSearchCV` on each outer-train part, then score its refit model on the outer-test part.
10. Shuffled KFold will score much higher, because `TimeSeriesSplit` only trains on the past.

## Check your understanding

**1. Why use the test set only once?** Each decision made from a dataset's score adapts the model to that dataset's quirks. After a few rounds, the test score is as optimistic as a validation score.

**2. A fraud model with 0.1% fraud reports ROC-AUC 0.99. What do you ask?** Ask for precision at the operating threshold and AP at realistic prevalence, since most alarms could still be false. Also ask how the data was split (by customer, by time) and when each feature is recorded, because a number that high invites a leakage check.

**3. How do you tell bias from variance?** If training and validation errors are both high, the problem is mostly bias: add flexibility or features. If training error is far below validation error, it is mostly variance: add data, regularize or simplify.

**4. Why does nested CV score lower than the best grid-search score?** The best grid score is the maximum of many noisy estimates, so selection biases it upward. Nested CV scores the whole procedure, search included, on data the search never saw.

**5. Models score 0.896 and 0.914 on the same 500 examples, and their intervals overlap. Is the gap real?** Overlap does not settle it, because the errors are correlated. A paired bootstrap or McNemar's test does. In script 5 the paired interval $[+0.004, +0.034]$ excludes zero. Then check that the gap holds across seeds and splits.

## Further reading

- **Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor, "An Introduction to Statistical Learning, with Applications in Python"** (2023, Springer; free PDF). https://www.statlearning.com/
  *Read for:* the gentlest full treatment of resampling, CV and bias-variance, with Python labs. *Level:* intro.
- **Trevor Hastie, Robert Tibshirani, Jerome Friedman, "The Elements of Statistical Learning", 2nd ed., Chapter 7** (2009, Springer; free PDF). https://hastie.su.domains/ElemStatLearn/
  *Read for:* the bias-variance decomposition, right and wrong CV (7.10.2), and conditional versus expected error (7.12). *Level:* intermediate.
- **scikit-learn developers, "Common pitfalls and recommended practices"** (scikit-learn 1.9.1 User Guide, checked October 2026). https://scikit-learn.org/stable/common_pitfalls.html
  *Read for:* runnable examples of preprocessing leakage and pipelines. *Level:* intro.
- **scikit-learn developers, "Probability calibration"** (scikit-learn 1.9.1 User Guide, checked October 2026). https://scikit-learn.org/stable/modules/calibration.html
  *Read for:* reliability diagrams and sigmoid versus isotonic calibration. *Level:* intro.
- **Shachar Kaufman, Saharon Rosset, Claudia Perlich, Ori Stitelman, "Leakage in Data Mining: Formulation, Detection, and Avoidance"** (2012, ACM Transactions on Knowledge Discovery from Data 6(4); publisher version may be paywalled). https://doi.org/10.1145/2382577.2382579
  *Read for:* a formal definition of leakage and the learn-predict separation principle. *Level:* intermediate.
- **Sayash Kapoor, Arvind Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science"** (2023, Patterns 4(9):100804; open access; arXiv:2207.07048). https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/
  *Read for:* the eight-type leakage taxonomy, its prevalence, and model info sheets. *Level:* intro.
- **Sudhir Varma, Richard Simon, "Bias in error estimation when using cross-validation for model selection"** (2006, BMC Bioinformatics 7:91; open access). https://doi.org/10.1186/1471-2105-7-91
  *Read for:* why tuning by CV makes the CV score optimistic, and nested CV. *Level:* intermediate.
- **Stephen Bates, Trevor Hastie, Robert Tibshirani, "Cross-Validation: What Does It Estimate and How Well Does It Do It?"** (2024, Journal of the American Statistical Association 119(546); arXiv:2104.00673). https://arxiv.org/abs/2104.00673
  *Read for:* what CV estimates and why naive CV intervals undercover. *Level:* advanced.
- **Tom Fawcett, "An introduction to ROC analysis"** (2006, Pattern Recognition Letters 27(8); publisher version may be paywalled). https://doi.org/10.1016/j.patrec.2005.10.010
  *Read for:* a practical primer on ROC curves and AUC. *Level:* intro.
- **Takaya Saito, Marc Rehmsmeier, "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets"** (2015, PLOS ONE 10(3):e0118432; open access). https://doi.org/10.1371/journal.pone.0118432
  *Read for:* why ROC can flatter classifiers on imbalanced data. *Level:* intermediate.
- **Alexandru Niculescu-Mizil, Rich Caruana, "Predicting Good Probabilities with Supervised Learning"** (2005, ICML). https://doi.org/10.1145/1102351.1102430
  *Read for:* how learners distort probabilities, and Platt and isotonic repairs. *Level:* intermediate.
- **Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger, "On Calibration of Modern Neural Networks"** (2017, ICML; arXiv:1706.04599). https://arxiv.org/abs/1706.04599
  *Read for:* overconfidence in deep networks and temperature scaling. *Level:* intermediate.
