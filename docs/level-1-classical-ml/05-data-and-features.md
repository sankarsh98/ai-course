# Data and Features

> **Level 1 · Module 5** · Prerequisites: L1-01, L1-02, L1-03 · Time: 5 to 6 hours · Code: `code/L1-05-data-and-features/`

## What you will be able to do

- Audit a dataset for how it was collected and labeled, and estimate how label errors affect training and evaluation.
- Encode categorical variables and scale numeric ones without leaking the target, and say which models need which treatment.
- Classify a missing-data problem as MCAR, MAR or MNAR, and choose an imputation strategy whose bias you can predict.
- Handle class imbalance by fixing the decision threshold and the metric before reaching for resampling, and explain what resampling does to probabilities.
- Detect covariate shift, label shift and train/serve skew with simple statistical checks, and correct label shift when its assumptions hold.

## 1. The data is most of the model

**Data-centric AI** means treating the dataset, not the model, as the main thing you engineer: what to collect, how to label it, how to clean it, and how to keep it consistent between training and use.

Neglecting this work is costly. Sambasivan and colleagues interviewed 53 practitioners on high-stakes AI projects in India, East and West Africa and the USA. They describe **data cascades**, compounding failures that start with data problems and surface late and far downstream, and report a 92% prevalence among their interviewees [Sambasivan et al., 2021].

A cheap first defense is to write down what the data is. **Datasheets for datasets** [Gebru et al., 2021] give a standard list of questions: why the dataset was created, what is in it, how it was collected, labeled and preprocessed, and what it should and shouldn't be used for.

## 2. Collection and labels

**Collection.** Ask what population the rows represent and how they were chosen. Survey responders, patients at one hospital and scraped images are each a biased sample of something larger. A model learns the sample, and feature engineering can't recover rows you never collected. (Fairness belongs to L4-05.)

**Label definition.** A label is a measurement with its own error. "Churned" might mean cancelled, inactive for 30 days or downgraded. "Fraud" might mean "confirmed by an investigation", which only happens to cases someone chose to investigate. Write the definition down.

**Label noise.** Annotators disagree and make mistakes. Measure agreement by having several people label one sample. Northcutt, Athalye and Mueller estimated an average of at least 3.3% label errors across the test sets of 10 widely used benchmarks, with at least 6% in the ImageNet validation set. They also found that, with corrected test labels, smaller models could outrank larger ones, for example ResNet-18 over ResNet-50 if the share of mislabeled test examples rose by just 6% [Northcutt et al., 2021]. Noisy *test* labels don't just add noise to your score. They can change which model you pick.

To find suspect labels, get **out-of-fold** predicted probabilities (each row scored by a model that didn't train on it) and inspect the rows where the probability of the *given* label is lowest. Northcutt and colleagues flagged candidates with a more refined version of this idea, *confident learning*, and then had crowd workers check them. On average only 51% of flagged candidates were real errors, so suspects need human review.

## 3. Feature engineering

A **feature** is any number the model receives. Feature engineering turns raw records into features that make the pattern easier to learn: transforms (logs of heavy-tailed amounts, ratios such as debt over income), date parts (day of week, days since last purchase), window aggregates (purchases in the last 30 days), and interactions for linear models (tree ensembles find many interactions themselves, L1-03).

The key rule is **point-in-time correctness**: a feature may use only information available when the prediction would be made. An aggregate over the whole table, including later events, is leakage (L1-01), and it produces models that look excellent offline and fail in production.

## 4. Encoding categorical variables

A **categorical** variable takes values from a set with no numeric meaning (city, product ID). Options:

- **One-hot**: one 0/1 column per category. Safe for linear models, and workable with regularization even at thousands of levels. Decide how unseen categories are handled at serving time (`handle_unknown="ignore"` maps them to all zeros).
- **Ordinal codes**: integers 0, 1, 2. Fine for trees, misleading for linear models because they impose an order.
- **Hashing**: hash categories into a fixed number of columns. Bounded memory, with collisions.
- **Target encoding**: replace each category with the mean target among training rows of that category, shrunk toward the global mean for rare categories. scikit-learn's documentation credits the scheme to Micci-Barreca (2001) [scikit-learn preprocessing guide]. It turns a 10,000-level column into one informative number.

The trap: if a row's own label is part of its category mean, the feature contains the label. For a two-row category the encoding is nearly the label itself, so the model learns to trust it and then fails on new data. The fix is **cross-fitting**: encode each fold of the training data with means computed on the other folds. `TargetEncoder.fit_transform` does this, and its documentation warns that `fit(X, y).transform(X)` is not equivalent and leaks.

## 5. Scaling

**Scaling** changes a feature's units. It matters for distance-based models (k-NN, k-means, RBF-kernel SVMs), for penalized linear models (units decide which coefficients the penalty shrinks), and for gradient-based training (conditioning, L0-05). Trees only use orderings and don't care.

- **Standardization**: $x' = (x - \hat{\mu})/\hat{\sigma}$, with mean and standard deviation estimated on training data.
- **Robust scaling**: subtract the median and divide by the interquartile range, so a few extreme values don't set the scale. scikit-learn's guide recommends it when there are many outliers.
- **Min-max** maps to $[0, 1]$ and is sensitive to extremes. **Log or power transforms** change the shape of skewed positive quantities, not only the scale.

Fit scalers on training data only. A `Pipeline` does this automatically inside cross-validation.

## 6. Missing data

Rubin's framework [Rubin, 1976], explained clearly in van Buuren's book [van Buuren, 2018], classifies *why* values are missing. Let $v$ be a variable with missing values, $x$ the observed variables, and $M$ the indicator that $v$ is missing:

- **MCAR** (missing completely at random): $P(M)$ doesn't depend on any data. Example: a sensor's battery dies at random times.
- **MAR** (missing at random): $P(M)$ depends only on observed data, $P(M \mid x, v) = P(M \mid x)$. Example: older respondents skip an income question, and age is recorded.
- **MNAR** (missing not at random): $P(M)$ depends on the missing value itself. Example: high earners skip the income question.

The mechanism decides what works. Dropping incomplete rows (**complete-case analysis**) is unbiased only under MCAR. **Mean imputation** keeps the mean under MCAR but shrinks the variance and weakens correlations, because every filled value sits at the center. **Regression imputation** (predict $v$ from $x$) removes bias under MAR, but deterministic predictions lie exactly on the regression line and exaggerate correlations. Adding noise drawn from the fitted model fixes that. Under MNAR, no method using only observed data is guaranteed to remove the bias. You need outside knowledge or a sensitivity analysis. The observed data alone can't distinguish MAR from MNAR.

For *prediction*, missingness can itself be informative, and a **missing-indicator** column lets the model use it. scikit-learn's `HistGradientBoostingClassifier` accepts missing values natively and learns which branch they follow. For *inference* (a coefficient, a mean), use multiple imputation, which van Buuren covers in depth.

## 7. Class imbalance

**Imbalance** means one class is rare (fraud, disease, failure). Before rebalancing the data, separate three questions:

1. **Can the model rank?** ROC AUC and average precision don't depend on a threshold, and imbalance doesn't stop good ranking.
2. **Are the probabilities right?** With 3% positives, calibrated predictions average about 0.03 (L1-01).
3. **Where is the threshold?** 0.5 is right only if both errors cost the same. With rare positives and costly misses it should be much lower. The threshold, not the data, is usually what needs changing.

**Resampling** (oversampling the minority, undersampling the majority) and **reweighting** (`class_weight="balanced"`) all pretend the classes are equally common. **SMOTE** [Chawla et al., 2002] oversamples by creating synthetic minority points on line segments between a minority point and its minority-class neighbors. All of these shift predicted probabilities toward the minority class. Van den Goorbergh and colleagues compared them for logistic-regression risk models and found that the corrections didn't improve ROC AUC, and that they produced badly miscalibrated models that strongly overestimated risk [van den Goorbergh et al., 2022]. Shifting the threshold gave similar classification results without that cost.

That evidence is for logistic regression, and corrections may still help some flexible learners with very few minority examples. A defensible default: train on the real distribution, check calibration, choose the threshold from the costs, and if you do reweight, convert the probabilities back (the hands-on shows how).

## 8. Dataset shift

**Dataset shift** means the deployment distribution differs from the training distribution. Moreno-Torres and colleagues review the inconsistent terminology in this area and propose a unified one [Moreno-Torres et al., 2012]. Write the joint distribution as $p(x, y) = p(y \mid x)\,p(x)$. The three cases worth knowing:

- **Covariate shift**: $p(x)$ changes, $p(y \mid x)$ doesn't. Harmless for a model that is correct everywhere, harmful for one that was only accurate where it had data.
- **Label (prior) shift**: $p(y)$ changes, $p(x \mid y)$ doesn't. The disease is more common this season, but it looks the same.
- **Concept shift**: $p(y \mid x)$ changes. The same behavior now means something different, for example after a policy change. Only new labels can reveal it.

**Detecting shift.** Compare feature statistics (means, quantiles, missing rates) between training and recent data. For a multivariate check, train a **domain classifier** to tell training rows from new rows. A cross-validated ROC AUC near 0.5 means nothing detectable moved, and its important features show what did.

**Correcting label shift.** If only $p(y)$ changed, Bayes' rule gives $p_t(y \mid x) \propto p_s(y \mid x)\, w_y$ with $w_y = p_t(y)/p_s(y)$, where $s$ is the source (training) and $t$ the target (deployment) distribution. You can't count target labels you don't have, but Lipton, Wang and Smola show how to estimate $w$ from a classifier's predictions. Let $C_{ij}$ be the probability, on labeled source data, of predicting $i$ when the truth is $j$, and let $\mu_i$ be the fraction of target data predicted as $i$. Then $\mu = C w$, so solve $w = C^{-1}\mu$. They call this **black box shift estimation** (BBSE) and give conditions for it to be consistent, mainly an invertible confusion matrix [Lipton et al., 2018].

Monitoring shift in production is part of MLOps (L4-01).

## 9. Train/serve skew

**Training-serving skew** is, in Zinkevich's definition, a difference between performance during training and performance during serving. He lists three causes: different data handling in the training and serving pipelines, data that changed between training and serving, and feedback loops between model and system [Zinkevich, Rules of ML]. The first is the most avoidable. Two teams implement "tenure" differently, a default differs, or a join reads a table that has since changed. The model gets different numbers from the ones it learned on, and nothing raises an error.

Remedies from the same guide: reuse code between training and serving, log the features actually used at serving time and train on those logs, and measure skew. Sculley and colleagues frame the wider problem as technical debt: ML systems have data dependencies and feedback loops that ordinary code review doesn't see, and the authors note it is common to incur "massive ongoing maintenance costs" [Sculley et al., 2015].

## Hands-on

Run each script from the repository root with `.venv/bin/python code/L1-05-data-and-features/<script>.py`. Each takes a few seconds. The output below came from scikit-learn 1.9.1 and NumPy 2.5.4. Last digits may differ on other platforms or versions.

### Script 1: missing-data mechanisms

```python
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
```

```text
truth: mean 0.01  std 2.24  corr(x,v) 0.89

mechanism  method              %missing   mean    std   corr(x,v)
MCAR       complete cases         40%    0.01   2.24      0.89
MCAR       mean impute            40%    0.01   1.73      0.69
MCAR       regression impute      40%    0.02   2.15      0.93
MCAR       regression + noise     40%    0.02   2.24      0.89

MAR        complete cases         43%   -1.04   1.90      0.85
MAR        mean impute            43%   -1.04   1.44      0.52
MAR        regression impute      43%    0.01   2.14      0.93
MAR        regression + noise     43%    0.01   2.24      0.89

MNAR       complete cases         43%   -1.21   1.78      0.84
MNAR       mean impute            43%   -1.21   1.34      0.53
MNAR       regression impute      43%   -0.34   1.93      0.93
MNAR       regression + noise     43%   -0.34   2.02      0.88

Predicting a target that depends on v (5-fold CV R^2):
  MCAR  mean impute              R^2 = 0.801
  MCAR  mean impute + indicator  R^2 = 0.801
  MNAR  mean impute              R^2 = 0.774
  MNAR  mean impute + indicator  R^2 = 0.829
```

Under MCAR, dropping rows is unbiased. Mean imputation keeps the mean but shrinks the standard deviation from 2.24 to 1.73 and the correlation from 0.89 to 0.69. Deterministic regression imputation overshoots the correlation (0.93), and adding noise restores both. Under MAR, the complete-case mean is badly biased (-1.04 against a truth of 0.01), and regression imputation fixes it because $x$ explains the missingness. Under MNAR, every method stays biased (-0.34 at best). For prediction, the missing indicator does nothing under MCAR and raises $R^2$ from 0.774 to 0.829 under MNAR, because *being missing* tells the model that $v$ was probably high.

### Script 2: scaling and target encoding

```python
"""Scaling for distance-based models, and target encoding done wrong and right.

Part 1: k-NN on the wine data with and without standardization.
Part 2: two high-cardinality columns, a useful "city" and a pure-noise
        "user_id", encoded three ways.
"""
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.datasets import load_wine
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

# Part 1 -------------------------------------------------------------------
Xw, yw = load_wine(return_X_y=True)
for name, model in [("k-NN, raw features   ", KNeighborsClassifier()),
                    ("k-NN, standardized   ", make_pipeline(StandardScaler(), KNeighborsClassifier()))]:
    acc = cross_val_score(model, Xw, yw, cv=KFold(5, shuffle=True, random_state=0))
    print(f"{name} 5-fold accuracy {acc.mean():.3f}")

# Part 2 -------------------------------------------------------------------
rng = np.random.default_rng(0)
n = 20000
city = rng.integers(0, 500, n)            # 500 cities, about 40 rows each
city_effect = rng.normal(0, 1.0, 500)     # each city has a real effect
user_id = rng.integers(0, 10000, n)       # about 2 rows per id, no effect at all
x_num = rng.normal(size=n)
logit = 0.8 * x_num + city_effect[city]
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
X = np.column_stack([x_num, city, user_id])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)


def naive_target_encode(col_tr, col_te):
    # Mean target per category computed on ALL training rows, including the
    # row being encoded. That row's own label leaks into its feature.
    prior = ytr.mean()
    means = {c: ytr[col_tr == c].mean() for c in np.unique(col_tr)}
    return (np.array([means[c] for c in col_tr]),
            np.array([means.get(c, prior) for c in col_te]))


def report(name, Ftr, Fte, model):
    model.fit(Ftr, ytr)
    tr = roc_auc_score(ytr, model.predict_proba(Ftr)[:, 1])
    te = roc_auc_score(yte, model.predict_proba(Fte)[:, 1])
    print(f"  {name:34s} train AUC {tr:.3f}   test AUC {te:.3f}")


def gbm():
    return HistGradientBoostingClassifier(random_state=0)


# Best possible test AUC: score test rows with the true logit we simulated.
oracle = 0.8 * Xte[:, 0] + city_effect[Xte[:, 1].astype(int)]
print(f"\nHigh-cardinality categoricals (oracle test AUC {roc_auc_score(yte, oracle):.3f})")

enc = [naive_target_encode(Xtr[:, j], Xte[:, j]) for j in (1, 2)]
report("naive target encoding + GBM",
       np.column_stack([Xtr[:, 0], enc[0][0], enc[1][0]]),
       np.column_stack([Xte[:, 0], enc[0][1], enc[1][1]]), gbm())

te_enc = TargetEncoder(cv=KFold(5, shuffle=True, random_state=0))
cats_tr = te_enc.fit_transform(Xtr[:, 1:].astype(int), ytr)   # cross-fitted
cats_te = te_enc.transform(Xte[:, 1:].astype(int))
report("cross-fitted TargetEncoder + GBM",
       np.column_stack([Xtr[:, 0], cats_tr]), np.column_stack([Xte[:, 0], cats_te]), gbm())

# One-hot the two categorical columns (sparse), keep the numeric column as is.
oh = OneHotEncoder(handle_unknown="ignore").fit(Xtr[:, 1:].astype(int))
Ftr = hstack([csr_matrix(Xtr[:, :1]), oh.transform(Xtr[:, 1:].astype(int))]).tocsr()
Fte = hstack([csr_matrix(Xte[:, :1]), oh.transform(Xte[:, 1:].astype(int))]).tocsr()
report("one-hot + L2 logistic regression", Ftr, Fte, LogisticRegression(C=0.5, max_iter=2000))
```

```text
k-NN, raw features    5-fold accuracy 0.714
k-NN, standardized    5-fold accuracy 0.967

High-cardinality categoricals (oracle test AUC 0.781)
  naive target encoding + GBM        train AUC 0.968   test AUC 0.555
  cross-fitted TargetEncoder + GBM   train AUC 0.787   test AUC 0.754
  one-hot + L2 logistic regression   train AUC 0.873   test AUC 0.759
```

Standardizing lifts k-NN on wine from 0.714 to 0.967, because proline dominates raw distances (as in L1-04). Naive target encoding scores 0.968 on train and 0.555 on test. With about two rows per `user_id`, the noise column's "encoding" is nearly each row's own label. Cross-fitting closes the gap (0.787 train, 0.754 test), and one-hot with an L2 penalty does about as well. Both come close to the oracle's 0.781.

### Script 3: class imbalance

```python
"""Class imbalance: resampling and reweighting vs simply moving the threshold.

Logistic regression on data with about 3% positives, trained four ways, scored
on ranking (ROC AUC, average precision), calibration (Brier score, mean
predicted probability) and precision/recall at a threshold.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

rng = np.random.default_rng(0)
X, y = make_classification(n_samples=30000, n_features=10, n_informative=5,
                           weights=[0.97], flip_y=0.01, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)


def oversample(X, y):
    # Duplicate random minority rows until the classes are equal in size.
    pos = np.flatnonzero(y == 1)
    extra = rng.choice(pos, (y == 0).sum() - len(pos))
    return np.vstack([X, X[extra]]), np.r_[y, y[extra]]


def smote(X, y, k=5):
    # SMOTE [Chawla et al., 2002]: new minority points on line segments
    # between a minority point and one of its k nearest minority neighbors.
    P = X[y == 1]
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(P).kneighbors(P)[1][:, 1:]
    m = (y == 0).sum() - len(P)
    i = rng.integers(len(P), size=m)
    j = nbrs[i, rng.integers(k, size=m)]
    new = P[i] + rng.random((m, 1)) * (P[j] - P[i])
    return np.vstack([X, new]), np.r_[y, np.ones(m, int)]


variants = {
    "no correction": (Xtr, ytr, None),
    "class_weight='balanced'": (Xtr, ytr, "balanced"),
    "random oversampling": (*oversample(Xtr, ytr), None),
    "SMOTE": (*smote(Xtr, ytr), None),
}
print(f"positive rate: train {ytr.mean():.3f}, test {yte.mean():.3f}\n")
print("variant                   ROC AUC  avg prec  Brier   mean p   prec@0.5 rec@0.5")
probs = {}
for name, (Xf, yf, cw) in variants.items():
    p = LogisticRegression(class_weight=cw, max_iter=1000).fit(Xf, yf).predict_proba(Xte)[:, 1]
    probs[name] = p
    pred = p >= 0.5
    print(f"{name:25s} {roc_auc_score(yte, p):7.3f} {average_precision_score(yte, p):8.3f} "
          f"{brier_score_loss(yte, p):7.4f} {p.mean():7.3f} "
          f"{precision_score(yte, pred, zero_division=0):8.3f} {recall_score(yte, pred):7.3f}")

# Threshold moving: keep the uncorrected model, lower its threshold until it
# flags as many test cases as the balanced model did at 0.5. (In practice,
# choose the threshold on a validation set from the costs of each error.)
n_flag = (probs["class_weight='balanced'"] >= 0.5).sum()
t = np.sort(probs["no correction"])[::-1][n_flag - 1]
pred = probs["no correction"] >= t
print(f"\nno correction, threshold moved to {t:.3f} (flags {n_flag} cases): "
      f"precision {precision_score(yte, pred):.3f}, recall {recall_score(yte, pred):.3f}")

# Reweighting to 50/50 multiplies the model's odds by about (1 - pi) / pi,
# where pi is the real positive rate. Undo that to get usable probabilities.
pi = ytr.mean()
pw = probs["class_weight='balanced'"]
odds = pw / (1 - pw) * pi / (1 - pi)
p_fix = odds / (1 + odds)
print(f"balanced model after odds correction: Brier {brier_score_loss(yte, p_fix):.4f}, "
      f"mean p {p_fix.mean():.3f}, ROC AUC {roc_auc_score(yte, p_fix):.3f} (ranking unchanged)")
```

```text
positive rate: train 0.035, test 0.035

variant                   ROC AUC  avg prec  Brier   mean p   prec@0.5 rec@0.5
no correction               0.886    0.501  0.0231   0.034    0.781   0.291
class_weight='balanced'     0.894    0.404  0.1188   0.274    0.158   0.863
random oversampling         0.894    0.402  0.1198   0.277    0.157   0.863
SMOTE                       0.894    0.414  0.1155   0.266    0.161   0.857

no correction, threshold moved to 0.034 (flags 2881 cases): precision 0.147, recall 0.806
balanced model after odds correction: Brier 0.0263, mean p 0.029, ROC AUC 0.894 (ranking unchanged)
```

All three corrections give a mean predicted probability near 0.27 when the real rate is 0.035, and about five times the Brier score. Moving the uncorrected model's threshold to 0.034 gives a similar precision/recall trade (0.147 and 0.806, against 0.158 and 0.863) and keeps calibrated probabilities. Two details are messier than the textbook story. I checked both over six data seeds. First, reweighting slightly *raised* ROC AUC (by 0 to 0.017) while lowering average precision. The linear model is misspecified here, so reweighting changes which region the fit prioritizes, not only its intercept. Second, the odds correction brings probabilities back near the real rate (mean 0.029) but leaves the Brier score (0.0263) a little worse than the uncorrected model's 0.0231.

### Script 4: dataset shift

```python
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
```

```text
Part 1: covariate shift, linear model fit on x ~ N(0, 0.8)
  MSE on held-out training-like data: 0.033
  MSE on shifted data x ~ N(2.5, 0.8): 3.245

Part 2: domain-classifier AUC (0.5 = indistinguishable)
  same distribution          : 0.489
  one feature shifted 0.3 std: 0.559
  Part 1 inputs              : 0.989

Part 3: label shift, source positive rate 0.5, target 0.1
  mean predicted probability on target, unadjusted: 0.326
  BBSE estimate of target positive rate           : 0.116 (true 0.106)
  Brier score on target, unadjusted: 0.1373
  Brier score on target, adjusted  : 0.0637
```

The linear model's error rises about 100-fold when inputs move to where it was never fit, even though $p(y \mid x)$ didn't change. The domain classifier scores 0.49 on identical distributions, picks up a 0.3-standard-deviation move in one of five features (0.56), and separates the Part 1 inputs almost perfectly (0.99). Under label shift, the unadjusted model predicts 33% positives when the true rate is about 11%. BBSE estimates 11.6% without seeing any target labels, and reweighting the probabilities halves the Brier score.

### Script 5: train/serve skew and label errors

```python
"""Train/serve skew and label errors, two data problems a model can't see.

Part 1: serving computes features differently; a statistics check catches it.
Part 2: 10% of training labels are wrong; out-of-fold predictions flag suspects.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)

# Part 1 -------------------------------------------------------------------
n = 20000
tenure_days = rng.exponential(400, n)
sessions = rng.poisson(5, n).astype(float)
sessions[rng.random(n) < 0.2] = np.nan     # some users have no session log
logit = -1 + 1.5 * (tenure_days < 90) - 0.3 * np.nan_to_num(sessions, nan=2.0)
churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
X = np.column_stack([tenure_days, sessions])
Xtr, Xsv, ytr, ysv = train_test_split(X, churn, test_size=0.3, random_state=0)
model = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)

# The serving team re-implemented the features: tenure in weeks, and missing
# session counts filled with 0 instead of being left missing.
Xbug = Xsv.copy()
Xbug[:, 0] = Xbug[:, 0] / 7
Xbug[:, 1] = np.nan_to_num(Xbug[:, 1], nan=0.0)
print("Part 1: train/serve skew")
print(f"  AUC, serving features computed like training: {roc_auc_score(ysv, model.predict_proba(Xsv)[:, 1]):.3f}")
print(f"  AUC, serving features from buggy pipeline   : {roc_auc_score(ysv, model.predict_proba(Xbug)[:, 1]):.3f}")


def skew_report(train, serve, names):
    # Standardized mean difference and missing-rate difference, per feature.
    for j, name in enumerate(names):
        a, b = train[:, j], serve[:, j]
        smd = (np.nanmean(b) - np.nanmean(a)) / np.nanstd(a)
        dmiss = np.isnan(b).mean() - np.isnan(a).mean()
        flag = "  <-- check" if abs(smd) > 0.25 or abs(dmiss) > 0.05 else ""
        print(f"    {name:12s} std. mean diff {smd:+.2f}  missing-rate diff {dmiss:+.2f}{flag}")


print("  skew check, buggy serving data vs training data:")
skew_report(Xtr, Xbug, ["tenure_days", "sessions"])

# Part 2 -------------------------------------------------------------------
Xd, yd = load_digits(return_X_y=True)
Xa, Xb, ya, yb = train_test_split(Xd, yd, test_size=0.5, stratify=yd, random_state=0)
flip = rng.random(len(ya)) < 0.10
ya_noisy = ya.copy()
ya_noisy[flip] = (ya[flip] + rng.integers(1, 10, flip.sum())) % 10   # wrong label


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


print(f"\nPart 2: label errors ({flip.sum()} of {len(ya)} training labels flipped)")
acc_noisy = (clf().fit(Xa, ya_noisy).predict(Xb) == yb).mean()
print(f"  test accuracy, trained on noisy labels: {acc_noisy:.3f}")

# Out-of-fold probability the model assigns to each row's *given* label.
# Low values mark suspects (a simple form of the idea behind confident learning).
P = cross_val_predict(clf(), Xa, ya_noisy, cv=5, method="predict_proba")
self_conf = P[np.arange(len(ya)), ya_noisy]
suspects = np.argsort(self_conf)[:flip.sum()]
print(f"  of the {len(suspects)} lowest-confidence rows, {flip[suspects].mean():.0%} were truly flipped")
keep = np.setdiff1d(np.arange(len(ya)), suspects)
acc_clean = (clf().fit(Xa[keep], ya_noisy[keep]).predict(Xb) == yb).mean()
print(f"  test accuracy after dropping suspects : {acc_clean:.3f}")
print(f"  test accuracy, trained on true labels : {(clf().fit(Xa, ya).predict(Xb) == yb).mean():.3f}")

# Now corrupt 10% of the *test* labels and compare two models.
yb_noisy = yb.copy()
tf = rng.random(len(yb)) < 0.10
yb_noisy[tf] = (yb[tf] + rng.integers(1, 10, tf.sum())) % 10
print("  measured on clean vs noisy test labels:")
for name, m in [("trained on noisy labels ", clf().fit(Xa, ya_noisy)),
                ("after dropping suspects ", clf().fit(Xa[keep], ya_noisy[keep]))]:
    p = m.predict(Xb)
    print(f"    {name} clean {np.mean(p == yb):.3f}   noisy {np.mean(p == yb_noisy):.3f}")
```

```text
Part 1: train/serve skew
  AUC, serving features computed like training: 0.720
  AUC, serving features from buggy pipeline   : 0.636
  skew check, buggy serving data vs training data:
    tenure_days  std. mean diff -0.85  missing-rate diff +0.00  <-- check
    sessions     std. mean diff -0.47  missing-rate diff -0.20  <-- check

Part 2: label errors (87 of 898 training labels flipped)
  test accuracy, trained on noisy labels: 0.891
  of the 87 lowest-confidence rows, 80% were truly flipped
  test accuracy after dropping suspects : 0.929
  test accuracy, trained on true labels : 0.963
  measured on clean vs noisy test labels:
    trained on noisy labels  clean 0.891   noisy 0.798
    after dropping suspects  clean 0.929   noisy 0.829
```

The unit bug and the changed missing-value default cut AUC from 0.720 to 0.636 without any error. The statistics check flags both: tenure's mean moves 0.85 standard deviations, and the missing rate of `sessions` drops 20 points because missing values became zeros. Flipping 10% of labels costs 7 points of test accuracy. The 87 lowest-confidence rows are 80% truly flipped, and dropping them recovers about half the loss. Noisy test labels make both models look about 9 points worse and shrink the gap between them from 3.8 to 3.1 points. With closer models, the ranking could flip, as Northcutt and colleagues found.

## Common mistakes

- **Fitting preprocessing on all the data.** Scalers, imputers, encoders and feature selectors are learned. Fit them inside the cross-validation `Pipeline`.
- **Naive target encoding.** Train scores far above validation scores on a high-cardinality column are the signature. Use cross-fitting.
- **Mean-imputing and forgetting.** You shrink variances and hide informative missingness. Add an indicator, or use a model that handles missing values.
- **Rebalancing by reflex.** Check calibration and choose a threshold first. If you resample or reweight, don't report the resulting probabilities as risks.
- **Ignoring the label definition.** Write it down and check it against a hand-labeled sample.
- **Separate training and serving code paths.** Log serving features, compare their statistics with training, and alert on differences.

## Exercises

1. [warm-up] For each case, say MCAR, MAR or MNAR: (a) lab results missing because a sample tube broke; (b) depression scores missing more often for severely depressed patients; (c) income missing more often for younger respondents, with age recorded.
2. [warm-up] In Script 2, replace `StandardScaler` with `RobustScaler` and `MinMaxScaler`. Does k-NN accuracy change much? Why?
3. [warm-up] A model's predicted probabilities on test data average 0.27, and the true positive rate is 0.035. Name two possible causes.
4. [core] In Script 1, make the MNAR mechanism stronger (multiply $v$ by 3 inside the logistic). How do the biases change? Can any method in the script detect that the data is MNAR?
5. [core] In Script 2, set the noise column to 200 levels instead of 10,000. How does the naive encoding's train/test gap change? Explain with the number of rows per level.
6. [core] In Script 3, choose the threshold that minimizes expected cost when a false negative costs 20 and a false positive costs 1. Do it on a validation split, not the test set. Compare with $1/(1+20)$, the threshold decision theory gives for calibrated probabilities.
7. [core] In Script 4 Part 3, change the target rate to 0.02. How accurate is BBSE now? Then use only 500 target rows. What limits the estimate?
8. [core] Make the domain classifier in Script 4 report which feature moved, using its coefficients or permutation importance.
9. [stretch] Refine Script 5's ranking with per-class thresholds: flag a row if its self-confidence is below the mean self-confidence of rows with the same given label. Compare its precision and recall at finding flipped labels with the simple ranking in Script 5.
10. [stretch] Write a datasheet (motivation, composition, collection, preprocessing, uses) for a dataset you use at work or for one of scikit-learn's bundled datasets. Which questions couldn't you answer?

### Hints and answers

1. (a) MCAR. (b) MNAR. (c) MAR.
2. All three put features on comparable scales, so accuracy is similar (0.950 to 0.967 when I ran it). Robust scaling matters more when outliers would otherwise set the scale.
3. The model was trained on rebalanced or reweighted data, or the training rate was higher than the deployment rate (label shift). Compare the training positive rate with the deployment rate.
4. Biases grow for every method. The observed data can't prove MNAR, though disagreement with an outside benchmark (a census mean) can suggest it.
5. With about 70 training rows per level, no single row moves its category mean much, so the leak and the gap shrink. The leak is worst for rare categories.
6. Expected cost $= 20\,\text{FN} + \text{FP}$. For a calibrated model the optimal threshold is $c_{FP}/(c_{FP}+c_{FN}) = 1/21 \approx 0.048$. The validated threshold should come out close if the model is well calibrated.
7. The estimate gets noisier with low prevalence and small samples: $\mu$ comes from few predicted positives, and inverting $C$ amplifies the error. Lipton and colleagues give error bounds.
8. The coefficient (or permutation importance) for feature 3 should stand out.
9. Per-class thresholds matter most when some classes are much harder than others. Report precision and recall for both methods and say which classes differ.
10. There is no single answer. The questions you couldn't answer are the risks to investigate.

## Check your understanding

1. **Why does naive target encoding overfit even when the category truly matters?** Each row's own label is inside its encoding, so for rare categories the feature nearly equals the label and is far less informative on new data. Cross-fitting encodes each row with means that exclude it.

2. **Under MAR, why does regression imputation remove the bias that complete-case analysis has?** Under MAR, whether $v$ is missing depends only on observed $x$. So among rows with the same $x$, observed and missing $v$ have the same distribution, and a model of $v$ given $x$ fitted on observed rows is valid for the missing ones.

3. **A colleague oversamples the minority class to 50% and reports "27% risk" for a patient. What's wrong?** The model learned a 50% base rate, so its probabilities are inflated. Use a model trained on the real distribution, or correct the odds by the true-to-training base-rate ratio, and check calibration.

4. **How do you tell covariate shift from concept shift in production?** Covariate shift appears in the inputs, so feature statistics and a domain classifier detect it without labels. Concept shift changes $p(y \mid x)$ and may leave inputs unchanged. Only new labels reveal it.

5. **What is the best defense against train/serve skew?** One code path for features, ideally logging serving-time features and training on those logs. Statistics checks like Script 5's catch what is left.

## Further reading

- **Nithya Sambasivan, Shivani Kapania, Hannah Highfill, Diana Akrong, Praveen Kumar Paritosh and Lora Mois Aroyo, "'Everyone wants to do the model work, not the data work': Data Cascades in High-Stakes AI"** (2021, ACM CHI). <https://research.google/pubs/pub49953/>
  *Read for:* interview evidence on how data problems compound in real projects. *Level:* intro.
- **Curtis G. Northcutt, Anish Athalye and Jonas Mueller, "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks"** (2021, NeurIPS Datasets and Benchmarks Track, arXiv:2103.14749). <https://arxiv.org/abs/2103.14749>
  *Read for:* measured label error rates in 10 benchmarks and how they change model rankings. *Level:* intermediate.
- **Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé III and Kate Crawford, "Datasheets for Datasets"** (2021, Communications of the ACM; arXiv:1803.09010). <https://arxiv.org/abs/1803.09010>
  *Read for:* the full question list for documenting a dataset. *Level:* intro.
- **Donald B. Rubin, "Inference and Missing Data"** (1976, Biometrika 63(3), 581-592). <https://academic.oup.com/biomet/article/63/3/581/270932>
  *Read for:* the original conditions under which missingness can be ignored (paywalled). *Level:* advanced.
- **Stef van Buuren, "Flexible Imputation of Missing Data"** (2nd edition, 2018, Chapman & Hall/CRC; free online). <https://stefvanbuuren.name/fimd/>
  *Read for:* Section 1.2 on MCAR, MAR and MNAR, then multiple imputation in practice. *Level:* intermediate.
- **N. V. Chawla, K. W. Bowyer, L. O. Hall and W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique"** (2002, Journal of Artificial Intelligence Research 16). <https://www.jair.org/index.php/jair/article/view/10302>
  *Read for:* the SMOTE algorithm and the original experiments (open access). *Level:* intermediate.
- **Ruben van den Goorbergh, Maarten van Smeden, Dirk Timmerman and Ben Van Calster, "The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression"** (2022, Journal of the American Medical Informatics Association 29(9), 1525-1534). <https://pmc.ncbi.nlm.nih.gov/articles/PMC9382395/>
  *Read for:* evidence that imbalance corrections damage calibration without improving discrimination (open access). *Level:* intermediate.
- **José G. Moreno-Torres, Troy Raeder, Rocío Alaiz-Rodríguez, Nitesh V. Chawla and Francisco Herrera, "A unifying view on dataset shift in classification"** (2012, Pattern Recognition 45(1), 521-530). <https://doi.org/10.1016/j.patcog.2011.06.019>
  *Read for:* a consistent vocabulary for the kinds of dataset shift (publisher version paywalled). *Level:* intermediate.
- **Zachary C. Lipton, Yu-Xiang Wang and Alex Smola, "Detecting and Correcting for Label Shift with Black Box Predictors"** (2018, ICML; arXiv:1802.03916). <https://arxiv.org/abs/1802.03916>
  *Read for:* BBSE, its assumptions and error bounds, and a test for label shift. *Level:* advanced.
- **D. Sculley et al., "Hidden Technical Debt in Machine Learning Systems"** (2015, Advances in Neural Information Processing Systems 28). <https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html>
  *Read for:* why ML systems are costly to maintain, with data dependencies and feedback loops as central causes. *Level:* intro.
- **Martin Zinkevich, "Rules of Machine Learning: Best Practices for ML Engineering"** (Google for Developers). <https://developers.google.com/machine-learning/guides/rules-of-ml>
  *Read for:* practical rules on features, logging and training-serving skew (Rules 29 to 37). *Level:* intro.
- **scikit-learn developers, "Preprocessing data", User Guide (version 1.9.1)**. <https://scikit-learn.org/stable/modules/preprocessing.html>
  *Read for:* scalers, encoders, `TargetEncoder` and its cross-fitting, with code. *Level:* intro.
