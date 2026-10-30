# Trees and Ensembles

> **Level 1 · Module 3** · Prerequisites: L1-01, L1-02 · Time: 6 to 8 hours · Code: `code/L1-03-trees-and-ensembles/`

## What you will be able to do

- Build a classification tree from scratch, compute Gini and entropy splits by hand, and explain why deep trees overfit.
- Derive why averaging trees reduces variance, and use it to explain bagging, random forests, and `max_features`.
- Implement gradient boosting from scratch for squared loss and for log-loss with Newton leaf values, as in XGBoost.
- Choose and tune a tree ensemble for a tabular problem with a sensible default recipe.
- Summarize the evidence on trees versus deep learning and tabular foundation models as of October 2026, including where it is contested.
- Compute impurity and permutation importances and spot their failure modes.

## 1. Decision trees

A decision tree predicts by asking a sequence of yes/no questions about single features ("is `MedInc` ≤ 5? If yes, is `AveOccup` ≤ 2.4?"). Each leaf predicts a constant: the majority class (or class proportions) for classification, the mean target for regression. The result is a piecewise-constant function on axis-aligned boxes.

Finding the best tree is computationally intractable in general, so CART-style algorithms grow trees **greedily**: at each node, try every feature and threshold, take the split that most reduces impurity, and recurse. Stop when a node is pure, too small, or at a maximum depth. Consequences:

- **Only the order of each feature's values matters.** Any monotone transformation (log, rank, scaling) gives the same tree. No standardization needed.
- **Interactions come for free.** A split on `Latitude` below a split on `MedInc` is an interaction.
- **Axis-aligned boundaries.** A diagonal boundary like $x_1 + x_2 > 0$ needs a staircase of many splits.
- **High variance.** A small change in the data can change the top split and everything beneath it. A full-depth tree fits training data perfectly and generalizes poorly. Most of this module is about fixing that.

## 2. Splitting criteria

Let a node contain $n$ examples with class proportions $p_1, \dots, p_K$. Two common impurity measures:

$$\text{Gini} = 1 - \sum_{k=1}^K p_k^2, \qquad \text{Entropy} = -\sum_{k=1}^K p_k \log_2 p_k.$$

Both are zero for a pure node and largest when classes are evenly mixed. A split sending $n_L$ examples left and $n_R$ right is scored by the weighted child impurity $\frac{n_L}{n} I_L + \frac{n_R}{n} I_R$; the reduction from the parent's impurity is the **gain** (with entropy, the "information gain"). For regression the impurity is the variance (mean squared error around the node mean), and leaves predict the mean.

**A worked example.** Ten points with $x = 1, \dots, 10$ and labels $0,0,0,0,1,0,1,1,1,1$. The parent has $p = (0.5, 0.5)$ and Gini $= 1 - 0.25 - 0.25 = 0.5$.

- Cut at $x \le 4.5$: left is four 0s (Gini 0). Right is $\{1,0,1,1,1,1\}$ with $p = (1/6, 5/6)$, Gini $= 1 - 1/36 - 25/36 \approx 0.278$. Weighted: $0.4 \cdot 0 + 0.6 \cdot 0.278 = 0.167$.
- Cut at $x \le 5.5$: both sides have one minority point among five, Gini $= 1 - 0.64 - 0.04 = 0.32$ each. Weighted: $0.32$.
- Cut at $x \le 6.5$: the mirror image of the first, also $0.167$. **A tie.**

Ties are common, and how an implementation breaks them changes the tree (script 1). The misclassification rate is a poor split criterion because it is flat over many splits that clearly improve purity; Gini and entropy are strictly concave and reward purer children, and usually give similar trees.

Split search costs $O(n \log n)$ per feature to sort, then a linear scan with running class counts (cumulative sums in script 1's `best_split`). Histogram-based implementations first bin each feature into at most a few hundred values, which makes the scan much cheaper on large data.

To control overfitting, **stop early** (`max_depth`, `min_samples_leaf`) or grow a full tree and **prune** it (scikit-learn's `ccp_alpha` does cost-complexity pruning). Either way one tree stays a high-variance model. Combining many trees works far better.

## 3. Bagging: averaging away variance

Suppose you average $B$ predictors, each with variance $\sigma^2$ at a given input, with pairwise correlation $\rho$. Then

$$\operatorname{Var}\Big(\frac{1}{B}\sum_{b=1}^B Z_b\Big) = \frac{1}{B^2}\big(B\sigma^2 + B(B-1)\rho\sigma^2\big) = \rho\sigma^2 + \frac{1 - \rho}{B}\sigma^2.$$

More members kill the second term, but the first does not shrink with $B$: **correlation between members sets the floor**.

**Bagging** (bootstrap aggregating) [Breiman, 1996] makes the members different by training each on a bootstrap sample: $n$ rows drawn with replacement. Breiman's key observation is that bagging helps when the base method is *unstable*: small perturbations of the training set change the fitted predictor a lot. Deep trees are the textbook unstable learner; bagging a stable method such as linear regression does little.

Each bootstrap sample leaves out about $(1 - 1/n)^n \approx e^{-1} \approx 37\%$ of rows. Predicting each row with only the trees that did not see it gives the **out-of-bag (OOB)** error, a validation estimate for free.

## 4. Random forests

Bagged trees are still strongly correlated: they all split on the strongest feature first. **Random forests** [Breiman, 2001] consider only a random subset of `max_features` features at each split. Breiman showed that forest error depends on the **strength** of individual trees and the **correlation** between them. Restricting features weakens each tree but lowers correlation, which the formula above says can be a net win.

Script 2 measures this on the California housing data. With all 8 features per split (plain bagging), single trees have test MSE 0.583 and pairwise error correlation 0.462; the forest gets 0.273. With 3 features per split, single trees are barely worse (0.591) but correlation drops to 0.412, and the forest improves to 0.247. With 1 feature, the trees get too weak (0.778) and the forest worsens again. The formula $\rho s + (1-\rho)s/B$, with $s$ the single-tree MSE, matches the forest's MSE to within 0.001 in every row (the forest's error is the average of the trees' errors, so it applies).

scikit-learn's defaults are `max_features="sqrt"` for classification and `1.0` (all features: plain bagging) for regression, which its user guide calls empirically good defaults. Here a smaller value is better, so tune it. Otherwise forests are low-maintenance: more trees only cost time, full-depth trees are usually fine, and OOB error is a quick check.

## 5. Boosting

Bagging fights variance by averaging. **Boosting** fights bias by adding models sequentially, each correcting the current ensemble's mistakes. The first widely used version was AdaBoost (Freund and Schapire, Journal of Computer and System Sciences, 1997), which reweights misclassified examples. The view that took over is **gradient boosting** [Friedman, 2001]: gradient descent in function space.

Start with a constant $F_0$. At round $m$, compute for each example the negative gradient of the loss with respect to the current prediction,

$$r_{i} = -\frac{\partial \ell(y_i, F)}{\partial F}\Big|_{F = F_{m-1}(\mathbf{x}_i)},$$

fit a small regression tree $h_m$ to the $r_i$, and update $F_m = F_{m-1} + \nu h_m$, where $\nu \in (0, 1]$ is the **learning rate** (shrinkage).

- **Squared loss** $\ell = \frac{1}{2}(y - F)^2$: the negative gradient is $y - F$, the **residual**. Each tree fits what the ensemble still gets wrong.
- **Log-loss** for binary labels, with $F$ the log-odds and $p = \sigma(F)$: the negative gradient is $y - p$, exactly the residual from L1-02's logistic regression.

Friedman gives algorithms for several losses (least squares, least absolute deviation, Huber, logistic) and refits each leaf's value to minimize the actual loss in that region.

**The second-order (Newton) view.** XGBoost [Chen and Guestrin, 2016] makes this precise. Take a second-order Taylor expansion of the loss around $F_{m-1}$ with $g_i = \partial \ell/\partial F$ and $h_i = \partial^2 \ell/\partial F^2$, and add a penalty $\Omega(f) = \gamma T + \frac{1}{2}\lambda\lVert\mathbf{w}\rVert^2$ on the new tree, where $T$ is the number of leaves and $\mathbf{w}$ the leaf values. For a fixed tree structure, with $G_j = \sum_{i \in \text{leaf } j} g_i$ and $H_j$ likewise, the optimal leaf value and the gain of a split are

$$w_j^* = -\frac{G_j}{H_j + \lambda}, \qquad \text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma.$$

For log-loss, $g_i = p_i - y_i$ and $h_i = p_i(1 - p_i)$. A leaf of examples already predicted near $p = 0$ or $1$ has tiny $H_j$, so without $\lambda$ the Newton step can be huge. In script 3, $\lambda = 0$ closely matches scikit-learn's `GradientBoostingClassifier`, and $\lambda = 1$ improves test log-loss from 0.237 to 0.166. The paper also uses shrinkage and column subsampling (borrowed from random forests).

Two other widely used libraries: LightGBM (Ke et al., NeurIPS 2017) speeds up training by estimating split gains mainly from large-gradient examples and by bundling rarely-nonzero features; CatBoost (Prokhorenkova et al., NeurIPS 2018) identifies a "prediction shift" caused by target leakage in standard boosting and categorical encodings, and fixes it with ordered boosting.

**Boosting's knobs interact.** A smaller learning rate needs more rounds but usually generalizes better. In script 3, with $\nu = 1$ test MSE bottoms out near 100 rounds (0.275) and then rises (0.318 at 500); with $\nu = 0.1$ it is still improving at 500 (0.232). Pick a small learning rate, choose the rounds by **early stopping** on validation data, then tune tree size, minimum leaf size, and subsampling.

## 6. Practical tabular modeling

A default recipe for a table of rows and columns (the subject of capstone 1):

1. **Baselines first** (L1-01): the majority class or mean, then a regularized linear model (L1-02).
2. **A gradient-boosted tree library** with early stopping: scikit-learn's `HistGradientBoosting*`, LightGBM, XGBoost, or CatBoost. The scikit-learn user guide notes that its histogram estimators (features binned before split search) can be orders of magnitude faster than `GradientBoosting*` beyond tens of thousands of rows, and handle missing values and categorical features natively. In script 3, `HistGradientBoostingRegressor` with defaults reached test MSE 0.219 in 0.2 seconds (with 31-leaf trees, so not like for like).
3. **A random forest** as a low-tuning comparison and sanity check.
4. **Tune a few things well**: learning rate and rounds (via early stopping), tree size, minimum leaf size, subsampling. Use cross-validation that respects your data's structure (time order, groups) as in L1-01.
5. **Watch the data, not just the model**: leakage, encodings, train/serve skew (L1-05). Trees will happily exploit a leaked ID column.

Trees do not extrapolate (leaves predict constants within the training range of the target), represent smooth or rotated boundaries compactly, or use raw text and images without features from elsewhere.

## 7. Why tree ensembles still often beat deep nets on tabular data

**The 2021-2022 evidence.** Shwartz-Ziv and Armon (arXiv:2106.03253; Information Fusion, 2022) found XGBoost beat several deep tabular models, including on those models' own datasets, with much less tuning; an ensemble of both did best. Grinsztajn, Oyallon, and Varoquaux [Grinsztajn et al., 2022] benchmarked 45 datasets with about 20,000 compute-hours of hyperparameter search and found tree-based models still state of the art on medium-sized data (around 10,000 samples). They also asked *why*, and identified three inductive biases favoring trees: robustness to **uninformative features**, respecting the **orientation of the data** (individual columns are meaningful, so the problem is not rotation-invariant), and the ability to fit **irregular functions** that change abruptly. The gap was not mainly due to categorical features and did not vanish with tuning.

Script 4 echoes two of their experiments on one dataset. Adding 24 noise columns barely hurts boosted trees (0.219 to 0.233) but hurts an MLP badly (0.284 to 0.418). A random rotation, which scrambles the meaning of individual columns, leaves ridge exactly unchanged (its L2 penalty is rotation-invariant) and the MLP about the same, but hurts both tree ensembles (boosting: 0.219 to 0.361). Trees exploit column orientation: a strength when columns mean something, a weakness when the signal lives in combinations.

**The gap is smaller and more nuanced than slogans suggest.** McElfresh et al. (2023) compared 19 algorithms on 176 datasets and argued the debate is overemphasized: on many datasets the difference is negligible, or light tuning of a boosted tree matters more than the family. Boosted trees handled skewed or heavy-tailed features and other irregularities much better. And TabPFN, a transformer pretrained on synthetic tabular tasks, did best on average even when restricted to 3,000 training points.

**Tabular foundation models (as of October 2026).** TabPFN's Nature paper [Hollmann et al., 2025] reports that it outperforms previous methods on datasets of up to 10,000 samples, beating an ensemble of strong baselines tuned for four hours in 2.8 seconds in a classification setting. The TabArena living benchmark [Erickson et al., 2025] concluded that gradient-boosted trees remain strong on practical datasets, that deep learning methods have caught up under larger time budgets with ensembling, and that foundation models excel on smaller datasets. The field moves fast. The TabPFN developers' TabPFN-2.5 report (arXiv:2511.08667) claims support for up to 50,000 rows and 2,000 features and a TabArena lead over tuned trees; their TabPFN-3.5 technical report (arXiv:2609.17895, September 2026) claims a new TabArena state of the art, including on non-i.i.d. data. These are developer-reported. A broader June 2026 benchmark, BeyondArena [Purucker et al., 2026], tested 11 models on 142 datasets including temporal and grouped splits: foundation models were best on tiny-to-medium i.i.d. data, while tree-based and deep learning models still dominated on non-i.i.d., large, and high-dimensional data. Several authors overlap across the TabArena, BeyondArena, and TabPFN papers, so these are not fully independent assessments.

**An honest summary.** For a typical medium-sized table, a well-tuned gradient-boosted tree is still a fast, strong, low-risk default. On small i.i.d. tables, tabular foundation models have strong published results and are worth trying. Deep models and cross-model ensembles can match or beat trees with enough compute. Rankings depend on dataset selection, size, validation scheme, tuning budget, and metric, and many benchmarks are run by people who also build some of the models. When it matters, compare on your own data with an honest protocol (L1-01, L4-03).

## 8. Feature importance and its pitfalls

**Impurity-based importance (MDI, mean decrease in impurity)** sums, for each feature, the sample-weighted impurity reduction of every split on it, averaged over trees. It is free, and it is what `feature_importances_` returns. Its problems:

- **Bias toward features with many possible split points.** Strobl et al. (2007) showed that random-forest importance measures are unreliable when predictors differ in scale or number of categories, traced it to biased variable selection in the trees and to bootstrap sampling with replacement, and showed that subsampling without replacement gives more reliable measures.
- **Computed on training data**, so it rewards features the model used to overfit. The scikit-learn example comparing MDI with permutation importance [scikit-learn, permutation vs MDI] demonstrates both problems with planted random features.

**Permutation importance** shuffles one column on held-out data and measures the drop in score. It works for any model and reflects what the model uses to generalize. Its problems:

- **Correlated features share credit.** Shuffle one of two near-copies and the model leans on the other, so both look less important than the information they carry.
- **Shuffling creates unrealistic rows** (a 10-room house with 1 occupant), so you probe the model where it has no data.
- **It measures importance to this model**, not to the outcome in the world, and certainly not causal effect.

Script 5 plants three traps. A pure-noise continuous column gets MDI 0.017, more than six times the 0.003 of a pure-noise binary column, though both are useless; on test data their permutation importance is zero. MDI ranks `AveOccup` third, while test permutation importance ranks `Latitude` first. And adding a near-copy of `MedInc` drops `MedInc`'s test permutation importance from 0.646 to 0.276 (and the copy's is 0.138): the two together look like less than the original alone.

Remedies: use held-out data, permute correlated groups together, check stability across seeds and folds, and be clear about the question. Dropping a feature and refitting answers yet another one: how much worse is the best model without it?

## Hands-on

Run from the repository root, for example `.venv/bin/python code/L1-03-trees-and-ensembles/tree_from_scratch.py`. Scripts 2 to 5 use the California housing data (20,640 census block groups from the 1990 US census, 8 numeric features, target is median house value in units of 100,000 US dollars). scikit-learn downloads it on first use (a few hundred kilobytes; the data originally come from StatLib) and caches it in `~/scikit_learn_data`; the other datasets are bundled. Outputs are from scikit-learn 1.9.1 on Linux. Each script ran in under 35 seconds on 4 cores; printed timings vary by machine.

### Script 1: a classification tree from scratch

```python
"""A CART-style classification tree from scratch (Gini or entropy), compared
with scikit-learn's DecisionTreeClassifier at several depths.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def impurity(counts, criterion):
    """counts: (..., K) class counts. Returns Gini or entropy per row."""
    n = counts.sum(-1, keepdims=True)
    p = counts / np.maximum(n, 1)
    if criterion == "gini":
        return 1 - (p ** 2).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=-1)


def best_split(X, y, K, criterion):
    """Scan every feature and every midpoint between sorted distinct values.
    Returns (feature, threshold, weighted child impurity) or None."""
    n = len(y)
    best = (None, None, impurity(np.bincount(y, minlength=K), criterion))
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="stable")
        xs, ys = X[order, j], y[order]
        left = np.cumsum(np.eye(K)[ys], axis=0)[:-1]   # counts left of each cut
        right = left[-1] + np.eye(K)[ys[-1]] - left     # total minus left
        n_left = np.arange(1, n)
        score = (n_left * impurity(left, criterion)
                 + (n - n_left) * impurity(right, criterion)) / n
        score[xs[1:] == xs[:-1]] = np.inf               # can't cut between equal values
        i = np.argmin(score)
        if score[i] < best[2] - 1e-12:
            best = (j, (xs[i] + xs[i + 1]) / 2, score[i])
    return best if best[0] is not None else None


def build(X, y, K, depth, max_depth, criterion, min_samples=2):
    counts = np.bincount(y, minlength=K)
    leaf = {"leaf": True, "pred": int(np.argmax(counts))}
    if depth == max_depth or len(y) < min_samples or counts.max() == len(y):
        return leaf
    split = best_split(X, y, K, criterion)
    if split is None:
        return leaf
    j, t, _ = split
    go_left = X[:, j] <= t
    return {"leaf": False, "j": j, "t": t,
            "l": build(X[go_left], y[go_left], K, depth + 1, max_depth, criterion),
            "r": build(X[~go_left], y[~go_left], K, depth + 1, max_depth, criterion)}


def predict_one(node, x):
    while not node["leaf"]:
        node = node["l"] if x[node["j"]] <= node["t"] else node["r"]
    return node["pred"]


def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])


# A worked example: 10 points, one feature.
x_toy = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], dtype=float)
y_toy = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
j, t, s = best_split(x_toy, y_toy, 2, "gini")
print(f"toy: parent Gini = {impurity(np.bincount(y_toy), 'gini'):.3f}; "
      f"best cut x <= {t}, weighted child Gini = {s:.3f}")

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
print("\ndepth  criterion  ours train/test   sklearn train/test")
for crit in ["gini", "entropy"]:
    for d in [1, 2, 3, 5, 8, None]:
        tree = build(X_tr, y_tr, 2, 0, d if d is not None else 10**9, crit)
        sk = DecisionTreeClassifier(criterion=crit, max_depth=d, random_state=0).fit(X_tr, y_tr)
        print(f"{str(d):>5s}  {crit:9s}  {np.mean(predict(tree, X_tr) == y_tr):.3f} / "
              f"{np.mean(predict(tree, X_te) == y_te):.3f}     "
              f"{sk.score(X_tr, y_tr):.3f} / {sk.score(X_te, y_te):.3f}")

# sklearn visits features in a random order, so tied splits are broken
# differently per seed, even with all features considered.
accs = [DecisionTreeClassifier(random_state=s).fit(X_tr, y_tr).score(X_te, y_te)
        for s in range(10)]
print("\nfull-depth sklearn tree, test accuracy over 10 seeds:", np.round(accs, 3))
```

Output:

```text
toy: parent Gini = 0.500; best cut x <= 4.5, weighted child Gini = 0.167

depth  criterion  ours train/test   sklearn train/test
    1  gini       0.932 / 0.889     0.932 / 0.889
    2  gini       0.942 / 0.906     0.942 / 0.906
    3  gini       0.980 / 0.906     0.980 / 0.901
    5  gini       0.997 / 0.918     0.997 / 0.912
    8  gini       1.000 / 0.918     1.000 / 0.906
 None  gini       1.000 / 0.918     1.000 / 0.906
    1  entropy    0.932 / 0.889     0.932 / 0.889
    2  entropy    0.932 / 0.883     0.932 / 0.883
    3  entropy    0.977 / 0.912     0.977 / 0.912
    5  entropy    0.997 / 0.930     0.997 / 0.930
    8  entropy    1.000 / 0.924     1.000 / 0.930
 None  entropy    1.000 / 0.924     1.000 / 0.930

full-depth sklearn tree, test accuracy over 10 seeds: [0.906 0.906 0.93  0.912 0.906 0.906 0.918 0.936 0.912 0.912]
```

Our tree matches scikit-learn exactly at depths 1 and 2 and closely after. The differences are tie-breaking: scikit-learn visits features in a random order at each node. Its own full-depth tree ranges from 0.906 to 0.936 test accuracy depending only on `random_state`, which is the instability of single trees in one line. Training accuracy reaches 1.000 while test accuracy stalls near 0.92.

### Script 2: bagging and random forests

```python
"""Why averaging trees works: bagging by hand, then random forests with
different max_features, measuring single-tree error, tree-to-tree error
correlation, and ensemble error on the California housing data.
"""
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

X, y = fetch_california_housing(return_X_y=True)  # downloads ~400 KB once, then cached
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
rng = np.random.default_rng(0)


def mse(pred):
    return np.mean((pred - y_te) ** 2)


# 1. Bagging by hand: fit full-depth trees on bootstrap samples, average them.
preds = []
for b in range(100):
    idx = rng.integers(0, len(y_tr), len(y_tr))     # sample n rows with replacement
    tree = DecisionTreeRegressor(random_state=b).fit(X_tr[idx], y_tr[idx])
    preds.append(tree.predict(X_te))
preds = np.array(preds)
print(f"baseline (predict training mean): test MSE {mse(np.full_like(y_te, y_tr.mean())):.3f}")
print("bagged trees:  B    test MSE")
for B in [1, 2, 5, 10, 25, 50, 100]:
    print(f"            {B:4d}   {mse(preds[:B].mean(0)):.3f}")


# 2. Random forests: decorrelate trees by restricting features at each split.
def forest_stats(max_features):
    rf = RandomForestRegressor(n_estimators=100, max_features=max_features,
                               oob_score=True, n_jobs=-1, random_state=0).fit(X_tr, y_tr)
    P = np.array([t.predict(X_te) for t in rf.estimators_])
    E = P - y_te                                    # each tree's errors on test rows
    C = np.corrcoef(E)
    rho = C[np.triu_indices_from(C, k=1)].mean()   # mean pairwise error correlation
    return np.mean(E ** 2), rho, mse(rf.predict(X_te)), 1 - rf.oob_score_


print("\nmax_features  single-tree MSE  error corr rho  formula  forest MSE  OOB MSE")
var_y = np.var(y_tr)  # OOB R^2 is computed against training targets, so this converts it to MSE
for mf in [1, 2, 3, 5, 8]:
    single, rho, forest, oob_unexpl = forest_stats(mf)
    formula = single * (rho + (1 - rho) / 100)   # rho*s + (1 - rho)*s/B with B = 100
    print(f"{mf:12d}  {single:15.3f}  {rho:14.3f}  {formula:7.3f}  {forest:10.3f}"
          f"  {oob_unexpl * var_y:7.3f}")
```

Output:

```text
baseline (predict training mean): test MSE 1.322
bagged trees:  B    test MSE
               1   0.623
               2   0.423
               5   0.329
              10   0.299
              25   0.280
              50   0.276
             100   0.272

max_features  single-tree MSE  error corr rho  formula  forest MSE  OOB MSE
           1            0.778           0.350    0.278       0.277    0.284
           2            0.634           0.385    0.248       0.247    0.250
           3            0.591           0.412    0.247       0.247    0.246
           5            0.579           0.442    0.259       0.260    0.252
           8            0.583           0.462    0.273       0.273    0.265
```

One full-depth tree has test MSE 0.623; averaging 100 bootstrap trees cuts it to 0.272, with most of the gain in the first 10 trees, as the $(1-\rho)/B$ term predicts. Section 4 discussed the second table. The OOB MSE tracks the test MSE to within about 0.01 without touching the test set.

### Script 3: gradient boosting from scratch

```python
"""Gradient boosting from scratch.
Part 1: squared loss (fit trees to residuals) on California housing, with and
without shrinkage, checked against scikit-learn.
Part 2: log-loss with Newton leaf values (the XGBoost-style second-order step)
on the breast cancer data.
"""
import time

import numpy as np
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingRegressor)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

X, y = fetch_california_housing(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
CHECK = [1, 10, 50, 100, 200, 500]


def boost_l2(lr, n_rounds=500, depth=3):
    f_tr = np.full(len(y_tr), y_tr.mean())      # F_0: the best constant
    f_te = np.full(len(y_te), y_tr.mean())
    out = {}
    for m in range(1, n_rounds + 1):
        resid = y_tr - f_tr                     # negative gradient of (1/2)(y - F)^2
        t = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(X_tr, resid)
        f_tr += lr * t.predict(X_tr)
        f_te += lr * t.predict(X_te)
        if m in CHECK:
            out[m] = (np.mean((y_tr - f_tr) ** 2), np.mean((y_te - f_te) ** 2))
    return out


print("rounds   lr=1.0 train/test    lr=0.1 train/test")
r1, r01 = boost_l2(1.0), boost_l2(0.1)
for m in CHECK:
    print(f"{m:6d}   {r1[m][0]:.3f} / {r1[m][1]:.3f}       {r01[m][0]:.3f} / {r01[m][1]:.3f}")

t0 = time.perf_counter()
gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=3,
                               random_state=0).fit(X_tr, y_tr)
t1 = time.perf_counter()
hgb = HistGradientBoostingRegressor(random_state=0).fit(X_tr, y_tr)
t2 = time.perf_counter()
print(f"\nsklearn GradientBoostingRegressor (500 x depth 3, lr 0.1): test MSE "
      f"{np.mean((gb.predict(X_te) - y_te) ** 2):.3f}  [{t1 - t0:.1f} s]")
print(f"sklearn HistGradientBoostingRegressor (defaults): test MSE "
      f"{np.mean((hgb.predict(X_te) - y_te) ** 2):.3f}, {hgb.n_iter_} rounds  [{t2 - t1:.1f} s]")

# ---- Part 2: log-loss with Newton leaves ----
Xb, yb = load_breast_cancer(return_X_y=True)
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.3, random_state=0, stratify=yb)


def boost_logloss(n_rounds=200, lr=0.1, depth=3, lam=1.0):
    p0 = yb_tr.mean()
    f_tr = np.full(len(yb_tr), np.log(p0 / (1 - p0)))   # F_0 = log-odds of the base rate
    f_te = np.full(len(yb_te), f_tr[0])
    for _ in range(n_rounds):
        p = 1 / (1 + np.exp(-f_tr))
        g, h = p - yb_tr, p * (1 - p)           # first and second derivative wrt F
        # The tree picks the partition using the gradient ...
        t = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(Xb_tr, -g)
        # ... then each leaf gets the Newton step -sum(g) / (sum(h) + lambda).
        leaf_tr, leaf_te = t.apply(Xb_tr), t.apply(Xb_te)
        value = {L: -g[leaf_tr == L].sum() / (h[leaf_tr == L].sum() + lam)
                 for L in np.unique(leaf_tr)}
        f_tr += lr * np.array([value[L] for L in leaf_tr])
        f_te += lr * np.array([value[L] for L in leaf_te])
    return f_te


gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3,
                                 random_state=0).fit(Xb_tr, yb_tr)
pg = gbc.predict_proba(Xb_te)[:, 1]


def report(name, p):
    ll = -np.mean(yb_te * np.log(p) + (1 - yb_te) * np.log(1 - p))
    print(f"{name:38s} test accuracy {np.mean((p > 0.5) == yb_te):.3f}, test log-loss {ll:.4f}")


print()
for lam in [0.0, 1.0]:
    report(f"log-loss boosting, ours, lambda={lam}:", 1 / (1 + np.exp(-boost_logloss(lam=lam))))
report("sklearn GradientBoostingClassifier:", pg)
```

Output:

```text
rounds   lr=1.0 train/test    lr=0.1 train/test
     1   0.611 / 0.650       1.197 / 1.190
    10   0.311 / 0.337       0.636 / 0.652
    50   0.207 / 0.279       0.312 / 0.341
   100   0.162 / 0.275       0.259 / 0.289
   200   0.117 / 0.287       0.220 / 0.259
   500   0.060 / 0.318       0.171 / 0.232

sklearn GradientBoostingRegressor (500 x depth 3, lr 0.1): test MSE 0.232  [9.0 s]
sklearn HistGradientBoostingRegressor (defaults): test MSE 0.219, 100 rounds  [0.2 s]

log-loss boosting, ours, lambda=0.0:   test accuracy 0.942, test log-loss 0.2372
log-loss boosting, ours, lambda=1.0:   test accuracy 0.936, test log-loss 0.1660
sklearn GradientBoostingClassifier:    test accuracy 0.942, test log-loss 0.2388
```

With $\nu = 0.1$ our booster gives the same test MSE as scikit-learn's `GradientBoostingRegressor` (0.232). For log-loss, $\lambda = 0$ reproduces scikit-learn closely; I have not pinned down the small remaining difference (scikit-learn's default `friedman_mse` split criterion is one candidate). $\lambda = 1$ gives clearly better log-loss at the cost of one more misclassified test point.

### Script 4: trees vs an MLP under two perturbations

```python
"""A small, single-dataset echo of two experiments in Grinsztajn et al. (2022):
how do a tree ensemble and an MLP react to (a) extra uninformative features and
(b) a random rotation of the feature space? This is a demonstration of a
mechanism, not a benchmark: one dataset, one split, no tuning.
"""
import time

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = fetch_california_housing(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
rng = np.random.default_rng(0)

# Standardize once so that the rotation below mixes features on equal footing.
sc = StandardScaler().fit(X_tr)
Z_tr, Z_te = sc.transform(X_tr), sc.transform(X_te)


def models():
    return {
        "ridge": RidgeCV(alphas=np.logspace(-3, 3, 13)),
        "random forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=2,
                                               n_jobs=-1, random_state=0),
        "hist GBDT": HistGradientBoostingRegressor(random_state=0),
        "MLP": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=300,
            random_state=0)),
    }


def run(A_tr, A_te):
    row = {}
    for name, m in models().items():
        m.fit(A_tr, y_tr)
        row[name] = np.mean((m.predict(A_te) - y_te) ** 2)
    return row


# (a) 24 extra columns of pure noise. (b) a random orthogonal rotation Q.
noise_tr, noise_te = rng.normal(size=(len(Z_tr), 24)), rng.normal(size=(len(Z_te), 24))
Q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
settings = {
    "original 8 features": (Z_tr, Z_te),
    "+24 noise features": (np.hstack([Z_tr, noise_tr]), np.hstack([Z_te, noise_te])),
    "randomly rotated": (Z_tr @ Q, Z_te @ Q),
}

t0 = time.perf_counter()
print(f"{'test MSE':22s}" + "".join(f"{k:>15s}" for k in models()))
for label, (A_tr, A_te) in settings.items():
    row = run(A_tr, A_te)
    print(f"{label:22s}" + "".join(f"{v:15.3f}" for v in row.values()))
print(f"(total time {time.perf_counter() - t0:.0f} s)")
```

Output:

```text
test MSE                        ridge  random forest      hist GBDT            MLP
original 8 features             0.541          0.271          0.219          0.284
+24 noise features              0.542          0.303          0.233          0.418
randomly rotated                0.541          0.366          0.361          0.270
(total time 21 s)
```

Section 7 discussed these results. Caveats: one dataset, one split, default or near-default hyperparameters, and scikit-learn's `MLPRegressor` is a plain MLP, not a modern tabular deep learning architecture. Treat this as an illustration of mechanisms that Grinsztajn et al. measured properly, not as evidence about which family wins.

### Script 5: feature importance traps

```python
"""Impurity-based (MDI) importance vs permutation importance, with planted
traps: a pure-noise continuous column, a pure-noise binary column, and a
near-copy of the strongest feature.
"""
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
X, y, names = data.data, data.target, list(data.feature_names)
rng = np.random.default_rng(0)
n = len(y)
X = np.column_stack([
    X,
    rng.normal(size=n),                          # noise, ~20k distinct values
    rng.integers(0, 2, n),                       # noise, 2 distinct values
    X[:, 0] + rng.normal(0, 0.1 * X[:, 0].std(), n),   # near-copy of MedInc
])
names += ["noise_cont", "noise_bin", "MedInc_copy"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)


def importances(X_tr, X_te, cols):
    rf = RandomForestRegressor(n_estimators=100, max_features=0.5, n_jobs=-1,
                               random_state=0).fit(X_tr, y_tr)
    pi_tr = permutation_importance(rf, X_tr, y_tr, n_repeats=5, random_state=0, n_jobs=-1)
    pi_te = permutation_importance(rf, X_te, y_te, n_repeats=5, random_state=0, n_jobs=-1)
    print(f"  train R^2 {rf.score(X_tr, y_tr):.3f}, test R^2 {rf.score(X_te, y_te):.3f}")
    print(f"  {'feature':12s} {'MDI':>7s} {'perm(train)':>12s} {'perm(test)':>11s}")
    for i in np.argsort(-rf.feature_importances_):
        print(f"  {cols[i]:12s} {rf.feature_importances_[i]:7.3f} "
              f"{pi_tr.importances_mean[i]:12.3f} {pi_te.importances_mean[i]:11.3f}")
    return rf


print("All features (permutation importance = drop in R^2 when a column is shuffled):")
importances(X_tr, X_te, names)

# Remove the copy: what MedInc "really" carries becomes visible again.
keep = [i for i, c in enumerate(names) if c != "MedInc_copy"]
print("\nWithout MedInc_copy:")
importances(X_tr[:, keep], X_te[:, keep], [names[i] for i in keep])
```

Output:

```text
All features (permutation importance = drop in R^2 when a column is shuffled):
  train R^2 0.972, test R^2 0.794
  feature          MDI  perm(train)  perm(test)
  MedInc         0.266        0.379       0.276
  MedInc_copy    0.231        0.215       0.138
  AveOccup       0.123        0.270       0.180
  Latitude       0.089        0.351       0.319
  Longitude      0.088        0.288       0.244
  AveRooms       0.072        0.075       0.025
  HouseAge       0.056        0.114       0.060
  AveBedrms      0.030        0.032       0.009
  Population     0.027        0.026       0.006
  noise_cont     0.017        0.014       0.000
  noise_bin      0.003        0.003       0.000

Without MedInc_copy:
  train R^2 0.974, test R^2 0.808
  feature          MDI  perm(train)  perm(test)
  MedInc         0.421        0.771       0.646
  AveOccup       0.127        0.270       0.181
  Latitude       0.113        0.430       0.394
  Longitude      0.108        0.355       0.305
  AveRooms       0.097        0.091       0.036
  HouseAge       0.053        0.102       0.055
  AveBedrms      0.033        0.032       0.008
  Population     0.027        0.026       0.005
  noise_cont     0.018        0.014      -0.000
  noise_bin      0.003        0.002       0.000
```

Training-set permutation importances are inflated relative to test (train $R^2$ 0.97, test 0.79). A value like `-0.000` means shuffling made test predictions very slightly better: noise.

## Common mistakes

- **Trusting a single tree's structure.** Change the seed or a few rows and it changes (script 1).
- **Too many boosting rounds without early stopping**, or a large learning rate with many rounds (script 3, $\nu = 1$).
- **Tuning on the test set.** Early stopping and hyperparameter search need their own validation data (L1-01).
- **Assuming the defaults are right.** `max_features=1.0` was not the best choice for the regression forest in script 2.
- **Reading `feature_importances_` as truth.** MDI is biased toward high-cardinality features and computed on training data. Use held-out permutation importance and handle correlated features.
- **Expecting trees to extrapolate.** A tree's predictions are bounded by the training targets in its leaves; trends beyond the training range are flattened.
- **One-hot encoding a very high-cardinality feature** when native categorical support or a fold-fitted target encoding is available (L1-05).
- **Believing benchmark slogans** in either direction. Results depend on data size, structure, and budget.

## Exercises

1. [warm-up] Compute the entropy-based gain for the three cuts in the worked example in section 2. Is there still a tie?
2. [warm-up] Show that $(1 - 1/n)^n \to e^{-1}$, and measure the OOB fraction empirically for $n = 100$ and $n = 10{,}000$.
3. [core] Extend `build` in script 1 to regression (variance impurity, mean leaves) and compare with `DecisionTreeRegressor` on the diabetes data.
4. [core] In script 2, plot (or tabulate) forest test MSE against the number of trees $B$ for `max_features` of 1 and 8. Fit $\rho s + (1-\rho) s / B$ to each curve.
5. [core] Add early stopping to `boost_l2`: hold out 20% of the training set and stop when validation MSE has not improved for 20 rounds. How many rounds does $\nu = 0.1$ need?
6. [core] In script 5, permute `MedInc` and `MedInc_copy` together (one shared permutation) and compare the drop in test $R^2$ with each permuted alone.
7. [stretch] Implement the XGBoost gain formula directly: grow depth-3 trees by choosing splits with $\text{Gain}$ from $g_i, h_i$ instead of fitting a regression tree to $-g$. Compare with `boost_logloss`.
8. [stretch] Design a synthetic dataset where an MLP clearly beats boosted trees with default settings (hint: think about section 7's inductive biases), and one where the reverse holds. Explain each.

### Hints and answers

1. Parent entropy is 1 bit. The 4.5 cut leaves $0.6 \cdot H(1/6) = 0.6 \cdot 0.650 = 0.390$, the same for 6.5 by symmetry, and the 5.5 cut leaves $H(0.2) = 0.722$. Still a tie, gain 0.610.
2. $\log(1 - 1/n)^n = n \log(1 - 1/n) \to -1$. Both empirical fractions should be close to 0.368.
3. Replace class counts with running sums of $y$ and $y^2$; the variance of a set is $\overline{y^2} - \bar{y}^2$.
4. The 1-feature forest has lower $\rho$ but higher $s$; it converges more slowly in $B$ but to a floor set by $\rho s$.
5. Use the validation curve, not the test set. The answer depends on your split; on this data, $\nu = 0.1$ keeps improving beyond 500 rounds, so expect a large number.
6. When I ran it, the joint drop was about 0.78: far more than either alone (0.276, 0.138), and more than their sum. Permuting one copy leaves the information available through the other.
7. Gain for a candidate split needs only $G$ and $H$ sums on each side: cumulative sums over sorted feature values, as in `best_split`.
8. Try a smooth function of a rotated combination, such as $y = \sin(\mathbf{u}^\top \mathbf{x})$ for a random direction $\mathbf{u}$ with a few informative features, for the MLP; and many uninformative features, heavy-tailed inputs, and sharp thresholds on single columns for the trees. Check your predictions by running them.

## Check your understanding

1. **Why does a full-depth tree overfit, and why do ensembles fix it better than pruning?** It carves the space until each training point sits in a pure leaf, so it fits noise and depends heavily on the sample. Pruning cuts variance by simplifying the tree, which adds bias. Averaging many different deep trees cuts variance while keeping their low bias.

2. **What does `max_features` trade off in a random forest?** Fewer features per split make each tree weaker but less correlated with the others. Ensemble variance has a floor of $\rho\sigma^2$ that more trees cannot remove, so lowering $\rho$ can more than pay for weaker trees, up to a point (script 2's optimum is in the middle).

3. **What does a tree in gradient boosting fit, and why use a small learning rate?** The negative gradient of the loss at the current predictions: residuals for squared loss, $y - p$ for log-loss. Small steps in function space mean many trees each contribute a little, which regularizes the fit and makes the number of rounds a smooth control for early stopping.

4. **Name two reasons a tree ensemble may beat an MLP on tabular data, and one setting where it might not.** Trees are robust to uninformative features, exploit the meaning of individual columns, and fit abrupt functions. On small i.i.d. tables, tabular foundation models report better results, and when signal lies in smooth combinations of features, neural nets can win.

5. **Why can impurity-based importance rank a pure-noise feature above zero, and what should you use instead?** MDI comes from training-set splits, and a continuous noise feature offers many thresholds that reduce impurity by chance, especially in deep trees. Use held-out permutation importance (permuting correlated groups together) and check stability.

## Further reading

- **Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor, "An Introduction to Statistical Learning with Applications in Python"** (2023, Springer). https://www.statlearning.com/
  *Read for:* a gentle chapter on trees, bagging, forests, and boosting with Python labs; free PDF. *Level:* intro.
- **scikit-learn developers, "Permutation Importance vs Random Forest Feature Importance (MDI)"** (scikit-learn 1.9.1 examples, accessed October 2026). https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
  *Read for:* a runnable demonstration of MDI's two main biases. *Level:* intro.
- **Leo Breiman, "Bagging Predictors"** (1996, Machine Learning 24:123-140). https://doi.org/10.1007/BF00058655
  *Read for:* the original bagging paper and why instability matters. Publisher page; may require a subscription. *Level:* intermediate.
- **Leo Breiman, "Random Forests"** (2001, Machine Learning 45(1):5-32). https://doi.org/10.1023/A:1010933404324
  *Read for:* random feature selection, OOB estimates, and the strength-correlation view. Publisher page; may require a subscription. *Level:* intermediate.
- **Jerome H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine"** (2001, Annals of Statistics 29(5):1189-1232). https://projecteuclid.org/euclid.aos/1013203451
  *Read for:* gradient boosting as steepest descent in function space, with algorithms for several losses. *Level:* advanced.
- **Tianqi Chen, Carlos Guestrin, "XGBoost: A Scalable Tree Boosting System"** (2016, KDD; arXiv:1603.02754). https://arxiv.org/abs/1603.02754
  *Read for:* the regularized second-order objective, leaf-weight and gain formulas, and systems design. *Level:* intermediate.
- **Carolin Strobl, Anne-Laure Boulesteix, Achim Zeileis, Torsten Hothorn, "Bias in random forest variable importance measures: Illustrations, sources and a solution"** (2007, BMC Bioinformatics 8:25). https://doi.org/10.1186/1471-2105-8-25
  *Read for:* why importance measures favor some predictor types, and a remedy; open access. *Level:* intermediate.
- **Léo Grinsztajn, Edouard Oyallon, Gaël Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?"** (2022, NeurIPS Datasets and Benchmarks Track; arXiv:2207.08815). https://arxiv.org/abs/2207.08815
  *Read for:* a careful benchmark plus experiments on the inductive biases behind the gap. *Level:* intermediate.
- **Duncan McElfresh et al., "When Do Neural Nets Outperform Boosted Trees on Tabular Data?"** (2023, NeurIPS Datasets and Benchmarks Track; arXiv:2305.02997). https://arxiv.org/abs/2305.02997
  *Read for:* a 176-dataset comparison of when each family wins, and by how little. *Level:* intermediate.
- **Noah Hollmann et al., "Accurate predictions on small data with a tabular foundation model"** (2025, Nature 637:319-326). https://doi.org/10.1038/s41586-024-08328-6
  *Read for:* TabPFN, a transformer pretrained on synthetic tables for small-data prediction. *Level:* advanced.
- **Nick Erickson et al., "TabArena: A Living Benchmark for Machine Learning on Tabular Data"** (2025, NeurIPS Datasets and Benchmarks Track; arXiv:2506.16791). https://arxiv.org/abs/2506.16791
  *Read for:* a maintained benchmark showing how validation and ensembling change rankings; check its live leaderboard. *Level:* intermediate.
- **Lennart Purucker et al. (including Gaël Varoquaux and Frank Hutter), "Beyond IID: How General Are Tabular Foundation Models, Really?"** (2026, arXiv:2606.30410, preprint). https://arxiv.org/abs/2606.30410
  *Read for:* where tabular foundation models stop leading (temporal, grouped, large, high-dimensional data). Preprint, not peer reviewed as of October 2026. *Level:* advanced.
