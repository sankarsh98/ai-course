---
title: "Time Series and Graphs"
---

> **Level 3 · Module 9** · Prerequisites: L1-01, L1-03, L1-04, L2-04, L2-05 · Time: 8 to 10 hours · Code: `code/L3-09-time-series-and-graphs/`

## What you will be able to do

- Evaluate a forecaster with rolling-origin backtests and MASE, and show why a shuffled split misleads.
- Implement seasonal naive, Holt-Winters and a seasonal autoregressive model, and check whether their prediction intervals are honest.
- Judge claims about deep and foundation-model forecasters against the evidence, as of October 2026.
- Score time-series anomalies from forecast residuals, and evaluate detectors without the point-adjustment trap.
- Implement message passing on an edge list, verify a GCN layer, and explain what message passing cannot distinguish.
- Predict when a graph neural network helps (homophily) and when it hurts (heterophily, over-smoothing).

## 1. What makes time series different

A **time series** is a sequence $y_1, \dots, y_T$ observed in order. A **forecast** $\hat y_{T+h|T}$ predicts the value $h$ steps ahead (the **horizon**) from information up to the **origin** $T$. Three things separate this from L1-01's tabular setting. Rows are dependent: the **autocorrelation** at lag $k$, the correlation of $y_t$ with $y_{t-k}$, is 0.78 at lag 1 and 0.91 at lag 7 in script 2's daily series. The process drifts, so dataset shift (L1-05) is the normal state. And the question asked in deployment is always about the future, so evaluation must recreate it. Most series decompose into **trend**, **seasonality** with period $m$ (here $m = 7$, a weekly cycle in daily data) and a remainder. Hyndman and Athanasopoulos's free textbook is the reference for everything in sections 1 to 4 [Hyndman and Athanasopoulos, FPP3].

## 2. Validation for time-ordered data

**Rolling-origin evaluation** (backtesting) replays deployment many times: at each origin $T_i$, fit on data up to $T_i$, forecast $h$ steps, score, move the origin on. The window can expand or slide (drop stale history). Report the average and the error by horizon.

**Shuffled splits fail** because each test row is surrounded by training rows, so the model only has to interpolate. In script 1, gradient boosting fitted to random walks using the time index and day of week scores MAE 1.37 under shuffled 5-fold CV, 6.57 under `TimeSeriesSplit`, and 4.81 on the true next 100 days, no better than repeating last week (4.52). Trees cannot extrapolate past the last time index they saw (L1-03).

There is a precise exception. For **purely autoregressive** models (inputs are lagged values) whose errors are uncorrelated, standard K-fold CV is valid and can beat a single holdout [Bergmeir et al., 2018]. The result does not cover time-index features, drifting levels or misspecified models. Rolling origin is never wrong, only more expensive.

Other time-series leaks: features from centred windows or scalers fitted to the whole series; data later revised (use the values available at the time); and reporting the backtest you tuned on. In M5, 415 teams (7.5%) beat the best statistical benchmark with the forecasts they chose to submit, but 672 (12.2%) would have with their best submission, which the organisers attribute to "misleading validation scores" [Makridakis et al., 2022].

**Metrics.** MAE and RMSE are in the series' units and cannot be averaged across series. Percentage errors explode near zero. The **mean absolute scaled error** divides the forecast MAE by the in-sample MAE of the one-season-back forecast [Hyndman and Koehler, 2006]:

$$\text{MASE} = \frac{\frac1h\sum_{k=1}^h |y_{T+k} - \hat y_{T+k|T}|}{\frac{1}{T-m}\sum_{t=m+1}^{T} |y_t - y_{t-m}|}.$$

Below 1 beats that in-sample benchmark. For probabilistic forecasts, also report quantile (pinball) loss or interval coverage.

## 3. Baselines that are hard to beat

The **mean**, **naive** ($\hat y_{T+k} = y_T$), **seasonal naive** (repeat the last season) and **drift** (naive plus the average slope) forecasts cost nothing and belong in every result. On script 1's series, seasonal naive has MASE 0.99 and naive 2.39; naive is fine one step ahead and terrible mid-week.

The big competitions show how high this bar is. **M4** (100,000 series, 61 methods) was won by Smyl's hybrid of exponential smoothing and a recurrent network, ahead of a method that used XGBoost to weight statistical forecasts. Both built on series-specific statistical components and were about as accurate as a median of four statistical methods [Makridakis et al., 2020; Makridakis et al., 2022]. **M5** (42,840 Walmart sales series, 5,507 teams) was the first where all top methods were "pure" machine learning and significantly better than every statistical benchmark: most were built on LightGBM, trained across many series at once, and the top three each averaged several models. The winner beat the best benchmark (bottom-up exponential smoothing) by 22.4%, yet only 7.5% of teams beat it at all [Makridakis et al., 2022]. The lesson is not "statistics beats ML" or the reverse: a well-implemented baseline is a high bar, and many related series with informative covariates are where learned global models shine.

## 4. Exponential smoothing and ARIMA

**Simple exponential smoothing** updates a level $\ell_t = \ell_{t-1} + \alpha e_t$ with $e_t = y_t - \ell_{t-1}$, which makes the forecast a weighted average of the past with weights $\alpha(1-\alpha)^j$ decaying with age $j$. **Holt-Winters** adds a trend $b_t$, damped by $\phi < 1$ so long-range forecasts flatten, and a seasonal state $s_t$. In the additive error-correction form of script 2:

$$\hat y_{t|t-1} = \ell_{t-1} + \phi b_{t-1} + s_{t-m}, \quad \ell_t = \ell_{t-1} + \phi b_{t-1} + \alpha e_t, \quad b_t = \phi b_{t-1} + \beta e_t, \quad s_t = s_{t-m} + \gamma e_t.$$

Written as state-space models with an explicit error (the ETS family), these have a likelihood, so variants can be compared by AIC and intervals derived [FPP3, ch. 8].

**ARIMA** models the autocorrelation directly. An AR($p$) model is $z_t = c + \sum_{i=1}^p \phi_i z_{t-i} + \varepsilon_t$; MA($q$) terms add lagged errors $\sum_j \theta_j \varepsilon_{t-j}$. ARIMA applies this to a differenced series: $y_t - y_{t-1}$ removes trend, $y_t - y_{t-m}$ removes stable seasonality. $\text{ARIMA}(p,d,q)(P,D,Q)_m$ counts orders and differences, chosen from the autocorrelations of the differenced series and by AIC [FPP3, ch. 9]. In script 2, lag-7 differencing turns the lag-7 autocorrelation from 0.91 to $-0.42$, and AIC picks $p = 8$. Script 2 fits only the AR part, by least squares; MA terms need maximum likelihood, which standard libraries provide.

Over 71 origins, Holt-Winters reaches MASE 0.933 and the seasonal AR 0.912, against 1.019 for seasonal naive: real but modest gains of about 10%. The intervals teach more. A constant band of $\pm 1.28$ one-step standard deviations covers the truth only 54% of the time at steps 12 to 14, not 80%, because it ignores compounding uncertainty. Simulating paths through the model's own equations widens the interval from 8.2 to 20.8 units and covers 79% at short horizons (92% at long ones, an over-correction). Check coverage by horizon on a backtest.

## 5. Global, deep and foundation models: the evidence

A **global** model is one set of parameters fitted to windows pooled from many series, each window scaled so different series look alike. Global models need less history per series and can learn patterns shared across series. The M5 organisers report that essentially all of the top 50 methods used this "cross-learning" [Makridakis et al., 2022]. Exercise 9 asks you to build one.

**Deep architectures** have a mixed record. A one-layer linear model on the lag window (LTSF-Linear, including DLinear) outperformed the Transformer long-horizon forecasters of the time on all nine benchmark datasets Zeng and colleagues used [Zeng et al., 2023]. When Tan and colleagues ablated three methods that repurpose a pretrained language model, removing the language model or replacing it with one attention layer did not hurt and usually helped [Tan et al., 2024].

**Foundation models** are pretrained on large corpora of series and used zero-shot. Chronos scales and quantises values into tokens and trains T5-family models (20M to 710M parameters); its authors report that it beats other methods on datasets in its training corpus and is comparable to, sometimes better than, dataset-specific methods on new ones [Ansari et al., 2024]. TimesFM, a decoder-only model over patches, reports zero-shot accuracy close to supervised per-dataset models [Das et al., 2024]. Newer models (Chronos-2, TiRex, TimesFM-2.5 and others) accept covariates.

What the benchmarks say, as of October 2026:

- **GIFT-Eval** (23 datasets, over 144,000 series, 177 million points) ships a pretraining corpus free of its test data, and its leaderboard added a test-data-leakage column in August 2025 [Aksu et al., 2024].
- **fev-bench** (100 tasks, 46 with covariates) scores skill relative to seasonal naive with bootstrapped intervals. In its June 2026 version, Chronos-2 leads with probabilistic and point skill scores of 47.3% and 35.5%, against 20.7% and 16.4% for a statistical ensemble and 23.0% and 23.7% for CatBoost [Shchur et al., 2025]. The margin is large, but the authors are the Chronos team, and leakage reporting is self-declared (0% to 28% for the listed models).
- **Leakage** is the open problem. Test sets overlap with pretraining corpora because the same public datasets are reused, and correlated series overlap in time, for example one electricity meter in training and a neighbouring one in test over the same months [Meyer et al., 2025]. A NeurIPS 2025 workshop study found zero-shot accuracy depends heavily on pretraining domains, and that fine-tuned foundation models often did not clearly beat much smaller task-specific models on unseen data [Karaouli et al., 2025].

A fair summary, as of October 2026: pretrained forecasters are strong general-purpose baselines that beat statistical ensembles on average on the broadest benchmarks, but their advantage on a given problem is an empirical question that contamination makes hard to settle from published numbers. Backtest them on data after their training cutoff, against seasonal naive, ETS or ARIMA, and a global boosted-tree model with your covariates (benchmark contamination in general: L4-03).

## 6. Anomaly detection in time series

Time-series anomalies are **point** (a spike), **contextual** (normal in general, not at this time, like daytime load at 3 am) or **collective** (a level shift, a burst of variance). The detectors of L1-04 treat points as exchangeable and miss context. The time-series move is to score **forecast residuals**, $s_t = |y_t - \hat y_{t|t-1}| / \hat\sigma_t$, with a robust scale (the median absolute deviation, MAD) estimated from the past only, so it runs online. Script 3's **seasonal robust z-score** compares each hour with the same hour on the previous 14 days and catches all six injected segments. A global $z$-score misses two of three spikes and the level shift; it catches the 3 am value only because the trend lifts late values.

**Evaluation** has been the weak point. Under **point adjustment** (PA), common in papers, one flagged point credits a whole true segment. Kim and colleagues showed that random scores can then look state-of-the-art, and that an untrained model matches published methods even without PA [Kim et al., 2022]. In script 3, random scores get point-wise F1 0.11 but PA-F1 0.66, and the weak global $z$-score jumps from 0.27 to 0.92. Wu and Keogh argue that popular benchmarks are also flawed by trivial problems, unrealistically dense anomalies, mislabels and run-to-failure bias [Wu and Keogh, 2023]. Script 3's "best F1 over thresholds" is an oracle number; in practice the threshold is set by how many alerts a person can review.

## 7. Graph data

A **graph** $G = (V, E)$ has $n$ nodes, edges $E$, node features $X \in \mathbb R^{n \times d}$ and adjacency matrix $A$ ($A_{uv} = 1$ if $u$ and $v$ are linked). $\mathcal N(v)$ is the neighbourhood of $v$ and $\deg(v) = |\mathcal N(v)|$. Tasks are node classification, link prediction (recommendation, L3-08) and graph-level prediction (molecules). Applications include molecular property prediction [Gilmer et al., 2017], travel-time estimation in Google Maps, where the authors report over 40% fewer negative ETA outcomes in cities like Sydney [Derrow-Pinion et al., 2021], and GraphCast, a weather model (L3-10) that runs 16 rounds of message passing on an icosahedral multi-mesh [Lam et al., 2023].

Node order is arbitrary, so a node-level model must be **permutation equivariant**, $f(PX, PAP^\top) = P f(X, A)$ for any permutation matrix $P$, and a graph-level model invariant. An MLP on the flattened adjacency matrix is neither. Self-attention without positions is equivariant (L2-05), which is why a Transformer can be read as message passing on a complete graph.

## 8. Message passing and graph neural networks

A **message-passing network** updates node states $h_v$ (with $h_v^{(0)} = x_v$) by [Gilmer et al., 2017]

$$h_v^{(\ell+1)} = U_\ell\Big(h_v^{(\ell)},\ \bigoplus_{u \in \mathcal N(v)} M_\ell\big(h_u^{(\ell)}, h_v^{(\ell)}, e_{uv}\big)\Big),$$

where $\bigoplus$ is a permutation-invariant aggregator (sum, mean, max), $M_\ell, U_\ell$ are learned and $e_{uv}$ are edge features. After $L$ rounds, $h_v$ summarises the $L$-hop neighbourhood; a **readout** (sum or mean over nodes) gives a graph embedding. On an edge list a round is a gather (`x[src]`) and a scatter-add (`index_add_`), linear in the number of edges (script 4).

The **GCN** of Kipf and Welling, a first-order approximation of spectral graph convolution, uses $H^{(\ell+1)} = \sigma(\tilde D^{-1/2} \tilde A \tilde D^{-1/2} H^{(\ell)} W^{(\ell)})$ with $\tilde A = A + I$ and $\tilde D$ its degree matrix: a degree-normalised average over the node and its neighbours, then a shared linear map [Kipf and Welling, 2017]. Script 4 checks the edge-list version against this formula ($6 \times 10^{-8}$) and checks equivariance.

**Expressivity.** Message-passing networks are at most as powerful as the 1-dimensional Weisfeiler-Lehman test at telling graphs apart, and reaching that bound needs sum aggregation, because mean and max cannot tell $\{1, 1\}$ from $\{1\}$; the GIN layer $h_v' = \text{MLP}((1+\epsilon)h_v + \sum_{u} h_u)$ attains it [Xu et al., 2019]. In two triangles and in one hexagon every node has degree 2, so with identical features every node receives identical messages forever: script 4 gets identical embeddings (difference 0.0). A triangle-count feature, which message passing cannot compute itself, separates them.

**Homophily** (edges mostly join same-label nodes) is what makes neighbour averaging denoise. Under **heterophily**, many GNNs do worse than an MLP; separating ego and neighbour embeddings, higher-order neighbourhoods and combining layers help [Zhu et al., 2020]. In script 5 (3 classes, 20 labels per class), at homophily 0.9 the GCN reaches 0.97 against the MLP's 0.50; at 0.1 it falls to 0.37, while an ego+neighbour layer ($hW_{\text{self}} + \text{mean}(h_{\mathcal N})W_{\text{nbr}}$) gets 0.60, because predictably *different* neighbours are informative too.

**Over-smoothing.** The GCN operation is a form of Laplacian smoothing, which is why stacking many layers makes nodes indistinguishable [Li et al., 2018]. In script 5, repeated propagation lifts a linear classifier from 0.57 to 1.00 after 2 to 16 rounds, then drops it to 0.44 at 64 as the rows collapse together.

**Evaluation.** Many GNN papers used one fixed split of the same three citation datasets; other splits reorder the models, and simpler models often win under equal tuning [Shchur et al., 2018]. Script 5 reports spreads over five graphs and seeds for this reason. In link prediction, never leave the test edges in the message-passing graph.

## Hands-on

All data is synthetic. Run from `code/L3-09-time-series-and-graphs/` with the course Python (NumPy 2.5.4, scikit-learn 1.9.1, PyTorch 2.15.0 CPU). Each script took under 35 s on the 24-core machine used (script 5 is slowest, 28 to 32 s across runs, depending on other load). The scripts cap thread counts: in one run with other jobs on the machine, script 2 took 265 s with default threading instead of 4 s, because thousands of tiny least-squares fits were each spread over 24 threads. Trailing digits can change with platform, library versions and thread count.

### Script 1: baselines, backtesting and a shuffled-split leak

```python
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
```

```text
series length 1095, 352 forecast origins, horizon 14 days

         method    MAE   RMSE   MASE
           mean  14.10  16.29   3.37
          naive   9.99  12.63   2.39
 seasonal naive   4.15   5.25   0.99
          drift  10.03  12.68   2.40

MAE by step ahead:   1     3     5     7    14
          naive:  7.05 13.33 11.83  3.92  4.48
 seasonal naive:  3.90  3.90  3.91  3.92  4.48

Gradient boosting on (time index, day of week), 10 random walks, mean MAE:
  shuffled 5-fold CV            1.37   (range  1.22 to  1.48)
  TimeSeriesSplit, 5 folds      6.57   (range  4.84 to  8.25)
  true future, last 100 days    4.81   (range  2.49 to 11.55)
  seasonal naive, same 100      4.52   (range  2.26 to  9.99)
```

Seasonal naive's error is flat for steps 1 to 7 and rises at step 14, when values two weeks old are reused. The true-future error ranges from 2.49 to 11.55 across walks: one holdout period is a noisy measurement.

### Script 2: Holt-Winters and a seasonal AR model from scratch

```python
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
```

```text
autocorrelation at lags [1, 2, 7, 14]
  raw series:            [0.78 0.46 0.91 0.89]
  after lag-7 difference [ 0.61  0.39 -0.42 -0.02]

last fit: alpha=0.62 beta=0.000 gamma=0.05 phi=0.83; AR order chosen by AIC p=8

MASE over 71 origins x 14 steps (lower is better):
  seasonal naive         mean 1.019   steps 1-7 0.946   steps 8-14 1.092
  Holt-Winters (damped)  mean 0.933   steps 1-7 0.852   steps 8-14 1.014
  seasonal AR            mean 0.912   steps 1-7 0.837   steps 8-14 0.986

empirical coverage of nominal 80% intervals:
  simulated paths, steps 1-3: 79%, steps 12-14: 92%   (mean width 8.2 at step 1, 20.8 at step 14)
  constant one-step band,  steps 1-3: 69%, steps 12-14: 54%
```

The fitted $\beta = 0$: the level and seasonal states absorb the trend. Even the simulated intervals slightly under-cover at short horizons, because in-sample residuals understate out-of-sample error.

### Script 3: anomaly detection and point adjustment

```python
"""Time-series anomaly detection on an hourly series with labelled anomalies:
a global threshold, a seasonal robust z-score, and a forecast-residual score.
Then: how point adjustment makes even random scores look excellent."""
import numpy as np

rng = np.random.default_rng(1)
P = 24                                    # hourly data, daily cycle
n = 60 * P                                # 60 days
t = np.arange(n)
daily = 10 * np.sin(2 * np.pi * (t % P) / P - np.pi / 2)          # low at night
y = 50 + daily + 0.01 * t + rng.normal(0, 1.5, n)
label = np.zeros(n, bool)

# Four kinds of anomaly, each a labelled segment (start hour, length).
for s in [400, 700, 1100]:                       # point spikes
    y[s] += 12; label[s] = True
y[900:924] += 6; label[900:924] = True           # level shift lasting one day
night = 1203                                     # contextual: a daytime value at 3 am
y[night:night + 3] = 60 + 0.01 * t[night:night + 3] + rng.normal(0, 1.5, 3)  # daily-peak level
label[night:night + 3] = True
y[1300:1340] += rng.normal(0, 6, 40); label[1300:1340] = True      # variance burst
print(f"{n} points, {label.sum()} labelled anomalous ({label.mean():.1%}) in 6 segments")


def seasonal_robust_z(y, window_days=14):
    """Compare each point with the same hour on previous days: subtract the
    trailing median for that hour, divide by a robust scale (MAD). Uses only
    the past, so it can run online."""
    z = np.zeros(len(y))
    for i in range(window_days * P, len(y)):
        same_hour = y[i - window_days * P:i:P]          # same hour, previous days
        med = np.median(same_hour)
        mad = np.median(np.abs(same_hour - med)) * 1.4826 + 1e-9  # ~std under normality
        z[i] = abs(y[i] - med) / mad
    return z


def forecast_residual(y):
    """|y_t - y_{t-24}|: the error of a seasonal naive forecast, scaled by a
    trailing robust scale of past errors."""
    r = np.zeros(len(y))
    r[P:] = np.abs(y[P:] - y[:-P])
    score = np.zeros(len(y))
    for i in range(8 * P, len(y)):
        score[i] = r[i] / (np.median(r[i - 7 * P:i]) + 1e-9)
    return score


def segments(lab):
    """List of (start, end) index pairs of contiguous True runs."""
    d = np.diff(np.r_[0, lab.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def point_adjust(pred, lab):
    """The 'point adjustment' protocol: if any point of a true segment is
    flagged, count the whole segment as detected."""
    adj = pred.copy()
    for s, e in segments(lab):
        if pred[s:e].any():
            adj[s:e] = True
    return adj


def f1(pred, lab):
    tp = np.sum(pred & lab)
    prec = tp / max(pred.sum(), 1); rec = tp / lab.sum()
    return 0 if tp == 0 else 2 * prec * rec / (prec + rec)


def best_f1(score, lab, adjust=False):
    """Best F1 over thresholds (an optimistic, oracle-tuned number, as in many papers)."""
    best = 0
    for q in np.linspace(0.80, 0.999, 200):
        pred = score > np.quantile(score, q)
        best = max(best, f1(point_adjust(pred, lab) if adjust else pred, lab))
    return best


def segment_recall(score, lab, k):
    """Of the anomaly segments, how many contain at least one of the top-k scores?"""
    top = np.zeros(len(score), bool); top[np.argsort(score)[-k:]] = True
    return [bool(top[s:e].any()) for s, e in segments(lab)]


warm = 14 * P                         # skip the warm-up period the online scores need
scores = {"global |z| of raw value": np.abs(y - y.mean()) / y.std(),
          "seasonal robust z": seasonal_robust_z(y),
          "forecast residual": forecast_residual(y),
          "random scores": rng.random(n)}
names = ["spike", "spike", "spike", "level shift", "night value", "variance burst"]
print(f"\n{'detector':<25} {'F1':>5} {'PA-F1':>6}   segments caught by top-{label.sum()} scores")
for name, s in scores.items():
    s, lab = s[warm:], label[warm:]
    caught = segment_recall(s, lab, label.sum())
    hit = [nm for nm, c in zip(names, caught) if c]
    print(f"{name:<25} {best_f1(s, lab):5.2f} {best_f1(s, lab, adjust=True):6.2f}   {', '.join(hit) or 'none'}")
```

```text
1440 points, 70 labelled anomalous (4.9%) in 6 segments

detector                     F1  PA-F1   segments caught by top-70 scores
global |z| of raw value    0.27   0.92   spike, night value, variance burst
seasonal robust z          0.54   1.00   spike, spike, spike, level shift, night value, variance burst
forecast residual          0.53   0.96   spike, spike, spike, level shift, night value, variance burst
random scores              0.11   0.66   spike, variance burst
```

Point-wise F1 tops out near 0.5 because many hours in the 40-hour variance burst look normal. The forecast-residual score has an "echo": an anomaly at hour $t$ also inflates the residual at $t + 24$, when it becomes the forecast. The 14-day seasonal median is robust to one bad day.

### Script 4: message passing from scratch

```python
"""Message passing from scratch on an edge list, checked four ways:
1. one hand-sized step, 2. sparse GCN layer == dense formula,
3. permutation equivariance, 4. what sum/mean/max aggregation can and cannot
tell apart, including two graphs no message-passing network can distinguish."""
import torch

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False)


def to_directed(edges):
    """Undirected edge list [(u, v), ...] -> (2, E) tensor holding both directions."""
    e = torch.tensor(edges, dtype=torch.long).t()
    return torch.cat([e, e.flip(0)], dim=1)


def aggregate(msg, dst, n, how="sum"):
    """Combine the messages arriving at each node. msg: (E, d), dst: (E,)."""
    out = torch.zeros(n, msg.shape[1])
    if how == "max":
        idx = dst[:, None].expand_as(msg)
        return out.scatter_reduce_(0, idx, msg, "amax", include_self=False)  # no edges: 0
    out = out.index_add_(0, dst, msg)
    if how == "mean":
        deg = torch.zeros(n).index_add_(0, dst, torch.ones(len(dst)))
        out = out / deg.clamp(min=1)[:, None]
    return out


def propagate(x, edge_index, how="sum", weight=None):
    """One round: every edge (src -> dst) sends x[src] (optionally scaled)."""
    src, dst = edge_index
    msg = x[src] if weight is None else x[src] * weight[:, None]
    return aggregate(msg, dst, x.shape[0], how)


def gcn_edges(edge_index, n):
    """Add self-loops and the symmetric weights 1/sqrt(deg_u deg_v) of
    Kipf and Welling (2017), where deg counts the self-loop."""
    loops = torch.arange(n).repeat(2, 1)
    ei = torch.cat([edge_index, loops], dim=1)
    deg = torch.zeros(n).index_add_(0, ei[1], torch.ones(ei.shape[1]))
    w = deg[ei[0]].rsqrt() * deg[ei[1]].rsqrt()
    return ei, w


class GCNLayer(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = torch.nn.Linear(d_in, d_out, bias=False)

    def forward(self, x, ei, w):
        return propagate(self.lin(x), ei, "sum", w)


# 1. A path graph 0-1-2-3 with scalar features: watch one step by hand.
ei = to_directed([(0, 1), (1, 2), (2, 3)])
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
print("features         ", x.squeeze().tolist())
print("sum of neighbours ", propagate(x, ei, "sum").squeeze().tolist())
print("mean of neighbours", propagate(x, ei, "mean").squeeze().tolist())
print("max of neighbours ", propagate(x, ei, "max").squeeze().tolist())

# 2. Sparse GCN layer versus the dense formula  D^-1/2 (A + I) D^-1/2 X W.
n, d = 50, 8
pairs = torch.combinations(torch.arange(n), 2)
edges = pairs[torch.rand(len(pairs)) < 0.08].tolist()
ei = to_directed(edges)
x = torch.randn(n, d)
layer = GCNLayer(d, 4)
gei, w = gcn_edges(ei, n)
sparse = layer(x, gei, w)
A = torch.zeros(n, n); A[ei[0], ei[1]] = 1
A_hat = A + torch.eye(n)
D_inv_sqrt = torch.diag(A_hat.sum(1).rsqrt())
dense = D_inv_sqrt @ A_hat @ D_inv_sqrt @ layer.lin(x)
print(f"\nrandom graph: {n} nodes, {len(edges)} edges")
print(f"max |sparse - dense| = {(sparse - dense).abs().max():.2e}")

# 3. Relabel the nodes with a permutation: outputs must be permuted the same way.
perm = torch.randperm(n)
inv = torch.empty_like(perm); inv[perm] = torch.arange(n)   # old id -> new id
gei_p, w_p = gcn_edges(inv[ei], n)
out_p = layer(x[perm], gei_p, w_p)
print(f"max |f(Px) - P f(x)| = {(out_p - sparse[perm]).abs().max():.2e}")

# 4a. Multisets. Node 0 has neighbours with features {1, 1}; node 3 has {1}.
#     Mean and max see the same thing; only sum can tell them apart.
ei = to_directed([(0, 1), (0, 2), (3, 4)])
x = torch.tensor([[0.0], [1.0], [1.0], [0.0], [1.0]])
for how in ["sum", "mean", "max"]:
    h = propagate(x, ei, how)
    print(f"{how:>4}: node 0 -> {h[0].item():.1f}, node 3 -> {h[3].item():.1f}")

# 4b. Two triangles versus one hexagon: 6 nodes, every node has degree 2.
#     With identical starting features, every node receives identical messages
#     at every round, so a message-passing network (here a small random GIN-like
#     net with sum readout) gives both graphs the same embedding.
two_triangles = to_directed([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
hexagon = to_directed([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])
mlps = [torch.nn.Sequential(torch.nn.Linear(d_in, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16))
        for d_in in [1, 16, 16]]


def graph_embedding(ei, x):
    h = x
    for mlp in mlps:
        h = mlp(h + propagate(h, ei, "sum"))       # GIN update with epsilon = 0
    return h.sum(0)                                 # sum readout


ones = torch.ones(6, 1)
gap = (graph_embedding(two_triangles, ones) - graph_embedding(hexagon, ones)).abs().max()
print(f"\nconstant features: max embedding difference = {gap.item():.2e}")
# Give each node a feature that message passing cannot compute itself:
# whether it lies on a triangle (here, the number of closed 3-walks / 2).
def triangles(ei, n):
    A = torch.zeros(n, n); A[ei[0], ei[1]] = 1
    return (torch.linalg.matrix_power(A, 3).diagonal() / 2)[:, None]


gap = (graph_embedding(two_triangles, triangles(two_triangles, 6))
       - graph_embedding(hexagon, triangles(hexagon, 6))).abs().max()
print(f"triangle-count features: max embedding difference = {gap.item():.2e}")
```

```text
features          [1.0, 2.0, 3.0, 4.0]
sum of neighbours  [2.0, 4.0, 6.0, 3.0]
mean of neighbours [2.0, 2.0, 3.0, 3.0]
max of neighbours  [2.0, 3.0, 4.0, 3.0]

random graph: 50 nodes, 106 edges
max |sparse - dense| = 5.96e-08
max |f(Px) - P f(x)| = 1.49e-08
 sum: node 0 -> 2.0, node 3 -> 1.0
mean: node 0 -> 1.0, node 3 -> 1.0
 max: node 0 -> 1.0, node 3 -> 1.0

constant features: max embedding difference = 0.00e+00
triangle-count features: max embedding difference = 2.74e+00
```

The remaining errors are float32 rounding from summing in different orders.

### Script 5: node classification, homophily and over-smoothing

```python
"""Semi-supervised node classification on synthetic graphs (a contextual
stochastic block model): MLP vs GCN vs a GNN that keeps the node's own
features separate, as edges go from homophilous to heterophilous. Then:
over-smoothing when propagation is repeated many times."""
import os

# Many small fits: extra threads mostly add overhead, and on a shared machine
# they can slow things down a lot. Must be set before numpy is imported.
for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(var, "4")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

torch.set_num_threads(4)              # small sparse ops: more threads only add overhead

N, C, D, DEG = 1200, 3, 16, 10        # nodes, classes, feature dim, mean degree


def make_graph(h, seed):
    """Classes are balanced. Features: class mean + noise (weakly informative).
    A fraction h of edges (in expectation) joins nodes of the same class."""
    g = np.random.default_rng(seed)
    y = np.repeat(np.arange(C), N // C)
    means = 1.0 * np.eye(C, D)                   # equally separated class means
    x = means[y] + g.normal(0, 1, (N, D))
    same = y[:, None] == y[None, :]
    p_in = h * DEG / (N / C)
    p_out = (1 - h) * DEG / (N - N / C)
    upper = np.triu(g.random((N, N)) < np.where(same, p_in, p_out), 1)
    src, dst = np.nonzero(upper | upper.T)
    split = g.permutation(N)
    train = np.concatenate([split[y[split] == c][:20] for c in range(C)])  # 20 labels/class
    rest = np.setdiff1d(split, train)
    return (torch.tensor(x, dtype=torch.float32), torch.tensor(y),
            np.stack([src, dst]), train, rest[:300], rest[300:])


def sparse_op(ei, kind):
    """'gcn': D^-1/2 (A+I) D^-1/2 (as checked in script 4); 'mean': D^-1 A."""
    src, dst = torch.tensor(ei[0]), torch.tensor(ei[1])
    if kind == "gcn":
        loops = torch.arange(N)
        src, dst = torch.cat([src, loops]), torch.cat([dst, loops])
    deg = torch.zeros(N).index_add_(0, dst, torch.ones(len(dst)))
    w = (deg[src] * deg[dst]).rsqrt() if kind == "gcn" else 1 / deg[dst].clamp(min=1)
    return torch.sparse_coo_tensor(torch.stack([dst, src]), w, (N, N),
                                   check_invariants=True).coalesce()


class Net(torch.nn.Module):
    """kind='mlp': ignores edges. 'gcn': h' = A_gcn h W.
    'ego+nbr': h' = h W_self + (mean of neighbours) W_nbr, so a node's own
    features are never averaged away (one of the designs Zhu et al. 2020 recommend)."""

    def __init__(self, kind, hidden=32):
        super().__init__()
        self.kind = kind
        self.l1, self.l2 = torch.nn.Linear(D, hidden), torch.nn.Linear(hidden, C)
        self.n1, self.n2 = torch.nn.Linear(D, hidden), torch.nn.Linear(hidden, C)

    def layer(self, h, lin, nbr, op):
        if self.kind == "mlp":
            return lin(h)
        if self.kind == "gcn":
            return torch.sparse.mm(op, lin(h))
        return lin(h) + torch.sparse.mm(op, nbr(h))

    def forward(self, x, op):
        h = torch.relu(self.layer(x, self.l1, self.n1, op))
        h = torch.nn.functional.dropout(h, 0.5, self.training)
        return self.layer(h, self.l2, self.n2, op)


def train_eval(kind, data, seed):
    x, y, ei, tr, va, te = data
    torch.manual_seed(seed)
    op = sparse_op(ei, "gcn" if kind == "gcn" else "mean")
    net = Net(kind)
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, test_at_best = 0, 0
    for _ in range(200):
        net.train()
        loss = torch.nn.functional.cross_entropy(net(x, op)[tr], y[tr])
        opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(x, op).argmax(1)
        val = (pred[va] == y[va]).float().mean().item()
        if val > best_val:                         # model selection on validation only
            best_val, test_at_best = val, (pred[te] == y[te]).float().mean().item()
    return test_at_best


print(f"{N} nodes, {C} classes, mean degree {DEG}, 20 labelled nodes per class")
print("test accuracy, mean (sd) over 5 graphs/seeds; chance = 0.33\n")
print(f"{'homophily':>9} {'MLP':>12} {'GCN':>12} {'ego+nbr':>12}")
for h in [0.9, 0.7, 0.5, 0.33, 0.1]:
    row = []
    for kind in ["mlp", "gcn", "ego+nbr"]:
        accs = [train_eval(kind, make_graph(h, s), s) for s in range(5)]
        row.append(f"{np.mean(accs):.2f} ({np.std(accs):.2f})")
    print(f"{h:>9} " + " ".join(f"{r:>12}" for r in row))

# ---- Over-smoothing: propagate features K times, then fit logistic regression.
x, y, ei, tr, va, te = make_graph(0.9, 0)
op = sparse_op(ei, "gcn")
print("\nrepeated GCN propagation on a homophilous graph (h = 0.9), no learning inside:")
print(f"{'K':>4} {'test acc':>9} {'spread':>8}   (spread: mean distance of unit-length rows from their mean)")
z = x.clone()
for K in range(0, 65):
    if K in [0, 1, 2, 4, 8, 16, 32, 64]:
        clf = LogisticRegression(max_iter=2000).fit(z[tr].numpy(), y[tr].numpy())
        acc = clf.score(z[te].numpy(), y[te].numpy())
        u = torch.nn.functional.normalize(z, dim=1)
        spread = (u - u.mean(0)).norm(dim=1).mean().item()
        print(f"{K:>4} {acc:9.2f} {spread:8.3f}")
    z = torch.sparse.mm(op, z)
```

```text
1200 nodes, 3 classes, mean degree 10, 20 labelled nodes per class
test accuracy, mean (sd) over 5 graphs/seeds; chance = 0.33

homophily          MLP          GCN      ego+nbr
      0.9  0.50 (0.03)  0.97 (0.04)  0.93 (0.02)
      0.7  0.50 (0.03)  0.81 (0.13)  0.75 (0.06)
      0.5  0.50 (0.03)  0.52 (0.13)  0.53 (0.08)
     0.33  0.50 (0.03)  0.35 (0.10)  0.43 (0.05)
      0.1  0.50 (0.03)  0.37 (0.10)  0.60 (0.10)

repeated GCN propagation on a homophilous graph (h = 0.9), no learning inside:
   K  test acc   spread   (spread: mean distance of unit-length rows from their mean)
   0      0.57    0.990
   1      0.92    0.930
   2      1.00    0.830
   4      1.00    0.689
   8      1.00    0.505
  16      1.00    0.230
  32      0.54    0.039
  64      0.44    0.001
```

The MLP column is constant because only the edges change with homophily. The GCN's large spreads at moderate homophily come from a few runs that train poorly with only 60 labels. At $K = 16$, accuracy is still 1.00 with spread 0.23, because a linear classifier needs directions, not distances; by $K = 64$ the directions have merged.

## Common mistakes

- **Shuffled CV on temporal data**, or preprocessing fitted to the whole series.
- **No seasonal naive baseline**, or a single test period instead of many origins.
- **Intervals never checked** for coverage by horizon.
- **Trusting zero-shot claims** without asking what the model was pretrained on.
- **Point-adjusted F1** without a random-score baseline under the same protocol.
- **A GCN on a heterophilous graph** without first measuring homophily and comparing with an MLP.
- **Deep plain GCN stacks**, which over-smooth; two or three layers usually suffice.
- **Test edges left in the graph** during link prediction.

## Exercises

1. [warm-up] For monthly data with yearly seasonality, write the seasonal naive forecast for $h = 1$ and $h = 13$.
2. [warm-up] In script 1, space origins 7 days apart. Predict what happens to naive's per-step MAE.
3. [core] Show that the SES forecast is $\sum_{j \ge 0} \alpha(1-\alpha)^j y_{T-j}$ (ignoring initialisation). What is $\alpha = 1$?
4. [core] In script 1 part 2, replace the time index by the values 1 and 7 days earlier and predict one step ahead. Does shuffled K-fold still mislead? Then predict the daily *change* from lagged changes. Relate both results to Bergmeir et al. (2018).
5. [core] In script 2, use intervals $\pm 1.28\,\hat\sigma_1\sqrt{k}$ at step $k$. Compare coverage with the simulated intervals.
6. [core] In script 3, implement PA%K (credit a segment only if at least $K\%$ of its points are flagged) for $K = 0, 20, 50, 100$.
7. [core] In script 4, find two small graphs with constant features that sum aggregation separates and mean does not.
8. [stretch] In script 5, measure each graph's edge homophily, then add a two-hop term to the ego+nbr model. Does it help at $h = 0.1$?
9. [stretch] Generate 300 related series (each with its own level, weekly shape and noise). Compare per-series seasonal AR models with one global ridge model fitted to 56-step windows pooled across series, each window scaled by its own mean and standard deviation. Then try an MLP in place of ridge.

### Hints and answers

1. $y_{T+1-12}$ and $y_{T+13-24}$: the same calendar month in the last observed year.
2. Every origin falls on the same weekday, so naive's per-step errors become a fixed weekday pattern with some steps far worse.
3. Unroll $\ell_T = \alpha y_T + (1-\alpha)\ell_{T-1}$. $\alpha = 1$ is the naive forecast.
4. With lagged levels, shuffled K-fold is still optimistic (in our run, MAE 1.34 against 2.24 for `TimeSeriesSplit` and 3.27 on the future), because a random walk keeps reaching levels the trees never saw. Modelling changes from lagged changes gives a stationary, autoregressive problem with nearly uncorrelated errors, and the three estimates agree (0.88, 0.87, 0.87), as Bergmeir et al. predict.
5. $\sqrt{k}$ assumes random-walk error accumulation; the real growth follows the model's seasonal and damped-trend states, so coverage is off at some horizons.
6. $K = 0$ is PA-F1. Random scores fall fastest as $K$ grows; the seasonal robust z-score holds up best.
7. A path of three nodes and a single edge, with all features 1. With mean aggregation and a mean readout, every node state is identical in both graphs; with sum aggregation the path's middle node receives 2 and the graphs separate.
8. The measured value should be near `h`. Two hops help under strong heterophily, since two steps away in a 3-class graph you often return to your own class.
9. Evaluate only on the last 14 points of each series, with training windows that end before them. Expect the global model to beat the local ones; whether the MLP beats ridge depends on how nonlinear your generator is. If the series are linear processes, ridge is hard to beat.

## Check your understanding

**1. Why does shuffled K-fold flatter a model that uses the time index?** Test points sit between training points, so the model interpolates, which trees do well. Forecasting needs extrapolation past the last index, where a tree's prediction is frozen at its edge value.

**2. What does MASE 0.9 mean, and why can seasonal naive score above 1?** The forecast MAE is 90% of the in-sample MAE of the one-season-back forecast. That denominator is an in-sample, one-season error; at longer horizons on new data, seasonal naive's errors are usually larger.

**3. A vendor says its foundation model beats statistical methods zero-shot. What do you ask?** Which benchmark, against which tuned baselines (ETS, ARIMA, a global boosted-tree model), with what uncertainty, and whether test series or correlated series from the same period were in pretraining. Then backtest on your own data after the model's training cutoff.

**4. Why can't message passing tell two triangles from a hexagon?** With identical features and degree 2 everywhere, every node receives the same messages at every round, so all states and any readout agree. This is the 1-WL limit [Xu et al., 2019].

**5. When is a GCN worse than an MLP?** When neighbours' labels are uninformative or misleading (low homophily), averaging mixes noise into a node's own signal. Keeping ego and neighbour terms separate avoids this.

## Further reading

- **Rob J Hyndman, George Athanasopoulos, "Forecasting: Principles and Practice", 3rd edition** (OTexts; free online, checked October 2026). https://otexts.com/fpp3/
  *Read for:* baselines, decomposition, ETS (ch. 8), ARIMA (ch. 9) and evaluation, with R code. *Level:* intro.
- **Benjamin Sanchez-Lengeling, Emily Reif, Adam Pearce, Alexander B. Wiltschko, "A Gentle Introduction to Graph Neural Networks"** (2021, Distill). https://distill.pub/2021/gnn-intro/
  *Read for:* an interactive visual introduction to graphs and message passing. *Level:* intro.
- **Spyros Makridakis, Evangelos Spiliotis, Vassilios Assimakopoulos, "The M4 Competition: 100,000 time series and 61 forecasting methods"** (2020, International Journal of Forecasting 36(1):54-74; may be paywalled). https://doi.org/10.1016/j.ijforecast.2019.04.014
  *Read for:* how a large forecasting competition is run and what won. *Level:* intermediate.
- **Spyros Makridakis, Evangelos Spiliotis, Vassilios Assimakopoulos, "M5 accuracy competition: Results, findings, and conclusions"** (2022, International Journal of Forecasting 38(4):1346-1364; may be paywalled). https://doi.org/10.1016/j.ijforecast.2021.11.013
  *Read for:* why global LightGBM models and combinations won, and how few teams beat the benchmarks. *Level:* intermediate.
- **Rob J. Hyndman, Anne B. Koehler, "Another look at measures of forecast accuracy"** (2006, International Journal of Forecasting 22(4):679-688; may be paywalled). https://doi.org/10.1016/j.ijforecast.2006.03.001
  *Read for:* why percentage errors mislead and the case for MASE. *Level:* intro.
- **Christoph Bergmeir, Rob J. Hyndman, Bonsoo Koo, "A note on the validity of cross-validation for evaluating autoregressive time series prediction"** (2018, Computational Statistics & Data Analysis 120:70-83). https://doi.org/10.1016/j.csda.2017.11.003
  *Read for:* exactly when K-fold CV is valid for time series. *Level:* advanced.
- **Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu, "Are Transformers Effective for Time Series Forecasting?"** (2023, AAAI; arXiv:2205.13504). https://arxiv.org/abs/2205.13504
  *Read for:* a linear model beating Transformer forecasters, and a lesson in baselines. *Level:* intermediate.
- **Abdul Fatir Ansari et al., "Chronos: Learning the Language of Time Series"** (2024, Transactions on Machine Learning Research; arXiv:2403.07815). https://arxiv.org/abs/2403.07815
  *Read for:* how a time-series foundation model is built and evaluated, in its authors' words. *Level:* advanced.
- **Oleksandr Shchur, Abdul Fatir Ansari, Caner Turkmen, Lorenzo Stella, Nick Erickson, Pablo Guerron, Michael Bohlke-Schneider, Yuyang Wang, "fev-bench: A Realistic Benchmark for Time Series Forecasting"** (2025, arXiv:2509.26468, v4 June 2026). https://arxiv.org/abs/2509.26468
  *Read for:* a current benchmark with covariates, intervals and leakage reporting; read its limitations too. *Level:* advanced.
- **Siwon Kim, Kukjin Choi, Hyun-Soo Choi, Byunghan Lee, Sungroh Yoon, "Towards a Rigorous Evaluation of Time-Series Anomaly Detection"** (2022, AAAI; arXiv:2109.05257). https://arxiv.org/abs/2109.05257
  *Read for:* why point adjustment inflates scores, and the PA%K alternative. *Level:* intermediate.
- **William L. Hamilton, "Graph Representation Learning"** (2020, Morgan & Claypool, Synthesis Lectures on AI and ML 14(3); free pre-publication PDF). https://www.cs.mcgill.ca/~wlh/grl_book/
  *Read for:* a compact, rigorous book on message passing and GNN theory. *Level:* intermediate.
- **Thomas N. Kipf, Max Welling, "Semi-Supervised Classification with Graph Convolutional Networks"** (2017, ICLR; arXiv:1609.02907). https://arxiv.org/abs/1609.02907
  *Read for:* the GCN layer and the semi-supervised setting of script 5. *Level:* intermediate.

**Other sources cited in the text** (checked October 2026): Das, Kong, Sen, Zhou, "A decoder-only foundation model for time-series forecasting" (ICML 2024), https://proceedings.mlr.press/v235/das24c.html · Tan, Merrill, Gupta, Althoff, Hartvigsen, "Are Language Models Actually Useful for Time Series Forecasting?" (NeurIPS 2024), https://arxiv.org/abs/2406.16964 · Aksu et al., "GIFT-Eval" (2024, arXiv:2410.10393), https://arxiv.org/abs/2410.10393, leaderboard notes at https://github.com/SalesforceAIResearch/gift-eval · Meyer, Kaltenpoth, Zalipski, Müller, "Rethinking Evaluation in the Era of Time Series Foundation Models" (2025, arXiv:2510.13654), https://arxiv.org/abs/2510.13654 · Karaouli et al., "How Foundational are Foundation Models for Time Series Forecasting?" (NeurIPS 2025 workshop), https://openreview.net/forum?id=EOwmP8FM92 · Wu, Keogh, "Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress" (IEEE TKDE 2023), https://arxiv.org/abs/2009.13807 · Gilmer et al., "Neural Message Passing for Quantum Chemistry" (ICML 2017), https://proceedings.mlr.press/v70/gilmer17a · Xu, Hu, Leskovec, Jegelka, "How Powerful are Graph Neural Networks?" (ICLR 2019), https://arxiv.org/abs/1810.00826 · Zhu et al., "Beyond Homophily in Graph Neural Networks" (NeurIPS 2020), https://arxiv.org/abs/2006.11468 · Li, Han, Wu, "Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning" (AAAI 2018), https://arxiv.org/abs/1801.07606 · Shchur, Mumme, Bojchevski, Günnemann, "Pitfalls of Graph Neural Network Evaluation" (2018), https://arxiv.org/abs/1811.05868 · Derrow-Pinion et al., "ETA Prediction with Graph Neural Networks in Google Maps" (CIKM 2021), https://arxiv.org/abs/2108.11482 · Lam et al., "Learning skillful medium-range global weather forecasting" (Science 2023), https://doi.org/10.1126/science.adi2336
