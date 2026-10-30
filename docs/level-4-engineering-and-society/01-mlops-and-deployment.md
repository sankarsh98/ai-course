# MLOps and Deployment

> **Level 4 · Module 1** · Prerequisites: L1-01, L1-05, L2-02 · Time: 6 to 8 hours · Code: `code/L4-01-mlops-and-deployment/`

## What you will be able to do

- Turn a notebook model into a pipeline whose data, code, environment and model are versioned, so any prediction can be traced to what produced it.
- Write data-validation checks and tests that run offline in continuous integration, and say what each kind of test can and cannot catch.
- Choose between batch, online and streaming serving for a problem, and estimate what each costs.
- Release a model safely with an offline gate, a canary compared against a concurrent control, and a rollback that is a one-line change.
- Monitor inputs, predictions and outcomes with PSI and the Kolmogorov-Smirnov test, and explain why their alarm thresholds mislead at small and large sample sizes.
- Decide, from monitoring signals, whether to do nothing, watch, block inputs, retrain or roll back.

## 1. From notebook to production

A notebook that reaches 0.74 AUC on a held-out split (L1-01) answers one question: can a model learn this from this data? Production asks different ones. Will tomorrow's data look like the training data? Will the serving code compute features the same way? When accuracy drops, will anyone notice, and can they undo the last change in minutes? Sculley et al. argued that ML systems accumulate **technical debt** that ordinary software practice does not see: entangled features, hidden feedback loops, data dependencies, and configuration that nobody owns [Sculley et al., 2015] (L1-05 introduced this paper and the related problem of train/serve skew).

Two empirical views of the work. Shankar et al. interviewed 18 ML engineers across chatbots, autonomous vehicles, finance and other areas and summarized what decides success as three V's: **velocity** (iterating fast), **validation** (catching errors early) and **versioning** (being able to go back) [Shankar et al., 2022]. Paleyes, Urma and Lawrence surveyed published deployment case studies and found practitioners reporting problems at every stage, from data management to monitoring [Paleyes et al., 2022]. **MLOps** is the name for the practices that address this; Kreuzberger et al. note that the term itself is used vaguely (Kreuzberger et al., 2022, arXiv:2205.02302). This module treats it concretely, as a set of things a pipeline must do, and builds a small offline version of each.

Experiment tracking for research (logging configs, metrics and seeds for every run) is in L4-06; reproducibility of a single training run (seeds, threads, deterministic kernels) is in L2-02. Here the focus is on what happens after you pick a model.

## 2. The pipeline is the product

Google Cloud's MLOps guide describes three levels of automation [Google Cloud, MLOps]. At **level 0**, a data scientist trains by hand and hands a model file to engineers; releases are rare and nobody monitors production performance. At **level 1**, the training pipeline itself is deployed and runs automatically (**continuous training**), with automated data and model validation. At **level 2**, a CI/CD system also tests and deploys changes to the pipeline's own components. The guide lists the triggers for retraining: on demand, on a schedule, when new data arrives, when performance degrades, and when the data distribution shifts significantly.

The level-1 insight matters most: what you ship is not a model but the procedure that makes models. `pipeline.py` has one function or class for each stage:

| stage | function | what it guards against |
|---|---|---|
| validate | `validate` | broken upstream data reaching training or serving |
| version | `fingerprint`, `Registry.register` | not knowing which data and code made a model |
| features | `featurize` | training and serving computing features differently |
| train | `train` | irreproducible models |
| gate | `promotion_gate` | shipping a model worse than the current one |
| monitor | `psi`, `drift_report` | silent degradation |
| decide | `decide` | inconsistent responses to alarms |

`run_pipeline.py` tells one story with them: train and register a first model, reject a broken batch, train a better model, catch a train/serve skew bug in its canary, roll it back, fix it, promote it, then watch it drift for eight weeks.

## 3. Data validation

Most production failures that reach users start as data problems. A **schema** states what each column must look like: type, range, allowed categories, maximum missing fraction. Breck et al. describe Google's data-validation system, part of the TFX platform: it checks each batch against a schema and also looks for training/serving skew; hundreds of product teams use it, and the authors list earlier error detection among its benefits [Breck et al., 2019].

`validate` returns a list of problems and the pipeline refuses to proceed if the list is not empty. In step 2 of `run_pipeline.py`, an upstream team starts reporting income in dollars instead of thousands and capitalizes a region name. Training would have happily used both: income would just become a much larger number, and "North" a fifth region. Validation stops it:

```text
   REJECTED: income_k: 100.0% outside [0, 500]
   REJECTED: region: unseen categories ['North']
```

Schema checks have a blind spot. A value can be valid and wrong. In step 4, a new online service returns 0 for every customer's login count. Zero is inside the allowed range, so the canary's invalid-request rate is 0%; only a distribution check (section 7) catches it. You need both kinds.

Where should a schema come from? Write it by hand for the few columns you understand, infer the rest from a trusted batch and review it, and treat a schema change like a code change: reviewed and versioned.

## 4. Versioning and reproducibility

To answer "why did the model predict this?", every model must point to exactly what made it.

**Data.** Data is too big for Git, so tools store a content hash in Git and the bytes elsewhere. DVC, for example, writes a small `.dvc` file holding the data's MD5 hash and path, keeps the data in a cache addressed by that hash, and pushes it to remote storage; checking out an old `.dvc` file and running `dvc checkout` restores the matching data [DVC, data versioning]. `fingerprint` does the core of this: a SHA-256 hash of a canonical CSV rendering, which a test checks is stable under column reordering and changes with the data. (As of October 2026, PyPI lists DVC 3.67.1, released March 2026.)

**Code and environment.** Record the Git commit of the pipeline, and the versions of the libraries that the artifact depends on. The registry stub stores scikit-learn and NumPy versions with each model, because a pickled model is tied to them: scikit-learn's documentation says there is no supported way to load a model trained with a different scikit-learn version [scikit-learn, Model persistence].

**Models.** A **model registry** stores immutable model versions with their metadata, and movable **aliases** that say which version plays which role. `Registry` keeps versions numbered and lets the aliases `champion` and `challenger` point at them; deploying, canarying and rolling back are all alias moves, logged in a history. This is the design MLflow adopted: its documentation says model stages have been deprecated since MLflow 2.9.0 in favor of aliases and tags [MLflow, Model Registry]. (PyPI lists MLflow 3.17.2, released October 21, 2026, as the current version as of October 2026.)

**Determinism.** The same data and seed should give the same model. `run_pipeline.py` trains v1 twice and checks that the coefficients are identical, and a test does the same. For neural networks this is harder (L2-02 shows thread count changing results); then the test compares metrics within a tolerance, not weights.

## 5. Packaging and serving patterns

**Packaging.** The artifact is the model plus everything needed to use it: feature code, library versions, and metadata. scikit-learn's guide recommends saving a reference to an immutable snapshot of the training data, the source code, the library versions and the cross-validation score alongside the model, and pinning dependencies or using containers [scikit-learn, Model persistence]. It also warns that pickle-based formats can execute arbitrary code when loaded and must only be loaded from trusted sources; ONNX and skops are safer alternatives with limitations. The registry stub checks a SHA-256 digest of the artifact bytes before unpickling, so a modified file is refused (a test appends one byte and expects failure). A digest guards against corruption and accidental edits, not against an attacker who can also rewrite the registry.

**One feature function.** `featurize` is the only place features are computed, for training and for serving. Zinkevich's rules, discussed in L1-05, recommend exactly this kind of code reuse against train/serve skew. Step 4 shows what happens when a feature is computed by a different code path at serving time.

**Serving patterns.**

| pattern | how | latency | freshness | typical risk |
|---|---|---|---|---|
| batch | score all entities on a schedule, store results in a table | none at read time | as old as the last run | stale predictions, wasted work on entities never queried |
| online | a service computes a prediction per request | milliseconds, on the request path | features as of the request | tail latency, availability, skew between offline and online features |
| streaming | a consumer scores events from a stream as they arrive | seconds | near real time | out-of-order and late events, state management |

Choose the simplest one that meets the freshness requirement. Churn scores used in a weekly email campaign should be batch; fraud scores for a card payment must be online; a feed reacting to a user's last few clicks may need streaming features with an online model.

**Cost.** Serving cost per prediction is roughly the price of the machines divided by the predictions they actually make. An online service must be sized for peak traffic plus headroom and runs all day; if its average utilization is 20%, each prediction costs five times what the same hardware would at full use. A nightly batch job runs at high utilization for an hour and stops. As an illustration with made-up round numbers: one machine at 1 unit per hour that can score 2,000 rows per second costs $1/(2{,}000 \times 3{,}600) \approx 1.4 \times 10^{-7}$ units per prediction in batch, and $1.4 \times 10^{-7}/0.2 = 6.9 \times 10^{-7}$ online at 20% utilization. Training cost, labeling, storage of versioned data, and the engineers who maintain all of it are usually larger than serving for small models; for large models L4-02 does the compute accounting.

## 6. Releasing safely: gates, canaries, rollback

**Offline gate.** Before anything reaches users, `promotion_gate` compares the candidate with the champion **on the same holdout**: it must clear an absolute floor and must not be worse than the champion by more than a tolerance. In step 3, v2 (with the new login feature) scores 0.737 against v1's 0.713 and passes. Use the uncertainty tools of L1-01 to set the tolerance; a difference inside the noise is not an improvement.

**Canary.** An offline gate cannot see serving bugs, so the candidate first serves a small slice of real traffic. Google's SRE workbook chapter on canarying recommends comparing the canary with a **control** group served at the same time, rather than comparing with the period before the release, because time-varying effects make before/after comparisons noisy; it also advises a canary large and long enough to be representative, only one canary at a time, and a small set of metrics that reflect what users see [Warner et al., SRE Workbook]. `serve_week` routes 10% of requests to the challenger by hashing the request id, so each request consistently gets the same arm.

Step 4 is the interesting failure. The canary's login feature comes from a new online service that returns 0 for everyone:

```text
   canary n=802: invalid requests 0%, max feature PSI 9.31, prediction PSI 2.26
   canary vs control: AUC 0.726 vs 0.710, log loss 0.706 vs 0.408, mean prediction 0.504 vs 0.169, observed churn 0.146 vs 0.165
   decision: ROLLBACK (canary log loss 0.706 > control 0.408 + 5%)
```

The canary's **AUC is higher** than the control's. AUC only measures ranking, and a feature that is constant for everyone shifts all scores by the same amount without reordering them, so the ranking from the other features survives. But the canary predicts a 50% churn rate when the truth is 15%: anything that uses the probabilities (a threshold, an expected cost, a budget) is badly wrong. Log loss, which punishes miscalibration (L1-01), catches it. The label-free signals catch it even earlier: the login feature's PSI is 9.31 and the prediction PSI 2.26 while the control's predictions have not moved, and `decide` treats exactly that pattern as grounds for rollback. Choose canary metrics that match how predictions are used.

**Rollback.** Because aliases point at immutable versions, rolling back is moving an alias: here, removing `challenger` so all traffic returns to v1. The ML Test Score paper stresses that being able to revert quickly to a known-good model matters as much as for any serving system, and that teams should practice rollbacks in normal conditions because in an emergency they must just work [Breck et al., 2017]. Step 5 fixes the service, reruns the canary (AUC 0.756 against 0.712, log loss 0.401 against 0.410, mean prediction matching observed churn) and promotes v2.

## 7. Monitoring and drift detection

### 7.1 What to monitor

Monitor four things: **inputs** (validation failures, feature distributions), **predictions** (their distribution, which needs no labels), **outcomes** (accuracy, calibration, business metrics, which need labels that often arrive days or months late), and the **system** (latency, errors, cost). The kinds of shift (covariate, label, concept) are defined in L1-05. Only outcome monitoring can detect concept shift, where $p(y \mid x)$ changes; input monitoring detects covariate shift but cannot say whether it hurts.

### 7.2 PSI

The **population stability index** compares a current distribution with a reference over $B$ bins. With reference proportions $e_i$ and current proportions $a_i$:

$$\text{PSI} = \sum_{i=1}^{B} (a_i - e_i)\ln\frac{a_i}{e_i} = \mathrm{KL}(a\,\|\,e) + \mathrm{KL}(e\,\|\,a),$$

the symmetrized KL divergence of L0-04, so it is zero only when the binned distributions agree, and it is symmetric (a test checks this). `psi` uses reference quantiles as bin edges for numbers and categories for strings, and clips empty bins to avoid $\ln 0$. Common rules of thumb read PSI below 0.1 as little shift and above 0.25 as significant shift needing action; Yurdakul's dissertation points out that these benchmarks are used without reference to Type I or Type II error rates and studies PSI's statistical properties [Yurdakul, 2018].

You can see the problem with a short calculation. For small differences, $\ln(a_i/e_i) \approx (a_i - e_i)/e_i$, so $\text{PSI} \approx \sum_i (a_i - e_i)^2/e_i$, a chi-square-like statistic. If nothing has changed and the two windows have $n$ and $m$ samples, each $a_i - e_i$ has variance about $p_i(1 - p_i)(1/n + 1/m)$, so

$$\mathbb{E}[\text{PSI}] \approx (B - 1)\left(\frac{1}{n} + \frac{1}{m}\right).$$

With 10 bins and 100 samples per window, that is 0.18, above the 0.1 alarm line with **no drift at all**. `drift.py` confirms it: at $n = 100$, PSI exceeds 0.1 in 84% of driftless repetitions and 0.25 in 22%, and the mean PSI matches the formula at every sample size. At large $n$ the opposite happens: a 0.5 standard deviation shift, which a model may well notice, gives PSI near 0.24 and crosses 0.25 in only 2% of repetitions at $n = 100{,}000$. A fixed threshold means different things at different window sizes.

### 7.3 The Kolmogorov-Smirnov test

The two-sample **KS test** compares empirical cumulative distribution functions; its statistic is their largest vertical gap, and SciPy's `ks_2samp` returns a p-value under the assumption of continuous distributions [SciPy, ks_2samp]. Under no drift it alarms at about its nominal 5% rate at every $n$ (2 to 4% in `drift.py`). But it answers "is there any difference?", and with enough data the answer is always yes: a shift of 0.05 standard deviations is flagged in 87% of repetitions at $n = 10{,}000$ and 100% at 100,000. In step 6 of `run_pipeline.py`, the age feature's p-value is $2.4 \times 10^{-10}$ in week 1, while its PSI is 0.017 and the model's AUC is unchanged. Statistical significance is not practical importance (L0-04).

Many features make it worse. With 20 independent driftless features checked at $p < 0.05$, `drift.py` raises at least one alarm in 66% of checks (theory: $1 - 0.95^{20} = 64\%$). Correct for multiple comparisons, or alarm on effect sizes.

**What works better.** Rabanser, Günnemann and Lipton compared shift detectors on image data and found that two-sample tests on the outputs of a pretrained classifier (a low-dimensional summary) performed best, and that domain classifiers, which learn to tell old data from new (L1-05), help characterize what shifted [Rabanser et al., 2019]. The practical lessons for tabular monitoring: monitor the **prediction distribution** first, because it summarizes every input in one dimension that the model cares about; use effect sizes (PSI, differences in means or quantiles) with thresholds calibrated to your window size; and confirm with outcomes when labels arrive.

### 7.4 From signals to actions

`decide` encodes the policy, in order: roll back a canary that is worse than its control or whose predictions alone have moved; block inputs if more than 5% fail validation; flag retraining if any PSI exceeds 0.25; watch above 0.1. The key rule: **drift alone never triggers a rollback**, because the previous model sees the same drifted data. Step 6 shows eight weeks in which age drifts upward and, from week 5, logins become less predictive (concept drift). Age PSI crosses 0.1 in week 3 and 0.25 in week 5; AUC falls from 0.758 to 0.717. The prediction PSI stays below 0.14, so the prediction distribution understates the damage done by concept drift, as it must: only labels reveal that $p(y \mid x)$ changed. Retraining on recent data recovers part of the loss (0.719 to 0.729 on a recent holdout), which then goes through the same gate and canary.

## 8. Continuous integration for ML

**CI** runs automated checks on every change. For ML there are three kinds of thing to test. **Code**: unit tests of feature functions, the registry and the decision logic. **Data**: the schema and distribution checks of sections 3 and 7, run on every batch. **Models**: the offline gate, plus behavioral checks such as a minimum score on known slices.

The ML Test Score paper lists 28 tests in four sections (data, model development, infrastructure, monitoring), awards half a point for running a test manually and a full point for automating it, and scores a system by the **minimum** of its four section totals, so that no area can be neglected [Breck et al., 2017]. Its monitoring section includes checks that data invariants hold for serving inputs, that training and serving are not skewed, and that models are not too stale. It is a useful audit for a real system.

`test_pipeline.py` has 18 tests that run in under a tenth of a second with no network: validation catches each corruption, fingerprints are stable, training is deterministic, the registry rolls back and detects a tampered artifact, PSI behaves as section 7 says, and `decide` returns the intended action in each case. Tests like these are cheap enough to run on every commit. They are only as good as their ability to fail, so break the code on purpose now and then and check that a test notices (exercise 7).

## Hands-on

Run everything from `code/L4-01-mlops-and-deployment/` with the course environment. Nothing touches the network; `run_pipeline.py` creates and deletes a `registry_demo/` folder. Results are deterministic, except that PSI and KS values can differ in the last digits across library versions.

### The library, `pipeline.py`

```python
"""A small, offline ML pipeline: data validation, versioning, reproducible
training, a model registry stub, drift checks and a monitoring decision.

Everything runs locally with numpy, pandas, scipy and scikit-learn. Real
systems use tools such as MLflow, DVC or TFX for these jobs; the ideas are
the same, and the code is short enough to read in one sitting.
"""
import hashlib
import io
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- data
REGIONS = ["north", "south", "east", "west"]
SCHEMA = {   # what training data and serving requests must look like
    "age": dict(kind="num", min=18, max=100, max_missing=0.0),
    "income_k": dict(kind="num", min=0, max=500, max_missing=0.02),   # thousands
    "tenure_months": dict(kind="num", min=0, max=600, max_missing=0.0),
    "logins_30d": dict(kind="num", min=0, max=200, max_missing=0.0),
    "region": dict(kind="cat", values=REGIONS, max_missing=0.0),
}
TARGET = "churned"


def make_data(n, seed, age_shift=0.0, concept=0.0):
    """Synthetic customer table. age_shift moves p(x); concept changes p(y|x)."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(45 + age_shift, 12, n), 18, 100)
    income = np.clip(rng.lognormal(3.8, 0.5, n), 0, 500)
    tenure = np.clip(rng.exponential(36, n), 0, 600)
    logins = rng.poisson(np.clip(12 - 0.1 * (age - 45), 1, None))
    region = rng.choice(REGIONS, n, p=[0.4, 0.3, 0.2, 0.1])
    z = (-0.5 - 0.03 * tenure + 0.02 * (age - 45) - 0.15 * (logins - 10)
         + 0.6 * (region == "west") - 0.004 * income)
    z = z + concept * 0.08 * (logins - 10)       # concept drift: logins matter less
    y = rng.random(n) < 1 / (1 + np.exp(-z))
    df = pd.DataFrame(dict(age=age, income_k=income, tenure_months=tenure,
                           logins_30d=logins.astype(float), region=region))
    df.loc[rng.random(n) < 0.01, "income_k"] = np.nan   # 1% missing, allowed
    df[TARGET] = y.astype(int)
    return df


def validate(df, schema=SCHEMA, min_rows=100):
    """Return a list of problems. An empty list means the batch may proceed."""
    problems = []
    if len(df) < min_rows:
        problems.append(f"only {len(df)} rows (< {min_rows})")
    for col, rule in schema.items():
        if col not in df:
            problems.append(f"missing column {col!r}")
            continue
        s = df[col]
        miss = s.isna().mean()
        if miss > rule["max_missing"]:
            problems.append(f"{col}: {miss:.1%} missing (max {rule['max_missing']:.0%})")
        s = s.dropna()
        if rule["kind"] == "num":
            if not pd.api.types.is_numeric_dtype(s):
                problems.append(f"{col}: expected numbers, got {s.dtype}")
                continue
            bad = ((s < rule["min"]) | (s > rule["max"])).mean()
            if bad > 0:
                problems.append(f"{col}: {bad:.1%} outside [{rule['min']}, {rule['max']}]")
        else:
            unseen = sorted(set(s) - set(rule["values"]))
            if unseen:
                problems.append(f"{col}: unseen categories {unseen[:5]}")
    return problems


def fingerprint(df):
    """Content hash of a table: the same idea as DVC's content-addressed files."""
    buf = io.StringIO()
    df.sort_index(axis=1).to_csv(buf, index=False, float_format="%.10g")
    return hashlib.sha256(buf.getvalue().encode()).hexdigest()[:12]


# ---------------------------------------------------------------- features + model
def featurize(df, columns):
    """The ONE feature function, used by training and serving alike."""
    out = pd.DataFrame(index=df.index)
    for c in columns:
        if c == "region":
            for r in REGIONS:
                out[f"region_{r}"] = (df["region"] == r).astype(float)
        elif c == "income_k":
            out["log_income"] = np.log1p(df["income_k"].fillna(df["income_k"].median()))
        else:
            out[c] = df[c].astype(float)
    return out.to_numpy()


def train(df, columns, seed=0, C=1.0):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, random_state=seed, max_iter=1000))
    model.fit(featurize(df, columns), df[TARGET])
    return model


def auc(model, df, columns):
    return roc_auc_score(df[TARGET], model.predict_proba(featurize(df, columns))[:, 1])


def digest(blob):
    return hashlib.sha256(blob).hexdigest()[:12]


# ---------------------------------------------------------------- registry stub
class Registry:
    """Versions are immutable; aliases ('champion', 'challenger') are movable
    pointers, the same design MLflow uses since it deprecated stages."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "registry.json"
        if not self.index.exists():
            self._save(dict(versions=[], aliases={}, history=[]))

    def _load(self):
        return json.loads(self.index.read_text())

    def _save(self, reg):
        self.index.write_text(json.dumps(reg, indent=1))

    def register(self, model, columns, data_fp, metrics, params):
        reg = self._load()
        v = len(reg["versions"]) + 1
        path = self.root / f"model_v{v}.pkl"
        blob = pickle.dumps(model)   # pickles are tied to library versions: record them
        path.write_bytes(blob)
        reg["versions"].append(dict(version=v, file=path.name, digest=digest(blob),
                                    columns=columns, data=data_fp, metrics=metrics,
                                    params=params, sklearn=sklearn.__version__,
                                    numpy=np.__version__))
        self._save(reg)
        return v

    def set_alias(self, alias, version, reason):
        reg = self._load()
        reg["history"].append(dict(alias=alias, old=reg["aliases"].get(alias),
                                   new=version, reason=reason))
        reg["aliases"][alias] = version
        self._save(reg)

    def remove_alias(self, alias, reason):
        reg = self._load()
        reg["history"].append(dict(alias=alias, old=reg["aliases"].pop(alias, None),
                                   new=None, reason=reason))
        self._save(reg)

    def get(self, alias):
        reg = self._load()
        v = reg["aliases"].get(alias)
        if v is None:
            return None
        meta = reg["versions"][v - 1]
        blob = (self.root / meta["file"]).read_bytes()
        assert digest(blob) == meta["digest"], "artifact changed on disk"
        return pickle.loads(blob), meta

    def history(self):
        return self._load()["history"]


def promotion_gate(cand_auc, champ_auc, min_auc=0.70, tolerance=0.005):
    """Offline gate: good enough in absolute terms and not worse than the champion."""
    if cand_auc < min_auc:
        return False, f"AUC {cand_auc:.3f} below floor {min_auc}"
    if champ_auc is not None and cand_auc < champ_auc - tolerance:
        return False, f"AUC {cand_auc:.3f} worse than champion {champ_auc:.3f}"
    return True, "passed"


# ---------------------------------------------------------------- drift
def psi(reference, current, bins=10, eps=1e-4):
    """Population stability index: sum over bins of (a - e) * ln(a / e), with
    bins from the reference quantiles (numbers) or the categories (strings)."""
    reference, current = pd.Series(reference).dropna(), pd.Series(current).dropna()
    if pd.api.types.is_numeric_dtype(reference):
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(reference, edges)[0] / len(reference)
        a = np.histogram(current, edges)[0] / len(current)
    else:
        cats = sorted(set(reference) | set(current))
        e = reference.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        a = current.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)   # empty bins would give log(0)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(ref, cur, columns):
    rows = []
    for c in columns:
        ks_p = (stats.ks_2samp(ref[c].dropna(), cur[c].dropna()).pvalue
                if SCHEMA[c]["kind"] == "num" else float("nan"))
        rows.append(dict(feature=c, psi=psi(ref[c], cur[c]), ks_p=ks_p))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- monitoring decision
def decide(invalid_rate, max_feature_psi, pred_psi, canary=None, rel_margin=0.05):
    """Turn monitoring signals into one action. A bad release is rolled back
    before anything else. Drift alone never triggers a rollback, because the
    previous model sees the same drifted data; it triggers retraining.

    canary: for a new version serving a slice of traffic, a dict with its log
    loss, the control's log loss, and the control's prediction PSI.
    """
    if canary is not None:
        if canary["loss"] > canary["control_loss"] * (1 + rel_margin):
            return "ROLLBACK", (f"canary log loss {canary['loss']:.3f} > control "
                                f"{canary['control_loss']:.3f} + {rel_margin:.0%}")
        if pred_psi > 0.25 and canary["control_pred_psi"] < 0.10:
            return "ROLLBACK", f"only the canary's predictions moved (PSI {pred_psi:.2f})"
    if invalid_rate > 0.05:
        return "BLOCK_INPUTS", f"{invalid_rate:.1%} of requests fail validation"
    worst = max(max_feature_psi, pred_psi)
    if worst > 0.25:
        return "ALERT_RETRAIN", f"PSI {worst:.2f} > 0.25"
    if worst > 0.10:
        return "WATCH", f"PSI {worst:.2f} > 0.10"
    return "OK", "no signal"
```

### The story, `run_pipeline.py`

```python
"""End-to-end story with the pieces in pipeline.py, all offline and synthetic.

1. Train v1, register it, promote it to champion.
2. A broken upstream batch is stopped by validation.
3. Train v2 with a new feature; it passes the offline gate and goes to canary.
4. The canary's online feature is computed differently from training
   (train/serve skew); monitoring catches it and the canary is rolled back.
5. With the feature service fixed, v2 is promoted.
6. Eight weeks of production: drift appears, and the response is retraining.
"""
import shutil
import zlib

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from pipeline import (Registry, auc, decide, drift_report, featurize, fingerprint,
                      make_data, promotion_gate, psi, train, validate)

ROOT = "registry_demo"
shutil.rmtree(ROOT, ignore_errors=True)
reg = Registry(ROOT)
BASE = ["age", "income_k", "tenure_months", "region"]
WITH_LOGINS = BASE + ["logins_30d"]

print("1. train and register v1")
data = make_data(6000, seed=1)
train_df, hold = data.iloc[:5000], data.iloc[5000:]
print("   validation problems:", validate(train_df) or "none")
fp = fingerprint(train_df)
m1, m1_again = train(train_df, BASE, seed=0), train(train_df, BASE, seed=0)
same = np.array_equal(m1[-1].coef_, m1_again[-1].coef_)
print(f"   data fingerprint {fp}; two training runs give identical coefficients: {same}")
v1 = reg.register(m1, BASE, fp, dict(auc=auc(m1, hold, BASE)), dict(C=1.0, seed=0))
ok, why = promotion_gate(auc(m1, hold, BASE), None)
print(f"   v{v1}: holdout AUC {auc(m1, hold, BASE):.3f}; gate: {why}")
reg.set_alias("champion", v1, "first model")

print("\n2. a new batch arrives from upstream")
batch = make_data(5000, seed=2)
broken = batch.copy()
broken["income_k"] = broken["income_k"] * 1000          # upstream switched to dollars
broken.loc[broken["region"] == "north", "region"] = "North"
for p in validate(broken):
    print("   REJECTED:", p)

print("\n3. the corrected batch; train v2 with a new feature, logins_30d")
new = pd.concat([train_df, batch])
print("   validation problems:", validate(new) or "none")
m2 = train(new, WITH_LOGINS)
champ, meta = reg.get("champion")
a2, a1 = auc(m2, hold, WITH_LOGINS), auc(champ, hold, meta["columns"])
ok, why = promotion_gate(a2, a1)
v2 = reg.register(m2, WITH_LOGINS, fingerprint(new), dict(auc=a2), dict(C=1.0, seed=0))
print(f"   same holdout: champion v1 AUC {a1:.3f}, candidate v{v2} AUC {a2:.3f}; gate: {why}")
reg.set_alias("challenger", v2, "passed offline gate; 10% canary")


def serve_week(requests, canary_share, logins_service_ok):
    """Route by a hash of the request id; return predictions per arm."""
    arm = np.array([zlib.crc32(str(i).encode()) % 100 < canary_share * 100
                    for i in requests.index])
    out = {}
    for name, alias, mask in [("control", "champion", ~arm), ("canary", "challenger", arm)]:
        model, meta = reg.get(alias)
        req = requests[mask].copy()
        if name == "canary" and not logins_service_ok:
            req["logins_30d"] = 0.0        # online service returns a default, not the count
        bad = sum(bool(validate(req.iloc[[i]], min_rows=1)) for i in range(min(len(req), 200)))
        out[name] = dict(df=req, p=model.predict_proba(featurize(req, meta["columns"]))[:, 1],
                         invalid=bad / min(len(req), 200), columns=meta["columns"])
    return out


def canary_check(week):
    m_chal, meta = reg.get("challenger")
    m_champ, meta_c = reg.get("champion")
    p_chal = m_chal.predict_proba(featurize(new, meta["columns"]))[:, 1]     # at training time
    p_champ = m_champ.predict_proba(featurize(train_df, meta_c["columns"]))[:, 1]
    c, k = week["control"], week["canary"]
    feat_psi = drift_report(new, k["df"], [x for x in meta["columns"] if x != "region"])["psi"].max()
    pred_psi = psi(p_chal, k["p"])
    # Labels arrive later in real life; here we wait for them.
    ca, co = roc_auc_score(k["df"]["churned"], k["p"]), roc_auc_score(c["df"]["churned"], c["p"])
    cl, col = log_loss(k["df"]["churned"], k["p"]), log_loss(c["df"]["churned"], c["p"])
    action, why = decide(k["invalid"], feat_psi, pred_psi,
                         canary=dict(loss=cl, control_loss=col, control_pred_psi=psi(p_champ, c["p"])))
    print(f"   canary n={len(k['df'])}: invalid requests {k['invalid']:.0%}, max feature PSI"
          f" {feat_psi:.2f}, prediction PSI {pred_psi:.2f}")
    print(f"   canary vs control: AUC {ca:.3f} vs {co:.3f}, log loss {cl:.3f} vs {col:.3f},"
          f" mean prediction {k['p'].mean():.3f} vs {c['p'].mean():.3f},"
          f" observed churn {k['df']['churned'].mean():.3f} vs {c['df']['churned'].mean():.3f}")
    print(f"   decision: {action} ({why})")
    return action


print("\n4. one week of traffic, 10% to the canary")
week = serve_week(make_data(8000, seed=3), 0.10, logins_service_ok=False)
if canary_check(week) == "ROLLBACK":
    reg.remove_alias("challenger", "canary rolled back")
    print("   challenger alias removed; all traffic back on v1")

print("\n5. feature service fixed; canary again")
reg.set_alias("challenger", v2, "retry after fixing logins service")
week = serve_week(make_data(8000, seed=4), 0.10, logins_service_ok=True)
if canary_check(week) == "OK":
    reg.set_alias("champion", v2, "canary healthy")
    reg.remove_alias("challenger", "promoted")
    print(f"   v{v2} promoted to champion")

print("\n6. eight weeks in production (age drifts upward; from week 5 logins matter less)")
model, meta = reg.get("champion")
ref = new
p_ref = model.predict_proba(featurize(ref, meta["columns"]))[:, 1]
print(f"   {'week':>4} {'age PSI':>8} {'age KS p':>9} {'pred PSI':>9} {'AUC*':>6}  decision")
for w in range(1, 9):
    cur = make_data(5000, seed=100 + w, age_shift=1.5 * w, concept=1.0 if w >= 5 else 0.0)
    rep = drift_report(ref, cur, meta["columns"]).set_index("feature")
    p = model.predict_proba(featurize(cur, meta["columns"]))[:, 1]
    pred_psi = psi(p_ref, p)
    action, _ = decide(0.0, rep["psi"].max(), pred_psi)
    print(f"   {w:>4} {rep.loc['age', 'psi']:>8.3f} {rep.loc['age', 'ks_p']:>9.1e}"
          f" {pred_psi:>9.3f} {auc(model, cur, meta['columns']):>6.3f}  {action}")
print("   * AUC needs labels, which in practice arrive late; the other columns do not")

recent = make_data(6000, seed=200, age_shift=12, concept=1.0)
m3 = train(recent.iloc[:5000], WITH_LOGINS)
test = recent.iloc[5000:]
print(f"\n   retrain on recent data: v2 AUC {auc(model, test, WITH_LOGINS):.3f},"
      f" retrained AUC {auc(m3, test, WITH_LOGINS):.3f} on the same recent holdout")
print("\nregistry history:")
for h in reg.history():
    print(f"   {h['alias']:<10} {str(h['old']):>4} -> {str(h['new']):<4} {h['reason']}")
shutil.rmtree(ROOT)
```

```text
1. train and register v1
   validation problems: none
   data fingerprint 5c1ab821fddf; two training runs give identical coefficients: True
   v1: holdout AUC 0.713; gate: passed

2. a new batch arrives from upstream
   REJECTED: income_k: 100.0% outside [0, 500]
   REJECTED: region: unseen categories ['North']

3. the corrected batch; train v2 with a new feature, logins_30d
   validation problems: none
   same holdout: champion v1 AUC 0.713, candidate v2 AUC 0.737; gate: passed

4. one week of traffic, 10% to the canary
   canary n=802: invalid requests 0%, max feature PSI 9.31, prediction PSI 2.26
   canary vs control: AUC 0.726 vs 0.710, log loss 0.706 vs 0.408, mean prediction 0.504 vs 0.169, observed churn 0.146 vs 0.165
   decision: ROLLBACK (canary log loss 0.706 > control 0.408 + 5%)
   challenger alias removed; all traffic back on v1

5. feature service fixed; canary again
   canary n=802: invalid requests 0%, max feature PSI 0.02, prediction PSI 0.04
   canary vs control: AUC 0.756 vs 0.712, log loss 0.401 vs 0.410, mean prediction 0.163 vs 0.173, observed churn 0.173 vs 0.167
   decision: OK (no signal)
   v2 promoted to champion

6. eight weeks in production (age drifts upward; from week 5 logins matter less)
   week  age PSI  age KS p  pred PSI   AUC*  decision
      1    0.017   2.4e-10     0.002  0.758  OK
      2    0.060   4.8e-27     0.009  0.746  OK
      3    0.149   2.1e-73     0.022  0.743  WATCH
      4    0.228  3.1e-118     0.030  0.742  WATCH
      5    0.397  6.5e-194     0.056  0.729  ALERT_RETRAIN
      6    0.550  2.1e-260     0.076  0.723  ALERT_RETRAIN
      7    0.736  1.3e-320     0.111  0.714  ALERT_RETRAIN
      8    0.951  1.1e-320     0.133  0.717  ALERT_RETRAIN
   * AUC needs labels, which in practice arrive late; the other columns do not

   retrain on recent data: v2 AUC 0.719, retrained AUC 0.729 on the same recent holdout

registry history:
   champion   None -> 1    first model
   challenger None -> 2    passed offline gate; 10% canary
   challenger    2 -> None canary rolled back
   challenger None -> 2    retry after fixing logins service
   champion      1 -> 2    canary healthy
   challenger    2 -> None promoted
```

### Drift statistics, `drift.py`

```python
"""How PSI and the two-sample KS test behave, on synthetic data.

1. With no drift at all: how often does each rule raise an alarm, by sample size?
2. With a tiny shift (0.05 standard deviations) and a large one (0.5):
   which rule notices, and does noticing mean it matters?
3. Many features: the chance of at least one false alarm per check.
Each cell is the fraction of 200 repetitions that alarm.
"""
import numpy as np
from scipy import stats

from pipeline import psi

rng = np.random.default_rng(0)
REPS, BINS = 200, 10


def alarm_rates(n, shift):
    psi_vals, ks_hits = [], 0
    for _ in range(REPS):
        ref = rng.normal(0, 1, n)
        cur = rng.normal(shift, 1, n)
        psi_vals.append(psi(ref, cur, bins=BINS))
        ks_hits += stats.ks_2samp(ref, cur).pvalue < 0.05
    v = np.array(psi_vals)
    return v.mean(), (v > 0.1).mean(), (v > 0.25).mean(), ks_hits / REPS


print("reference and current windows both of size n; alarm rate over 200 repetitions")
for shift in (0.0, 0.05, 0.5):
    print(f"\nshift = {shift} standard deviations")
    print(f"  {'n':>7} {'mean PSI':>9} {'PSI>0.10':>9} {'PSI>0.25':>9} {'KS p<0.05':>10}"
          f" {'(B-1)(2/n)':>11}")
    for n in (100, 1_000, 10_000, 100_000):
        m, a10, a25, ks = alarm_rates(n, shift)
        print(f"  {n:>7} {m:>9.4f} {a10:>9.0%} {a25:>9.0%} {ks:>10.0%} {(BINS - 1) * 2 / n:>11.4f}")

print("\n20 independent features, no drift, n=5000, one check")
hits = 0
for _ in range(REPS):
    p = [stats.ks_2samp(rng.normal(size=5000), rng.normal(size=5000)).pvalue for _ in range(20)]
    hits += min(p) < 0.05
print(f"  at least one KS p<0.05: {hits / REPS:.0%}   (theory 1 - 0.95^20 = {1 - 0.95 ** 20:.0%});"
      f" Bonferroni threshold 0.05/20 would give about 5%")
```

```text
reference and current windows both of size n; alarm rate over 200 repetitions

shift = 0.0 standard deviations
        n  mean PSI  PSI>0.10  PSI>0.25  KS p<0.05  (B-1)(2/n)
      100    0.1959       84%       22%         2%      0.1800
     1000    0.0180        0%        0%         4%      0.0180
    10000    0.0019        0%        0%         4%      0.0018
   100000    0.0002        0%        0%         4%      0.0002

shift = 0.05 standard deviations
        n  mean PSI  PSI>0.10  PSI>0.25  KS p<0.05  (B-1)(2/n)
      100    0.1965       86%       25%         6%      0.1800
     1000    0.0194        0%        0%        18%      0.0180
    10000    0.0042        0%        0%        87%      0.0018
   100000    0.0026        0%        0%       100%      0.0002

shift = 0.5 standard deviations
        n  mean PSI  PSI>0.10  PSI>0.25  KS p<0.05  (B-1)(2/n)
      100    0.4847      100%       92%        89%      0.1800
     1000    0.2596      100%       54%       100%      0.0180
    10000    0.2383      100%       22%       100%      0.0018
   100000    0.2384      100%        2%       100%      0.0002

20 independent features, no drift, n=5000, one check
  at least one KS p<0.05: 66%   (theory 1 - 0.95^20 = 64%); Bonferroni threshold 0.05/20 would give about 5%
```

### Tests, `test_pipeline.py`

Run with `python -m unittest -v test_pipeline`.

```python
"""Tests for pipeline.py. Run: python -m unittest -v test_pipeline

These are the kinds of checks a CI job runs on every change: they are fast,
need no network, and fail loudly if a refactor changes behavior.
"""
import tempfile
import unittest

import numpy as np

from pipeline import (Registry, auc, decide, featurize, fingerprint, make_data,
                      promotion_gate, psi, train, validate)

COLS = ["age", "income_k", "tenure_months", "logins_30d", "region"]


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.df = make_data(500, seed=0)

    def test_clean_data_passes(self):
        self.assertEqual(validate(self.df), [])

    def test_missing_column(self):
        self.assertIn("missing column 'age'", validate(self.df.drop(columns="age")))

    def test_unit_change_is_caught(self):
        bad = self.df.assign(income_k=self.df["income_k"] * 1000)
        self.assertTrue(any("income_k" in p and "outside" in p for p in validate(bad)))

    def test_unseen_category(self):
        bad = self.df.copy()
        bad.loc[0, "region"] = "Mars"
        self.assertTrue(any("unseen" in p for p in validate(bad)))

    def test_too_many_missing(self):
        bad = self.df.copy()
        bad.loc[:50, "income_k"] = np.nan          # about 10% missing, limit is 2%
        self.assertTrue(any("missing" in p for p in validate(bad)))

    def test_wrong_type(self):
        bad = self.df.assign(age=self.df["age"].astype(str))
        self.assertTrue(any("expected numbers" in p for p in validate(bad)))


class TestReproducibility(unittest.TestCase):
    def test_same_data_same_fingerprint(self):
        self.assertEqual(fingerprint(make_data(300, 1)), fingerprint(make_data(300, 1)))
        self.assertNotEqual(fingerprint(make_data(300, 1)), fingerprint(make_data(300, 2)))

    def test_fingerprint_ignores_column_order(self):
        df = make_data(300, 1)
        self.assertEqual(fingerprint(df), fingerprint(df[df.columns[::-1]]))

    def test_training_is_deterministic(self):
        df = make_data(2000, 3)
        a, b = train(df, COLS), train(df, COLS)
        np.testing.assert_array_equal(a[-1].coef_, b[-1].coef_)

    def test_model_beats_chance(self):
        df = make_data(4000, 4)
        self.assertGreater(auc(train(df.iloc[:3000], COLS), df.iloc[3000:], COLS), 0.65)

    def test_featurize_one_hot_is_complete(self):
        X = featurize(make_data(200, 5), ["region"])
        np.testing.assert_array_equal(X.sum(axis=1), np.ones(200))


class TestRegistry(unittest.TestCase):
    def test_promote_and_rollback(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            df = make_data(1000, 6)
            v1 = reg.register(train(df, COLS), COLS, fingerprint(df), {}, {})
            v2 = reg.register(train(df, COLS[:2]), COLS[:2], fingerprint(df), {}, {})
            reg.set_alias("champion", v1, "first")
            reg.set_alias("champion", v2, "better")
            self.assertEqual(reg.get("champion")[1]["version"], 2)
            previous = reg.history()[-1]["old"]          # rollback = move the alias back
            reg.set_alias("champion", previous, "rollback")
            self.assertEqual(reg.get("champion")[1]["version"], 1)
            self.assertIsNone(reg.get("challenger"))

    def test_tampered_artifact_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            df = make_data(500, 7)
            v = reg.register(train(df, COLS), COLS, fingerprint(df), {}, {})
            reg.set_alias("champion", v, "first")
            with open(f"{d}/model_v1.pkl", "ab") as f:
                f.write(b"x")
            with self.assertRaises(AssertionError):
                reg.get("champion")


class TestDriftAndDecisions(unittest.TestCase):
    def test_psi_near_zero_without_drift(self):
        rng = np.random.default_rng(0)
        self.assertLess(psi(rng.normal(size=20000), rng.normal(size=20000)), 0.01)

    def test_psi_large_with_drift(self):
        rng = np.random.default_rng(0)
        self.assertGreater(psi(rng.normal(size=20000), rng.normal(1.0, 1, 20000)), 0.25)

    def test_psi_categorical_and_symmetric(self):
        a = np.array(["x"] * 70 + ["y"] * 30)
        b = np.array(["x"] * 40 + ["y"] * 60)
        self.assertAlmostEqual(psi(a, b), psi(b, a))
        self.assertGreater(psi(a, b), 0.25)

    def test_gate(self):
        self.assertFalse(promotion_gate(0.65, None)[0])
        self.assertFalse(promotion_gate(0.75, 0.80)[0])
        self.assertTrue(promotion_gate(0.798, 0.80)[0])

    def test_decisions(self):
        healthy = dict(loss=0.40, control_loss=0.41, control_pred_psi=0.01)
        self.assertEqual(decide(0.0, 0.01, 0.01, canary=healthy)[0], "OK")
        worse = dict(loss=0.50, control_loss=0.41, control_pred_psi=0.01)
        self.assertEqual(decide(0.0, 0.01, 0.01, canary=worse)[0], "ROLLBACK")
        shifted = dict(loss=0.41, control_loss=0.41, control_pred_psi=0.01)
        self.assertEqual(decide(0.0, 0.01, 0.9, canary=shifted)[0], "ROLLBACK")
        self.assertEqual(decide(0.2, 0.01, 0.01)[0], "BLOCK_INPUTS")
        self.assertEqual(decide(0.0, 0.40, 0.01)[0], "ALERT_RETRAIN")   # drift: retrain, not rollback
        self.assertEqual(decide(0.0, 0.15, 0.01)[0], "WATCH")


if __name__ == "__main__":
    unittest.main()
```

```text
test_decisions (test_pipeline.TestDriftAndDecisions.test_decisions) ... ok
test_gate (test_pipeline.TestDriftAndDecisions.test_gate) ... ok
test_psi_categorical_and_symmetric (test_pipeline.TestDriftAndDecisions.test_psi_categorical_and_symmetric) ... ok
test_psi_large_with_drift (test_pipeline.TestDriftAndDecisions.test_psi_large_with_drift) ... ok
test_psi_near_zero_without_drift (test_pipeline.TestDriftAndDecisions.test_psi_near_zero_without_drift) ... ok
test_promote_and_rollback (test_pipeline.TestRegistry.test_promote_and_rollback) ... ok
test_tampered_artifact_is_detected (test_pipeline.TestRegistry.test_tampered_artifact_is_detected) ... ok
test_featurize_one_hot_is_complete (test_pipeline.TestReproducibility.test_featurize_one_hot_is_complete) ... ok
test_fingerprint_ignores_column_order (test_pipeline.TestReproducibility.test_fingerprint_ignores_column_order) ... ok
test_model_beats_chance (test_pipeline.TestReproducibility.test_model_beats_chance) ... ok
test_same_data_same_fingerprint (test_pipeline.TestReproducibility.test_same_data_same_fingerprint) ... ok
test_training_is_deterministic (test_pipeline.TestReproducibility.test_training_is_deterministic) ... ok
test_clean_data_passes (test_pipeline.TestValidation.test_clean_data_passes) ... ok
test_missing_column (test_pipeline.TestValidation.test_missing_column) ... ok
test_too_many_missing (test_pipeline.TestValidation.test_too_many_missing) ... ok
test_unit_change_is_caught (test_pipeline.TestValidation.test_unit_change_is_caught) ... ok
test_unseen_category (test_pipeline.TestValidation.test_unseen_category) ... ok
test_wrong_type (test_pipeline.TestValidation.test_wrong_type) ... ok

----------------------------------------------------------------------
Ran 18 tests in 0.084s

OK
```

## Common mistakes

- **Validating schema only.** Valid values can be wrong (a default of 0 for every login count). Add distribution checks against training data.
- **Comparing a canary with last week.** Compare with a concurrent control, on metrics that match how predictions are used; AUC missed a 3-fold calibration error in step 4.
- **Rolling back for drift.** The old model sees the same new data. Drift calls for investigation and retraining.
- **One PSI threshold for every window size.** At 100 samples per window, 0.1 is noise; at 100,000, a meaningful shift can stay under 0.25.
- **Treating a tiny KS p-value as an emergency.** At large $n$ everything is significant; look at effect sizes and model metrics.
- **Ignoring multiple comparisons** across features and time windows.
- **Unversioned data.** "The model trained on the table as of some Tuesday" cannot be reproduced; hash it or snapshot it.
- **Loading pickles across library versions or from untrusted sources.** Pin versions, store them with the model, and only load trusted artifacts.
- **Separate feature code for training and serving.** Share one function, or log served features and train on the logs.
- **Rollback procedures never rehearsed.** The first rollback should not happen during an incident.

## Exercises

1. [warm-up] Add a rule to `SCHEMA` that `logins_30d` must be an integer-valued number, and a test that fails without it.
2. [warm-up] Compute by hand the PSI between category proportions (0.7, 0.3) and (0.4, 0.6). Check with `psi`.
3. [warm-up] Using the formula in 7.2, what window size makes the expected driftless PSI equal to 0.02 with 10 bins and equal windows?
4. [core] Make the canary's bug subtler: return the login count of the previous month instead of 0 (simulate by adding noise with standard deviation 3). Which signals in `canary_check` still catch it?
5. [core] Replace the fixed PSI thresholds in `decide` with thresholds calibrated by simulation: alarm when PSI exceeds the 99th percentile of driftless PSI for the current window sizes.
6. [core] Add label delay to step 6: AUC for week $w$ becomes available only in week $w + 2$. Rewrite the decision so it uses the latest available AUC and the current PSI.
7. [core] Mutation testing by hand: introduce three bugs in `pipeline.py` (for example, drop the `np.clip` in `psi`, swap the alias update order, or remove the region check). Does a test fail each time? Add tests where none does.
8. [core] Add a domain classifier check (L1-05) to `drift_report`: train a classifier to tell reference rows from current rows and report its cross-validated AUC. Compare it with the per-feature PSI in weeks 1 to 8.
9. [stretch] Implement shadow deployment: the challenger scores every request but its predictions are only logged. What can shadowing catch that a canary cannot, and the reverse?
10. [stretch] Write a minimal CI workflow (for example, a shell script or a configuration for the CI service you use) that installs pinned dependencies, runs the tests, runs `run_pipeline.py`, and fails if the final champion's holdout AUC is below 0.70.

### Hints and answers

1. Check `(s.dropna() % 1 == 0).all()` for that column; test with `df.assign(logins_30d=df.logins_30d + 0.5)`.
2. $(0.4-0.7)\ln(0.4/0.7) + (0.6-0.3)\ln(0.6/0.3) = 0.168 + 0.208 = 0.376$.
3. $9 \times 2/n = 0.02$ gives $n = 900$.
4. The feature PSI drops sharply but the prediction distribution and log loss may still differ; measure. Small, noisy bugs are why canaries need enough traffic and time.
5. Simulate PSI between two driftless samples of the actual sizes a few hundred times, take the 99th percentile, and cache it per window size.
6. Keep a queue of pending weeks; the decision in week $w$ sees AUC from week $w-2$. Expect a two-week lag in confirming the concept drift.
7. Without `np.clip`, empty bins give infinite PSI; whether a test notices depends on whether any test produces an empty bin. Add one.
8. A domain classifier AUC near 0.5 means nothing detectable moved; it rises with the age shift and its feature importances point at age.
9. Shadowing exercises the full serving path and compares predictions at no risk to users, but it cannot measure effects on user behavior; a canary can, at some risk.
10. The script should exit non-zero on failure so the CI service marks the run as failed. Keep the job offline and under a few minutes.

## Check your understanding

**1. Why ship a pipeline rather than a model?** Data keeps changing, so models must be retrained; a pipeline that validates, trains, gates and registers automatically makes retraining routine and reproducible, while a hand-made model file cannot be regenerated or audited.

**2. A canary has higher AUC than the control. Why might it still be broken?** AUC measures only ranking. A bug that shifts every score by the same amount, such as a constant feature value, leaves ranking intact but ruins calibration; log loss, calibration or the prediction distribution reveal it, as in step 4.

**3. Why does a fixed PSI threshold behave differently at different sample sizes?** Driftless PSI has expected value about $(B - 1)(1/n + 1/m)$ from sampling noise alone, so small windows cross 0.1 by chance, while large windows measure a real shift precisely and may keep it below 0.25.

**4. When should monitoring trigger a rollback rather than retraining?** When a newly released version is worse than the version it replaced on the same traffic, for example a canary worse than its control. Drift affects old and new models alike, so it calls for retraining on recent data.

**5. What should a model registry store for each version, and why aliases?** The artifact and its digest, the data fingerprint, code version, library versions, parameters and metrics. Versions are immutable so they can be audited; aliases are movable pointers, so deployment and rollback are small, logged changes.

## Further reading

- **D. Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner, Vinay Chaudhary, Michael Young, Jean-François Crespo, Dan Dennison, "Hidden Technical Debt in Machine Learning Systems"** (2015, Advances in Neural Information Processing Systems 28). <https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html>
  *Read for:* why ML systems are expensive to maintain. *Level:* intro.
- **Google Cloud (Jarek Kazmierczak, Khalid Salama, Valentín Huerta, et al.), "MLOps: Continuous delivery and automation pipelines in machine learning"** (architecture guide, last updated August 28, 2024). <https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning>
  *Read for:* MLOps levels 0 to 2, continuous training and retraining triggers. *Level:* intro.
- **Eric Breck, Shanqing Cai, Eric Nielsen, Michael Salib, D. Sculley, "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction"** (2017, IEEE Big Data). <https://research.google/pubs/pub46555/>
  *Read for:* 28 concrete tests for data, models, infrastructure and monitoring, and the scoring rule. *Level:* intro.
- **Eric Breck, Marty Zinkevich, Neoklis Polyzotis, Steven Whang, Sudip Roy, "Data Validation for Machine Learning"** (2019, SysML/MLSys). <https://research.google/pubs/data-validation-for-machine-learning/>
  *Read for:* schema-based validation and skew detection at production scale. *Level:* intermediate.
- **Shreya Shankar, Rolando Garcia, Joseph M. Hellerstein, Aditya G. Parameswaran, "Operationalizing Machine Learning: An Interview Study"** (2022, arXiv:2209.09125). <https://arxiv.org/abs/2209.09125>
  *Read for:* how 18 ML engineers actually run production pipelines. *Level:* intro.
- **Andrei Paleyes, Raoul-Gabriel Urma, Neil D. Lawrence, "Challenges in Deploying Machine Learning: a Survey of Case Studies"** (2022, ACM Computing Surveys; arXiv:2011.09926). <https://arxiv.org/abs/2011.09926>
  *Read for:* reported deployment problems organized by workflow stage. *Level:* intro.
- **Stephan Rabanser, Stephan Günnemann, Zachary C. Lipton, "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift"** (2019, NeurIPS; arXiv:1810.11953). <https://arxiv.org/abs/1810.11953>
  *Read for:* a careful comparison of shift detectors. *Level:* advanced.
- **Bilal Yurdakul, "Statistical Properties of Population Stability Index"** (2018, PhD dissertation, Western Michigan University). <https://scholarworks.wmich.edu/dissertations/3208/>
  *Read for:* PSI's sampling behavior and how its rules of thumb relate to error rates. *Level:* advanced.
- **Alec Warner and Štěpán Davidovič, with Alex Hidalgo, Betsy Beyer, Kyle Smith, Matt Duftler, "Canarying Releases"** (The Site Reliability Workbook, Google, free online). <https://sre.google/workbook/canarying-releases/>
  *Read for:* canary design: control groups, size, duration and metrics. *Level:* intro.
- **MLflow developers, "MLflow Model Registry"** (documentation, MLflow 3). <https://mlflow.org/docs/latest/ml/model-registry/>
  *Read for:* versions, aliases and tags in a widely used registry, and the deprecation of stages. *Level:* intro.
- **scikit-learn developers, "Model persistence"** (User Guide, version 1.9.1). <https://scikit-learn.org/stable/model_persistence.html>
  *Read for:* pickle's security and version risks, what metadata to save, and safer formats. *Level:* intro.
- **DVC developers, "Data versioning"** (DVC documentation). <https://doc.dvc.org/start/data-management/data-versioning>
  *Read for:* content-addressed data versioning with Git. *Level:* intro.
