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
