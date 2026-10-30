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
