"""Fairness criteria conflict when base rates differ (educational demo, synthetic data).

Two groups with different base rates. The score is each person's true risk, so it
is perfectly calibrated within each group. We then check which fairness criteria
hold under different decision rules, and verify Chouldechova's identity.
"""
import numpy as np

rng = np.random.default_rng(0)
N = 200_000  # per group; large so that the gaps below are not sampling noise

# True risk r ~ Beta; outcome y ~ Bernoulli(r). Group A mean risk 0.5, group B 0.2.
groups = {"A": rng.beta(2.0, 2.0, N), "B": rng.beta(1.0, 4.0, N)}
data = {g: (r, (rng.random(N) < r).astype(int)) for g, r in groups.items()}


def rates(y, d):
    tp = np.sum((d == 1) & (y == 1)); fp = np.sum((d == 1) & (y == 0))
    fn = np.sum((d == 0) & (y == 1)); tn = np.sum((d == 0) & (y == 0))
    ppv = tp / (tp + fp) if tp + fp > 0 else np.nan  # undefined if nobody is selected
    return dict(base=y.mean(), sel=d.mean(), TPR=tp / (tp + fn), FPR=fp / (fp + tn),
                PPV=ppv, acc=(tp + tn) / len(y))


def show(title, decisions):
    print(f"\n{title}")
    print(f"  {'group':5} {'base rate':>9} {'selected':>8} {'TPR':>6} {'FPR':>6} {'PPV':>6} {'acc':>6}")
    rs = {}
    for g, (r, y) in data.items():
        rs[g] = rates(y, decisions[g])
        x = rs[g]
        print(f"  {g:5} {x['base']:9.3f} {x['sel']:8.3f} {x['TPR']:6.3f} {x['FPR']:6.3f} "
              f"{x['PPV']:6.3f} {x['acc']:6.3f}")
    gap = {k: abs(rs["A"][k] - rs["B"][k]) for k in ["sel", "TPR", "FPR", "PPV"]}
    print("  gaps |A-B|: " + ", ".join(f"{k} {v:.3f}" for k, v in gap.items()))
    return rs


# 0. Calibration check: within each group, P(y=1 | score in bin) ~ bin center.
print("Calibration within groups (mean outcome by score bin):")
bins = np.linspace(0, 1, 6)
for g, (r, y) in data.items():
    b = np.digitize(r, bins[1:-1])
    print(f"  {g}: " + "  ".join(f"[{bins[i]:.1f},{bins[i+1]:.1f}): {y[b == i].mean():.3f}"
                                  for i in range(5)))

# 1. One threshold for everyone.
rs = show("Rule 1: same threshold 0.5 for both groups (calibrated score)",
          {g: (r > 0.5).astype(int) for g, (r, y) in data.items()})

# Chouldechova's identity: FPR = p/(1-p) * (1-PPV)/PPV * (1-FNR), with FNR = 1 - TPR.
print("  Chouldechova identity check, FPR vs p/(1-p)*(1-PPV)/PPV*TPR:")
for g, x in rs.items():
    p = x["base"]
    rhs = p / (1 - p) * (1 - x["PPV"]) / x["PPV"] * x["TPR"]
    print(f"    {g}: FPR = {x['FPR']:.4f}, identity gives {rhs:.4f}")


def threshold_for(r, target, stat, y):
    """Search a threshold on r so that a group statistic hits a target value."""
    grid = np.quantile(r, np.linspace(0, 1, 2001))
    vals = np.array([rates(y, (r > t).astype(int))[stat] for t in grid])
    return grid[np.nanargmin(np.abs(vals - target))]


# 2. Demographic parity: same selection rate (0.3) in both groups.
show("Rule 2: group thresholds giving equal selection rates (demographic parity)",
     {g: (r > np.quantile(r, 0.7)).astype(int) for g, (r, y) in data.items()})

# 3. Equal opportunity: equal TPR (0.7) in both groups.
show("Rule 3: group thresholds giving equal TPR = 0.7 (equal opportunity)",
     {g: (r > threshold_for(r, 0.7, "TPR", y)).astype(int) for g, (r, y) in data.items()})

# 4. Equal PPV (0.7) in both groups (predictive parity).
show("Rule 4: group thresholds giving equal PPV = 0.7 (predictive parity)",
     {g: (r > threshold_for(r, 0.7, "PPV", y)).astype(int) for g, (r, y) in data.items()})

# 5. The escape hatch: a perfect predictor satisfies all error-rate criteria at
#    once (but still not demographic parity, because base rates differ).
show("Rule 5: perfect predictor (decision = true outcome)",
     {g: y.copy() for g, (r, y) in data.items()})
