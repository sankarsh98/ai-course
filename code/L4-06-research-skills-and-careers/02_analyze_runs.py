"""Analyze the runs written by 01_ablation_harness.py.

1. Mean and 95% confidence interval per config (t interval over seeds).
2. Paired differences from "full": the same seed index shares data order and
   augmentation draws, so pairing removes some shared noise.
3. What one seed would have told you, and how much "best seed" reporting inflates.
Usage:  python 02_analyze_runs.py [--root runs/ablation] [--metric test_acc]
"""
import argparse
from collections import defaultdict

import numpy as np
from scipy import stats

from tracker import load_runs


def t_ci(x, level=0.95):
    x = np.asarray(x)
    half = stats.t.ppf((1 + level) / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - half, x.mean() + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/ablation")
    ap.add_argument("--metric", default="test_acc")
    args = ap.parse_args()

    rows, unfinished = load_runs(args.root)
    by = defaultdict(dict)  # config name -> {seed: metric}
    for r in rows:
        by[r["name"]][r["seed"]] = r[args.metric]
    print(f"{len(rows)} finished runs, {unfinished} unfinished, metric = {args.metric}\n")

    full = by["full"]
    print(f"{'config':18} {'n':>2} {'mean':>6} {'95% CI':>15} | {'diff vs full':>12} "
          f"{'paired 95% CI':>17} {'p':>6} | {'seed 0 diff':>11}")
    for name, res in sorted(by.items(), key=lambda kv: -np.mean(list(kv[1].values()))):
        seeds = sorted(res)
        m, lo, hi = t_ci([res[s] for s in seeds])
        line = f"{name:18} {len(seeds):2d} {m:6.3f} [{lo:6.3f}, {hi:6.3f}] |"
        if name != "full":
            common = sorted(set(seeds) & set(full))
            d = np.array([res[s] - full[s] for s in common])
            dm, dlo, dhi = t_ci(d)
            p = stats.ttest_rel([res[s] for s in common], [full[s] for s in common]).pvalue
            line += f" {dm:+12.3f} [{dlo:+7.3f}, {dhi:+7.3f}] {p:6.3f} | {d[0]:+11.3f}"
        print(line)

    # Reporting the best of k seeds instead of the mean.
    print("\nBest-seed reporting (max over seeds minus mean over seeds):")
    for name in ["full", "none (plain MLP)"]:
        v = np.array(list(by[name].values()))
        print(f"  {name:18} mean {v.mean():.3f}, best {v.max():.3f}, inflation {v.max() - v.mean():+.3f}")

    # A results sentence a reader can check, with the uncertainty in it.
    m, lo, hi = t_ci(list(full.values()))
    print(f"\nReport line: the full method reaches {m:.3f} test accuracy "
          f"(95% CI {lo:.3f} to {hi:.3f}, {len(full)} seeds, 400 training images).")


if __name__ == "__main__":
    main()
