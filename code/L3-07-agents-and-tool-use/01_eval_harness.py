"""Evaluate scripted agents on a 30-task suite: pass@1 with intervals, run-to-run
variance, pass@k versus pass^k, and cost. Agents are stand-ins for models (see shop.py).
"""
from math import comb

import numpy as np

from shop import AGENTS, make_tasks, run_trial

N_TRIALS = 16                      # trials per task per agent
tasks = make_tasks()
print(f"{len(tasks)} tasks:", {k: sum(t['kind'] == k for t in tasks)
                               for k in ("refund", "cancel", "partial")})

# results[name] is a (tasks x trials) boolean matrix; tokens likewise.
results, tokens = {}, {}
for a_idx, (name, agent) in enumerate(AGENTS.items()):
    ok = np.zeros((len(tasks), N_TRIALS), bool)
    tok = np.zeros((len(tasks), N_TRIALS))
    for i, task in enumerate(tasks):
        for j in range(N_TRIALS):
            seed = 10**6 * a_idx + 1000 * i + j          # independent draws per agent
            ok[i, j], env = run_trial(agent, tasks, task, seed)
            tok[i, j] = env.input_tokens + env.output_tokens
    results[name], tokens[name] = ok, tok

# ---- 1. pass@1 with a bootstrap interval over tasks -------------------------
rng = np.random.default_rng(0)
print("\n1. pass@1 (mean over 16 trials), 95% bootstrap CI over tasks")
for name, ok in results.items():
    per_task = ok.mean(axis=1)
    boots = [per_task[rng.integers(0, len(tasks), len(tasks))].mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  {name:17s} {per_task.mean():.3f}  [{lo:.3f}, {hi:.3f}]")

# ---- 2. What a single run per task would have told you ----------------------
print("\n2. Single run per task (one column), across the 16 possible runs")
for name in ("sampled-A", "sampled-B"):
    runs = results[name].mean(axis=0)
    print(f"  {name}: min {runs.min():.3f}  max {runs.max():.3f}  sd {runs.std(ddof=1):.3f}")
a, b = results["sampled-A"].mean(axis=0), results["sampled-B"].mean(axis=0)
pairs = [(x, y) for x in a for y in b]           # every pairing of one A run with one B run
print(f"  A truly better on average ({results['sampled-A'].mean():.3f} vs "
      f"{results['sampled-B'].mean():.3f}), yet one run each puts B level or ahead in "
      f"{np.mean([y >= x for x, y in pairs]):.0%} of pairings")
diff = results["sampled-A"].mean(1) - results["sampled-B"].mean(1)
boots = [diff[rng.integers(0, len(tasks), len(tasks))].mean() for _ in range(2000)]
print(f"  paired A-B difference over tasks, 16 trials: {diff.mean():+.3f}, "
      f"95% CI [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}]")


# ---- 3. pass@k (some trial succeeds) versus pass^k (every trial succeeds) ---
def pass_at_k(n, c, k):   # unbiased estimator, Chen et al. (2021)
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n, c, k):  # unbiased estimator, Yao et al. (2024)
    return comb(c, k) / comb(n, k)


print("\n3. pass@k and pass^k, averaged over tasks")
print("  agent              " + "".join(f"  @{k:<5d}" for k in (1, 2, 4, 8))
      + "".join(f"  ^{k:<5d}" for k in (2, 4, 8)))
for name, ok in results.items():
    c = ok.sum(axis=1)
    at = [np.mean([pass_at_k(N_TRIALS, ci, k) for ci in c]) for k in (1, 2, 4, 8)]
    hat = [np.mean([pass_hat_k(N_TRIALS, ci, k) for ci in c]) for k in (2, 4, 8)]
    print(f"  {name:17s}" + "".join(f"  {v:.3f} " for v in at + hat))
print(f"  one task with 4 of 16 trials passing: unbiased pass@8 = {pass_at_k(16, 4, 8):.3f}, "
      f"plug-in 1-(1-4/16)^8 = {1 - (1 - 4 / 16) ** 8:.3f}")

# ---- 4. Where the failures are ----------------------------------------------
print("\n4. pass@1 by task kind")
for name, ok in results.items():
    row = {k: ok[[t["kind"] == k for t in tasks]].mean() for k in ("refund", "cancel", "partial")}
    print(f"  {name:17s} " + "  ".join(f"{k} {v:.2f}" for k, v in row.items()))

# ---- 5. Cost ------------------------------------------------------------------
print("\n5. Tokens (simulated): mean per trial, and per successful trial")
for name, ok in results.items():
    t = tokens[name]
    print(f"  {name:17s} {t.mean():7.0f} per trial   {t.sum() / ok.sum():7.0f} per success")
