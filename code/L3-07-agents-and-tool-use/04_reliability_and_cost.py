"""Two pieces of arithmetic every agent designer should know:
how per-step reliability compounds, and how re-reading the context makes cost grow
quadratically with the number of turns. All prices and sizes are illustrative.
"""
import numpy as np

# ---- 1. Compounding: a task of n steps that all must succeed -----------------
print("1. P(task succeeds) = s^n when every one of n steps must succeed")
print("   s \\ n " + "".join(f"{n:>8d}" for n in (5, 20, 50, 100)))
for s in (0.99, 0.95, 0.90):
    print(f"   {s:5.2f} " + "".join(f"{s ** n:8.3f}" for n in (5, 20, 50, 100)))

# A checker that detects a fraction c of failed steps, followed by one retry,
# raises per-step success from s to s + (1 - s) * c * s.
print("\n   with a checker that catches a fraction c of step failures and retries once:")
for c in (0.5, 0.9):
    s = 0.95
    s_eff = s + (1 - s) * c * s
    print(f"   s=0.95, c={c}: effective step success {s_eff:.4f}, "
          f"20 steps -> {s_eff ** 20:.3f}, 50 steps -> {s_eff ** 50:.3f}")

# ---- 2. Cost: every turn re-reads the whole context ----------------------------
S, D = 3000, 600        # tokens: fixed prompt + tools; tokens added per turn


def input_tokens(turns, compact_every=None, summary=1500, start=S):
    """Total input tokens over a run. Optional compaction replaces history by a summary."""
    total, ctx = 0, start
    for t in range(1, turns + 1):
        total += ctx
        ctx += D
        if compact_every and t % compact_every == 0:
            ctx = S + summary
    return total


print("\n2. Total input tokens for one agent run (S=3000 fixed, +600 per turn)")
print("   turns   full history   closed form   compact every 20   ratio full/compact")
for T in (10, 50, 100, 200):
    closed = T * S + D * T * (T - 1) // 2
    full, comp = input_tokens(T), input_tokens(T, compact_every=20)
    print(f"   {T:5d}   {full:12,d}   {closed:11,d}   {comp:16,d}   {full / comp:8.1f}")

# Caching: if a provider bills re-read (cached) prefix tokens at a fraction f of the
# normal input price, the quadratic term shrinks by f but does not go away.
print("\n   cost in 'normal input token' units for 100 turns if cached re-reads cost f:")
T = 100
full = input_tokens(T)
fresh = S + D * T                                    # each token is new exactly once
for f in (1.0, 0.25, 0.1):
    print(f"   f={f:4.2f}: {fresh + f * (full - fresh):12,.0f}")

# ---- 3. Splitting the same 120 turns of work across agents ----------------------
print("\n3. 120 turns of work: one agent, or a lead (20 turns) + 4 subagents (25 turns each)")
single = input_tokens(120)
isolated = input_tokens(20) + 4 * input_tokens(25)
# Sharing the lead's full trace with each subagent (start from the lead's context):
shared = input_tokens(20) + 4 * input_tokens(25, start=S + 20 * D)
for label, v in [("single agent", single), ("subagents, fresh context", isolated),
                 ("subagents, shared trace", shared)]:
    print(f"   {label:25s} {v:11,d}   ({v / single:.2f}x)")
