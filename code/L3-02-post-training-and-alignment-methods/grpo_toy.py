"""SFT, then RL with a verifiable reward (GRPO-style), versus offline DPO.

Task: given two digits a and b, write a + b as two digits (tens, ones).
The policy is a small autoregressive network: it picks the tens digit,
then the ones digit given the tens digit. There are only 100 possible
responses per prompt, so probabilities, pass@k and KL are computed
exactly instead of estimated.

The 'pretraining plus SFT' stage imitates noisy demonstrations. Prompts
are of three kinds, by how often the demonstrations are right:
  common: 60% correct,  rare: 5% correct,  never: 0% correct.
The question: what can RL from a 0/1 correctness reward fix, and what not?
Everything runs twice: with a policy that sees the digits a and b (it can
share what it learns across prompts) and with one that sees only an
arbitrary prompt ID (it cannot).
"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(4)
A, B = torch.meshgrid(torch.arange(10), torch.arange(10), indexing="ij")
A, B = A.flatten(), B.flatten()                     # 100 prompts
ANSWER = A + B                                      # correct response id = 10*tens + ones
kind = torch.zeros(100, dtype=torch.long)           # 0 common, 1 rare, 2 never
perm = torch.randperm(100)
kind[perm[50:80]], kind[perm[80:]] = 1, 2
KINDS = ["common", "rare", "never"]
P_CORRECT = torch.tensor([0.60, 0.05, 0.0])


class Policy(nn.Module):
    def __init__(self, sees_digits, hidden=128):
        super().__init__()
        self.sees_digits = sees_digits
        # Input: the prompt (one-hot a and b, or a one-hot prompt ID), plus a
        # one-hot of the previous token (10 digits + start).
        self.prompt = (torch.cat([F.one_hot(A, 10), F.one_hot(B, 10)], 1) if sees_digits
                       else torch.eye(100)).float()
        self.net = nn.Sequential(nn.Linear(self.prompt.shape[1] + 11, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 10))

    def step_logits(self, prev):                    # prev: (100,) in 0..10
        return self.net(torch.cat([self.prompt, F.one_hot(prev, 11).float()], 1))

    def log_probs(self):
        """(100 prompts, 100 responses) table of log pi(response | prompt)."""
        lp_tens = F.log_softmax(self.step_logits(torch.full((100,), 10)), -1)
        lp_ones = torch.stack([F.log_softmax(self.step_logits(torch.full((100,), t)), -1)
                               for t in range(10)], 1)          # (100, tens, ones)
        return (lp_tens[:, :, None] + lp_ones).flatten(1)


def demo_distribution():
    """What the demonstrations look like: the right answer with probability
    P_CORRECT[kind], otherwise one of four typical mistakes."""
    p = torch.zeros(100, 100)
    s = ANSWER
    mistakes = torch.stack([s - 1, s + 1, s % 10, (s + 10) % 100], 1)  # off by one, lost carry, extra ten
    mistakes = torch.where(mistakes == s[:, None], (s[:, None] + 2) % 100, mistakes)
    mistakes = mistakes.clamp(0, 99)
    p_right = P_CORRECT[kind]
    for m in range(4):
        p[torch.arange(100), mistakes[:, m]] += (1 - p_right) / 4
    p[torch.arange(100), s] += p_right
    return p


def summarize(name, logp, logp_ref):
    p = logp.exp()
    p_ok = p[torch.arange(100), ANSWER]
    kl = (p * (logp - logp_ref)).sum(1)
    ent = -(p * logp).sum(1)
    cells = []
    for k in range(3):
        m = kind == k
        cells.append(f"{p_ok[m].mean():.3f} / {(1 - (1 - p_ok[m]) ** 64).mean():.3f}")
    print(f"{name:12s} " + "  ".join(f"{c:>13s}" for c in cells)
          + f"   {kl.mean():6.3f}  {ent.mean():5.2f}")


# ---- 1. SFT: maximum likelihood on 200 sampled demonstrations per prompt.
def sft(sees_digits, demos, epochs=300, lr=3e-3):
    pol = Policy(sees_digits)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    for _ in range(epochs):
        loss = -pol.log_probs().gather(1, demos).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return pol


def copy_of(pol):
    new = Policy(pol.sees_digits)
    new.load_state_dict(pol.state_dict())
    return new


# ---- 2. GRPO-style RL. For each prompt sample a group of G responses, reward
# 1 if correct, advantage = (reward - group mean) / (group std + eps). The
# loss is -advantage * log pi(response), plus beta times the k3 estimate of
# KL(pi || ref) on the sampled responses (the estimator GRPO uses). We take
# one gradient step per batch of fresh samples, so pi_old = pi and PPO's
# ratio clipping never activates; it is omitted.
def grpo(ref, logp_ref, steps=300, G=8, beta=0.02, lr=3e-3):
    pol = copy_of(ref)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    for _ in range(steps):
        logp = pol.log_probs()
        with torch.no_grad():
            y = torch.multinomial(logp.exp(), G, replacement=True)   # (100, G)
            reward = (y == ANSWER[:, None]).float()
            adv = (reward - reward.mean(1, keepdim=True)) / (reward.std(1, keepdim=True) + 1e-4)
        lp = logp.gather(1, y)
        log_ratio = logp_ref.gather(1, y) - lp                       # log(ref / pi)
        k3 = log_ratio.exp() - log_ratio - 1                         # >= 0, unbiased for KL
        loss = -(adv * lp).mean() + beta * k3.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return pol.log_probs()


# ---- 3. Offline DPO on pairs built from the SFT model's own samples:
# 16 samples per prompt, every (correct, incorrect) pair becomes
# (chosen, rejected). Prompts with no correct sample contribute nothing.
def dpo_pairs(logp_ref, n=16):
    y = torch.multinomial(logp_ref.exp(), n, replacement=True)
    ok = y == ANSWER[:, None]
    rows, chosen, rejected = [], [], []
    for x in range(100):
        for c in y[x][ok[x]].tolist():
            for r_ in y[x][~ok[x]].tolist():
                rows.append(x), chosen.append(c), rejected.append(r_)
    rows = torch.tensor(rows)
    covered = torch.zeros(100, dtype=torch.bool)
    covered[rows] = True
    print(f"   DPO data: {len(rows)} pairs; prompts with at least one pair: "
          + ", ".join(f"{KINDS[k]} {covered[kind == k].sum()}/{(kind == k).sum()}" for k in range(3)))
    return rows, torch.tensor(chosen), torch.tensor(rejected)


def dpo(ref, logp_ref, data, beta, steps=300, lr=3e-3):
    rows, chosen, rejected = data
    pol = copy_of(ref)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    for _ in range(steps):
        logp = pol.log_probs()
        h = ((logp[rows, chosen] - logp_ref[rows, chosen])
             - (logp[rows, rejected] - logp_ref[rows, rejected]))
        loss = -F.logsigmoid(beta * h).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return pol.log_probs()


demos = torch.multinomial(demo_distribution(), 200, replacement=True)   # (100, 200)
t0 = time.time()
for sees_digits in [True, False]:
    torch.manual_seed(1)
    print(f"\n=== policy input: {'the digits a, b' if sees_digits else 'a prompt ID only'} ===")
    ref = sft(sees_digits, demos)
    with torch.no_grad():
        logp_ref = ref.log_probs()
    data = dpo_pairs(logp_ref)
    print("pass@1 / pass@64 by prompt kind (exact), mean KL to SFT model, mean entropy (nats)")
    print(f"{'':12s} " + "  ".join(f"{k:>13s}" for k in KINDS) + "       KL  entropy")
    summarize("SFT model", logp_ref, logp_ref)
    summarize("GRPO", grpo(ref, logp_ref), logp_ref)
    summarize("DPO beta=0.1", dpo(ref, logp_ref, data, beta=0.1), logp_ref)
    summarize("DPO beta=1", dpo(ref, logp_ref, data, beta=1.0), logp_ref)
print(f"\n({time.time() - t0:.1f} s in total)")
