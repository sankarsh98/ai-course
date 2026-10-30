"""Fit a Bradley-Terry reward model, then over-optimize it (Goodhart in miniature).

A 'response' is a feature vector: four quality features and a length. A
hidden gold reward (standing in for what people actually want) likes a
little extra length but penalizes very long answers. Simulated labelers
compare pairs of responses drawn from the reference policy, with
Bradley-Terry noise, and are also swayed by length beyond its real value.
A linear reward model is fit to their labels; it cannot represent the
length penalty. Then we optimize against it two ways and watch the gold
reward.
"""
import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)
D_Q = 4                                   # quality features
w_gold = torch.full((D_Q,), 0.4)
LABELER_LENGTH_BIAS = 0.5


def gold(x):                              # x[:, :D_Q] quality, x[:, D_Q] length
    q, ln = x[:, :D_Q], x[:, D_Q]
    return q @ w_gold + 0.3 * ln - 0.35 * ln ** 2


def labeler_score(x):                    # what the raters actually respond to
    return gold(x) + LABELER_LENGTH_BIAS * x[:, D_Q]


def sample_ref(n):                        # the reference policy's responses
    return torch.randn(n, D_Q + 1)        # length is standardized: mean 0, sd 1


# 1. Preference data: 1,000 labeled pairs, P(y1 preferred) =
#    sigmoid(s(y1) - s(y2)), with s the labelers' (length-biased) score.
n_pairs = 1000
y1, y2 = sample_ref(n_pairs), sample_ref(n_pairs)
p1 = torch.sigmoid(labeler_score(y1) - labeler_score(y2))
label = torch.bernoulli(p1)              # 1 if labeler preferred y1
chosen = torch.where(label[:, None] == 1, y1, y2)
rejected = torch.where(label[:, None] == 1, y2, y1)

# 2. Reward model: linear in the features, trained with the Bradley-Terry loss
#    -log sigmoid(r(chosen) - r(rejected)). Only differences matter, so no bias.
w = torch.zeros(D_Q + 1, requires_grad=True)
opt = torch.optim.Adam([w], lr=0.05)
for _ in range(500):
    loss = -F.logsigmoid(chosen @ w - rejected @ w).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
w = w.detach()
proxy = lambda x: x @ w

t1, t2 = sample_ref(20000), sample_ref(20000)
agree = ((proxy(t1) > proxy(t2)) == (gold(t1) > gold(t2))).float().mean()
print(f"reward model: BT loss {loss.item():.3f}, agrees with gold ranking on "
      f"{agree:.1%} of fresh pairs")
print(f"  learned weights: quality {[round(v, 2) for v in w[:D_Q].tolist()]}, "
      f"length {w[D_Q]:.2f}")
print("  (gold: quality 0.4 each, length slope +0.3 at the mean, curving down;"
      " labelers add +0.5 per unit of length)")

# 3. Best-of-n: draw n responses from the reference, keep the one the proxy
#    scores highest. The KL from the reference of the winner's distribution
#    is ln n - (n - 1)/n for a continuous reward (derived in the text).
print("\nbest-of-n against the proxy (400 trials per n)")
for n in [1, 4, 16, 64, 256, 1024, 4096]:
    x = sample_ref(400 * n).view(400, n, D_Q + 1)
    best = x[torch.arange(400), proxy(x.flatten(0, 1)).view(400, n).argmax(1)]
    print(f"  n={n:5d}  KL {math.log(n) - (n - 1) / n:5.2f} nats   proxy "
          f"{proxy(best).mean():5.2f}   gold {gold(best).mean():5.2f}   "
          f"mean length {best[:, D_Q].mean():5.2f}")

# 4. KL-regularized RL: maximize E[r] - beta KL(pi || ref). The exact optimum
#    is pi(x) proportional to ref(x) exp(r(x) / beta). With a Gaussian
#    reference and r = a.x - c * length^2, that optimum is again Gaussian, so
#    every number below is exact rather than sampled.


def tilted(a, c, beta):
    """Mean and variance of N(0, I) * exp((a.x - c x_len^2) / beta), normalized."""
    var = torch.ones(D_Q + 1)
    var[D_Q] = 1 / (1 + 2 * c / beta)      # the quadratic term narrows length
    return a / beta * var, var


def kl_to_ref(mean, var):                  # KL(N(mean, diag var) || N(0, I))
    return 0.5 * (var + mean ** 2 - 1 - var.log()).sum()


def expected(mean, var, a, c):             # E[a.x - c x_len^2] under the Gaussian
    return a @ mean - c * (mean[D_Q] ** 2 + var[D_Q])


a_gold = torch.cat([w_gold, torch.tensor([0.3])])
for label_, a_opt, c_opt in [("optimizing the proxy reward model", w, 0.0),
                             ("optimizing the gold reward (the unreachable ideal)", a_gold, 0.35)]:
    print(f"\nKL-regularized optimum, {label_}")
    for beta in [4.0, 2.0, 1.0, 0.5, 0.35, 0.25, 0.1]:
        m, v = tilted(a_opt, c_opt, beta)
        print(f"  beta={beta:4.2f}  KL {kl_to_ref(m, v):6.2f} nats   proxy "
              f"{expected(m, v, w, 0.0):5.2f}   gold {expected(m, v, a_gold, 0.35):6.2f}"
              f"   mean length {m[D_Q]:5.2f}")
