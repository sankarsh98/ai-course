"""DPO on a five-response bandit: check the derivation, then break it.

One prompt, five possible responses, a known reference policy and known
true rewards. The RLHF objective max E[r] - beta KL(pi || ref) has the
exact solution pi* ~ ref * exp(r / beta). We check that:
  1. DPO trained on Bradley-Terry preference probabilities finds pi*;
  2. With one sampled label per pair (every pair then looks unanimous),
     DPO keeps drifting away from ref however long it trains, while IPO
     settles at a bounded distance.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
BETA = 1.0
ref = torch.tensor([0.30, 0.25, 0.20, 0.15, 0.10])
r = torch.tensor([1.0, 0.5, 0.0, -0.5, -1.0])
log_ref = ref.log()
pi_star = torch.softmax(log_ref + r / BETA, 0)

pairs = torch.combinations(torch.arange(5), 2)      # all 10 pairs (i, j), i < j
i, j = pairs[:, 0], pairs[:, 1]
p_true = torch.sigmoid(r[i] - r[j])                  # P(i preferred over j)


def set_pairs(new_pairs):
    global i, j
    i, j = new_pairs[:, 0], new_pairs[:, 1]


def kl(p, q):
    return (p * (p / q).log()).sum().item()


def margin(logits):
    """h = log pi(i)/ref(i) - log pi(j)/ref(j): DPO's implicit reward gap / beta."""
    logp = F.log_softmax(logits, 0)
    return (logp[i] - log_ref[i]) - (logp[j] - log_ref[j])


def dpo_loss(logits, p):
    # Each pair is labeled 'i wins' with probability p, 'j wins' otherwise.
    h = margin(logits)
    return -(p * F.logsigmoid(BETA * h) + (1 - p) * F.logsigmoid(-BETA * h)).mean()


def ipo_loss(logits, p):
    # IPO (Azar et al.): regress h toward +1/(2 beta) when i wins, -1/(2 beta)
    # when j wins; with label frequency p this is a weighted square loss.
    h = margin(logits)
    t = 1 / (2 * BETA)
    return (p * (h - t) ** 2 + (1 - p) * (h + t) ** 2).mean()


def train(loss_fn, p, steps, report_at=()):
    logits = log_ref.clone().requires_grad_(True)    # start at pi = ref
    opt = torch.optim.Adam([logits], lr=0.05)
    for step in range(1, steps + 1):
        loss = loss_fn(logits, p)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in report_at:
            pi = torch.softmax(logits.detach(), 0)
            print(f"    step {step:5d}  pi = {fmt(pi)}  KL(pi||ref) {kl(pi, ref):.3f}")
    return torch.softmax(logits.detach(), 0)


fmt = lambda v: "[" + ", ".join(f"{x:.3f}" for x in v.tolist()) + "]"
print(f"ref       = {fmt(ref)}")
print(f"pi* (RLHF optimum, beta={BETA}) = {fmt(pi_star)}  KL(pi*||ref) {kl(pi_star, ref):.3f}")

print("\n1. DPO on exact Bradley-Terry preference probabilities")
pi = train(dpo_loss, p_true, 3000)
print(f"   DPO result = {fmt(pi)}  max |pi - pi*| = {(pi - pi_star).abs().max():.1e}")

print("\n2. One labeled comparison for each adjacent pair (0,1) (1,2) (2,3) (3,4)")
set_pairs(torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]]))
p_true = torch.sigmoid(r[i] - r[j])
label = torch.bernoulli(p_true)                      # a single 0/1 label each
print(f"   true P(first wins): {fmt(p_true)}, sampled labels: {label.int().tolist()}")
print("  DPO:")
train(dpo_loss, label, 10000, report_at=(100, 1000, 3000, 10000))
print("  IPO:")
train(ipo_loss, label, 10000, report_at=(100, 1000, 3000, 10000))
