"""The score-function (REINFORCE) gradient on a 3-armed bandit with a softmax
policy, checked against the exact gradient; baselines; and the PPO clip."""
import numpy as np

rng = np.random.default_rng(0)
theta = np.array([0.5, 0.0, -0.5])        # policy logits
mean_r = np.array([1.0, 2.0, 3.0])        # expected reward of each arm
NOISE = 1.0                               # reward = mean_r[a] + N(0, NOISE^2)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


pi = softmax(theta)
J = pi @ mean_r
exact = pi * (mean_r - J)                 # dJ/dtheta_k = pi_k (r_k - J) for softmax
print(f"pi = {np.round(pi, 3)}, J = {J:.3f}, exact grad = {np.round(exact, 4)}")


def grad_samples(n, offset, baseline, G=8):
    """n single-sample gradient estimates (R - b) * grad log pi(a).
    Samples come in groups of G so that group baselines can be formed."""
    a = rng.choice(3, size=(n // G, G), p=pi)
    R = mean_r[a] + offset + NOISE * rng.normal(size=a.shape)
    if baseline == "none":
        b = 0.0
    elif baseline == "true J":
        b = J + offset
    elif baseline == "group mean":       # includes the sample itself (GRPO-style)
        b = R.mean(1, keepdims=True)
    elif baseline == "leave-one-out":    # mean of the other G-1 samples
        b = (R.sum(1, keepdims=True) - R) / (G - 1)
    score = np.eye(3)[a] - pi             # grad log softmax = onehot(a) - pi
    return ((R - b)[..., None] * score).reshape(-1, 3)


print("\n1. Single-sample estimators, 400k samples each")
print("   offset baseline        mean estimate                 bias/exact  total var")
for offset in [0.0, 100.0]:
    for baseline in ["none", "true J", "group mean", "leave-one-out"]:
        g = grad_samples(400_000, offset, baseline)
        m = g.mean(0)
        ratio = (m @ exact) / (exact @ exact)   # 1.0 means unbiased
        print(f"   {offset:6.0f} {baseline:14s} {np.array2string(m, precision=4):28s}"
              f"  {ratio:6.3f}   {g.var(0).sum():10.2f}")

print("\n2. Gradient-ascent steps with batch 64, lr 0.1, offset 100: P(best arm) after 200 steps")
for baseline in ["none", "leave-one-out"]:
    finals = []
    for seed in range(20):
        th = np.zeros(3)
        loc = np.random.default_rng(seed)
        for _ in range(200):
            p = softmax(th)
            a = loc.choice(3, size=(8, 8), p=p)
            R = mean_r[a] + 100 + NOISE * loc.normal(size=a.shape)
            b = 0.0 if baseline == "none" else (R.sum(1, keepdims=True) - R) / 7
            th += 0.1 * ((R - b)[..., None] * (np.eye(3)[a] - p)).reshape(-1, 3).mean(0)
        finals.append(softmax(th)[2])
    print(f"   {baseline:14s} mean {np.mean(finals):.3f}, worst seed {np.min(finals):.3f}")

print("\n3. PPO clipped objective L = min(r A, clip(r, 0.8, 1.2) A) and dL/dr")
print("   ratio r   A=+1: L, dL/dr     A=-1: L, dL/dr")
for r in [0.5, 0.8, 1.0, 1.1, 1.2, 1.5, 3.0]:
    row = []
    for A in [1.0, -1.0]:
        unclipped, clipped = r * A, np.clip(r, 0.8, 1.2) * A
        L = min(unclipped, clipped)
        dL = A if unclipped <= clipped else 0.0   # gradient flows only through r*A
        row.append(f"{L:6.2f}, {dL:4.1f}")
    print(f"   {r:5.2f}     {row[0]}       {row[1]}")
