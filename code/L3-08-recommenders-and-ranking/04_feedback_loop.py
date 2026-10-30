"""Part 1: a recommender retrained on the clicks it caused (a feedback loop).
Part 2: evaluating new policies offline from biased logs: naive replay of clicks
versus inverse propensity scoring (IPS). Ground truth comes from the synthetic world.
"""
import numpy as np

from recdata import gini, ials, make_world, organic_clicks

world = make_world()
cp = world["click_prob"]
n_u, n_i = cp.shape
K, ROUNDS = 10, 8
top20 = np.argsort(-world["pop"])[:20]


def choose(policy, clicks, rng):
    """Return a (users x K) slate of items for each user."""
    seen = clicks > 0
    if policy == "random":
        s = rng.random((n_u, n_i))
    elif policy == "oracle":
        s = cp.copy()
    elif policy == "popularity":
        s = np.tile(clicks.sum(0), (n_u, 1)) + 1e-3 * rng.random((n_u, n_i))
    else:                                              # "iALS" or "iALS+explore"
        U, V = ials(clicks, k=8, alpha=10, reg=1, iters=6)
        s = U @ V.T
    s = np.where(seen, -np.inf, s)                     # do not re-show clicked items
    slate = np.argsort(-s, axis=1)[:, :K]
    if policy == "iALS+explore":                       # 2 of 10 slots: random unseen items
        for u in range(n_u):
            pool = np.setdiff1d(np.flatnonzero(~seen[u]), slate[u, :K - 2])
            slate[u, K - 2:] = rng.choice(pool, 2, replace=False)
    return slate


def jaccard_homogeneity(slates, rng, pairs=3000):
    a, b = rng.integers(0, n_u, pairs), rng.integers(0, n_u, pairs)
    return np.mean([len(set(slates[x]) & set(slates[y])) / len(set(slates[x]) | set(slates[y]))
                    for x, y in zip(a, b) if x != y])


print("Part 1: 8 rounds; each round every user sees 10 new items and clicks some.")
print("policy        clicks/user  CTR rd1  CTR rd8  coverage rd8  exposure Gini  top-20  homog.")
for policy in ("random", "popularity", "iALS", "iALS+explore", "oracle"):
    rng = np.random.default_rng(0)
    clicks = organic_clicks(world)                     # same starting log for everyone
    exposure = np.zeros(n_i)
    ctr = []
    for t in range(ROUNDS):
        slate = choose(policy, clicks, rng)
        p = np.take_along_axis(cp, slate, axis=1)
        new = rng.random(p.shape) < p
        np.put_along_axis(clicks, slate, np.maximum(np.take_along_axis(clicks, slate, 1), new), 1)
        np.add.at(exposure, slate.ravel(), 1)
        ctr.append(new.mean())
    print(f"{policy:14s}  {K * sum(ctr):9.2f}  {ctr[0]:7.3f}  {ctr[-1]:7.3f}"
          f"  {len(np.unique(slate)) / n_i:12.0%}  {gini(exposure):13.2f}"
          f"  {np.isin(slate, top20).mean():6.0%}  {jaccard_homogeneity(slate, rng):6.3f}")

# ---------------------------------------------------------------- Part 2
# A logging policy shows ONE item per event, mostly by popularity, with 10% uniform
# exploration, and records the probability of what it showed (the propensity).
rng = np.random.default_rng(1)
pi0 = 0.9 * world["pop"] + 0.1 / n_i
events = 60_000
users = rng.integers(0, n_u, events)
shown = rng.choice(n_i, events, p=pi0)
clicked = rng.random(events) < cp[users, shown]
prop = pi0[shown]

# Candidate policies to evaluate, each recommending one item per user.
U, V = ials(organic_clicks(world), k=8, alpha=10, reg=1)
targets = {"popularity top-1": np.full(n_u, np.argmax(world["pop"])),
           "iALS top-1": np.argmax(U @ V.T, axis=1),
           "oracle top-1": np.argmax(cp, axis=1)}

print("\nPart 2: offline estimates of each policy's click rate from the logged events")
print("policy             true   replay   IPS (95% CI)            SNIPS   matched events")
for name, rec in targets.items():
    truth = cp[np.arange(n_u), rec].mean()
    match = shown == rec[users]
    replay = clicked[match].mean()                     # average over matched events only
    # IPS: reweight matched events by 1 / propensity so rare actions count more.
    w = match * clicked / prop
    ips, se = w.mean(), w.std(ddof=1) / np.sqrt(events)
    snips = w.sum() / (match / prop).sum()             # self-normalized IPS
    print(f"{name:17s}  {truth:5.3f}  {replay:6.3f}   {ips:5.3f} [{ips - 1.96 * se:.3f}, "
          f"{ips + 1.96 * se:.3f}]   {snips:5.3f}  {match.sum():8d}")

# The most common offline protocol: did the policy's top-10 contain the logged clicks?
print("\n'Hit rate' of each policy's top-10 on logged clicks (the usual offline metric):")
clicked_pairs = (users[clicked], shown[clicked])
for name, scores in [("popularity", np.tile(world["pop"], (n_u, 1))), ("iALS", U @ V.T),
                     ("oracle", cp)]:
    top10 = np.argsort(-scores, axis=1)[:, :10]
    hit = (top10[clicked_pairs[0]] == clicked_pairs[1][:, None]).any(1).mean()
    print(f"   {name:10s} {hit:.3f}   true mean click prob of its top-10: "
          f"{np.take_along_axis(cp, top10, 1).mean():.3f}")
