"""Three probing pitfalls on a small MLP trained on scikit-learn's 8x8 digits.

1. A probe needs baselines: random networks and raw inputs are decodable too.
2. A probe for a concept that is a function of the class label is easy for any
   representation that encodes the class; control tasks expose this.
3. Decodable is not the same as used: remove a probe's direction and see
   whether the model's behavior changes.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
rng = np.random.default_rng(0)
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)


def make_mlp():
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                         nn.Linear(64, 10))


def hidden(net, x):
    """Activations after the second ReLU, the layer we probe."""
    with torch.no_grad():
        return net[:4](torch.from_numpy(x)).numpy()


trained, random_net = make_mlp(), make_mlp()
opt = torch.optim.Adam(trained.parameters(), lr=3e-3)
xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
for epoch in range(300):
    loss = nn.functional.cross_entropy(trained(xt), yt)
    opt.zero_grad()
    loss.backward()
    opt.step()
with torch.no_grad():
    acc = (trained(torch.from_numpy(Xte)).argmax(1).numpy() == yte).mean()
print(f"trained MLP test accuracy: {acc:.3f}\n")


def probe_acc(Htr, Hte, ltr, lte):
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(Htr, ltr)
    return clf.score(Hte, lte), clf


reps = {"trained MLP layer 2": (hidden(trained, Xtr), hidden(trained, Xte)),
        "random MLP layer 2": (hidden(random_net, Xtr), hidden(random_net, Xte)),
        "raw pixels": (Xtr, Xte)}

# ---- 1 and 2: the concept "digit is even", against control partitions -------
even = np.array([0, 2, 4, 6, 8])
controls = [rng.permutation(10)[:5] for _ in range(20)]   # random 5-vs-5 class splits
print("probe accuracy (logistic regression, held-out digits)")
print(f"{'representation':22s} {'is-even':>8s} {'control (mean of 20)':>21s} {'selectivity':>12s}")
for name, (Htr, Hte) in reps.items():
    a_even, _ = probe_acc(Htr, Hte, np.isin(ytr, even), np.isin(yte, even))
    a_ctrl = np.mean([probe_acc(Htr, Hte, np.isin(ytr, c), np.isin(yte, c))[0]
                      for c in controls])
    print(f"{name:22s} {a_even:8.3f} {a_ctrl:21.3f} {a_even - a_ctrl:12.3f}")

# ---- 3: decodable but not used ------------------------------------------------
# "Ink": is the image's total brightness above the median? The classifier was
# never asked about ink. Find the probe direction, project it out of the layer-2
# activations, and see what the rest of the network does.
Htr, Hte = reps["trained MLP layer 2"]
ink_tr = Xtr.sum(1) > np.median(Xtr.sum(1))
ink_te = Xte.sum(1) > np.median(Xtr.sum(1))
a_ink, clf_ink = probe_acc(Htr, Hte, ink_tr, ink_te)
_, clf_cls = probe_acc(Htr, Hte, ytr, yte)


def accuracy_after_removing(directions):
    """Project the directions out of layer-2 activations, then run the last layer."""
    H = Hte
    if len(directions):
        Q, _ = np.linalg.qr(directions.T)      # orthonormal basis, (64, k)
        H = Hte - (Hte @ Q) @ Q.T
    with torch.no_grad():
        out = trained[4](torch.from_numpy(H.astype(np.float32)))
    return (out.argmax(1).numpy() == yte).mean()


print(f"\nink probe accuracy on layer 2: {a_ink:.3f}")
print("classifier accuracy after projecting directions out of layer 2:")
print(f"  nothing removed                     {accuracy_after_removing(np.zeros((0, 64))):.3f}")
print(f"  ink probe direction (1 dim)         {accuracy_after_removing(clf_ink.coef_):.3f}")
rand = rng.standard_normal((1, 64))
print(f"  random direction (1 dim)            {accuracy_after_removing(rand):.3f}")
print(f"  class-probe directions (10 dims)    {accuracy_after_removing(clf_cls.coef_):.3f}")
rand10 = rng.standard_normal((10, 64))
print(f"  random directions (10 dims)         {accuracy_after_removing(rand10):.3f}")
# Iterate: after removing one ink direction, is ink still decodable? (A single
# linear direction rarely removes all the information.)
Q, _ = np.linalg.qr(clf_ink.coef_.T)
a_ink2, _ = probe_acc(Htr - (Htr @ Q) @ Q.T, Hte - (Hte @ Q) @ Q.T, ink_tr, ink_te)
print(f"ink probe retrained after removal:    {a_ink2:.3f}")
