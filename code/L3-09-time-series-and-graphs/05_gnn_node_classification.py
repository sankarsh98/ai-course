"""Semi-supervised node classification on synthetic graphs (a contextual
stochastic block model): MLP vs GCN vs a GNN that keeps the node's own
features separate, as edges go from homophilous to heterophilous. Then:
over-smoothing when propagation is repeated many times."""
import os

# Many small fits: extra threads mostly add overhead, and on a shared machine
# they can slow things down a lot. Must be set before numpy is imported.
for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(var, "4")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

torch.set_num_threads(4)              # small sparse ops: more threads only add overhead

N, C, D, DEG = 1200, 3, 16, 10        # nodes, classes, feature dim, mean degree


def make_graph(h, seed):
    """Classes are balanced. Features: class mean + noise (weakly informative).
    A fraction h of edges (in expectation) joins nodes of the same class."""
    g = np.random.default_rng(seed)
    y = np.repeat(np.arange(C), N // C)
    means = 1.0 * np.eye(C, D)                   # equally separated class means
    x = means[y] + g.normal(0, 1, (N, D))
    same = y[:, None] == y[None, :]
    p_in = h * DEG / (N / C)
    p_out = (1 - h) * DEG / (N - N / C)
    upper = np.triu(g.random((N, N)) < np.where(same, p_in, p_out), 1)
    src, dst = np.nonzero(upper | upper.T)
    split = g.permutation(N)
    train = np.concatenate([split[y[split] == c][:20] for c in range(C)])  # 20 labels/class
    rest = np.setdiff1d(split, train)
    return (torch.tensor(x, dtype=torch.float32), torch.tensor(y),
            np.stack([src, dst]), train, rest[:300], rest[300:])


def sparse_op(ei, kind):
    """'gcn': D^-1/2 (A+I) D^-1/2 (as checked in script 4); 'mean': D^-1 A."""
    src, dst = torch.tensor(ei[0]), torch.tensor(ei[1])
    if kind == "gcn":
        loops = torch.arange(N)
        src, dst = torch.cat([src, loops]), torch.cat([dst, loops])
    deg = torch.zeros(N).index_add_(0, dst, torch.ones(len(dst)))
    w = (deg[src] * deg[dst]).rsqrt() if kind == "gcn" else 1 / deg[dst].clamp(min=1)
    return torch.sparse_coo_tensor(torch.stack([dst, src]), w, (N, N),
                                   check_invariants=True).coalesce()


class Net(torch.nn.Module):
    """kind='mlp': ignores edges. 'gcn': h' = A_gcn h W.
    'ego+nbr': h' = h W_self + (mean of neighbours) W_nbr, so a node's own
    features are never averaged away (one of the designs Zhu et al. 2020 recommend)."""

    def __init__(self, kind, hidden=32):
        super().__init__()
        self.kind = kind
        self.l1, self.l2 = torch.nn.Linear(D, hidden), torch.nn.Linear(hidden, C)
        self.n1, self.n2 = torch.nn.Linear(D, hidden), torch.nn.Linear(hidden, C)

    def layer(self, h, lin, nbr, op):
        if self.kind == "mlp":
            return lin(h)
        if self.kind == "gcn":
            return torch.sparse.mm(op, lin(h))
        return lin(h) + torch.sparse.mm(op, nbr(h))

    def forward(self, x, op):
        h = torch.relu(self.layer(x, self.l1, self.n1, op))
        h = torch.nn.functional.dropout(h, 0.5, self.training)
        return self.layer(h, self.l2, self.n2, op)


def train_eval(kind, data, seed):
    x, y, ei, tr, va, te = data
    torch.manual_seed(seed)
    op = sparse_op(ei, "gcn" if kind == "gcn" else "mean")
    net = Net(kind)
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, test_at_best = 0, 0
    for _ in range(200):
        net.train()
        loss = torch.nn.functional.cross_entropy(net(x, op)[tr], y[tr])
        opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(x, op).argmax(1)
        val = (pred[va] == y[va]).float().mean().item()
        if val > best_val:                         # model selection on validation only
            best_val, test_at_best = val, (pred[te] == y[te]).float().mean().item()
    return test_at_best


print(f"{N} nodes, {C} classes, mean degree {DEG}, 20 labelled nodes per class")
print("test accuracy, mean (sd) over 5 graphs/seeds; chance = 0.33\n")
print(f"{'homophily':>9} {'MLP':>12} {'GCN':>12} {'ego+nbr':>12}")
for h in [0.9, 0.7, 0.5, 0.33, 0.1]:
    row = []
    for kind in ["mlp", "gcn", "ego+nbr"]:
        accs = [train_eval(kind, make_graph(h, s), s) for s in range(5)]
        row.append(f"{np.mean(accs):.2f} ({np.std(accs):.2f})")
    print(f"{h:>9} " + " ".join(f"{r:>12}" for r in row))

# ---- Over-smoothing: propagate features K times, then fit logistic regression.
x, y, ei, tr, va, te = make_graph(0.9, 0)
op = sparse_op(ei, "gcn")
print("\nrepeated GCN propagation on a homophilous graph (h = 0.9), no learning inside:")
print(f"{'K':>4} {'test acc':>9} {'spread':>8}   (spread: mean distance of unit-length rows from their mean)")
z = x.clone()
for K in range(0, 65):
    if K in [0, 1, 2, 4, 8, 16, 32, 64]:
        clf = LogisticRegression(max_iter=2000).fit(z[tr].numpy(), y[tr].numpy())
        acc = clf.score(z[te].numpy(), y[te].numpy())
        u = torch.nn.functional.normalize(z, dim=1)
        spread = (u - u.mean(0)).norm(dim=1).mean().item()
        print(f"{K:>4} {acc:9.2f} {spread:8.3f}")
    z = torch.sparse.mm(op, z)
