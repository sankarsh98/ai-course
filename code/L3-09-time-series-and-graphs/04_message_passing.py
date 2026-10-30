"""Message passing from scratch on an edge list, checked four ways:
1. one hand-sized step, 2. sparse GCN layer == dense formula,
3. permutation equivariance, 4. what sum/mean/max aggregation can and cannot
tell apart, including two graphs no message-passing network can distinguish."""
import torch

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False)


def to_directed(edges):
    """Undirected edge list [(u, v), ...] -> (2, E) tensor holding both directions."""
    e = torch.tensor(edges, dtype=torch.long).t()
    return torch.cat([e, e.flip(0)], dim=1)


def aggregate(msg, dst, n, how="sum"):
    """Combine the messages arriving at each node. msg: (E, d), dst: (E,)."""
    out = torch.zeros(n, msg.shape[1])
    if how == "max":
        idx = dst[:, None].expand_as(msg)
        return out.scatter_reduce_(0, idx, msg, "amax", include_self=False)  # no edges: 0
    out = out.index_add_(0, dst, msg)
    if how == "mean":
        deg = torch.zeros(n).index_add_(0, dst, torch.ones(len(dst)))
        out = out / deg.clamp(min=1)[:, None]
    return out


def propagate(x, edge_index, how="sum", weight=None):
    """One round: every edge (src -> dst) sends x[src] (optionally scaled)."""
    src, dst = edge_index
    msg = x[src] if weight is None else x[src] * weight[:, None]
    return aggregate(msg, dst, x.shape[0], how)


def gcn_edges(edge_index, n):
    """Add self-loops and the symmetric weights 1/sqrt(deg_u deg_v) of
    Kipf and Welling (2017), where deg counts the self-loop."""
    loops = torch.arange(n).repeat(2, 1)
    ei = torch.cat([edge_index, loops], dim=1)
    deg = torch.zeros(n).index_add_(0, ei[1], torch.ones(ei.shape[1]))
    w = deg[ei[0]].rsqrt() * deg[ei[1]].rsqrt()
    return ei, w


class GCNLayer(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = torch.nn.Linear(d_in, d_out, bias=False)

    def forward(self, x, ei, w):
        return propagate(self.lin(x), ei, "sum", w)


# 1. A path graph 0-1-2-3 with scalar features: watch one step by hand.
ei = to_directed([(0, 1), (1, 2), (2, 3)])
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
print("features         ", x.squeeze().tolist())
print("sum of neighbours ", propagate(x, ei, "sum").squeeze().tolist())
print("mean of neighbours", propagate(x, ei, "mean").squeeze().tolist())
print("max of neighbours ", propagate(x, ei, "max").squeeze().tolist())

# 2. Sparse GCN layer versus the dense formula  D^-1/2 (A + I) D^-1/2 X W.
n, d = 50, 8
pairs = torch.combinations(torch.arange(n), 2)
edges = pairs[torch.rand(len(pairs)) < 0.08].tolist()
ei = to_directed(edges)
x = torch.randn(n, d)
layer = GCNLayer(d, 4)
gei, w = gcn_edges(ei, n)
sparse = layer(x, gei, w)
A = torch.zeros(n, n); A[ei[0], ei[1]] = 1
A_hat = A + torch.eye(n)
D_inv_sqrt = torch.diag(A_hat.sum(1).rsqrt())
dense = D_inv_sqrt @ A_hat @ D_inv_sqrt @ layer.lin(x)
print(f"\nrandom graph: {n} nodes, {len(edges)} edges")
print(f"max |sparse - dense| = {(sparse - dense).abs().max():.2e}")

# 3. Relabel the nodes with a permutation: outputs must be permuted the same way.
perm = torch.randperm(n)
inv = torch.empty_like(perm); inv[perm] = torch.arange(n)   # old id -> new id
gei_p, w_p = gcn_edges(inv[ei], n)
out_p = layer(x[perm], gei_p, w_p)
print(f"max |f(Px) - P f(x)| = {(out_p - sparse[perm]).abs().max():.2e}")

# 4a. Multisets. Node 0 has neighbours with features {1, 1}; node 3 has {1}.
#     Mean and max see the same thing; only sum can tell them apart.
ei = to_directed([(0, 1), (0, 2), (3, 4)])
x = torch.tensor([[0.0], [1.0], [1.0], [0.0], [1.0]])
for how in ["sum", "mean", "max"]:
    h = propagate(x, ei, how)
    print(f"{how:>4}: node 0 -> {h[0].item():.1f}, node 3 -> {h[3].item():.1f}")

# 4b. Two triangles versus one hexagon: 6 nodes, every node has degree 2.
#     With identical starting features, every node receives identical messages
#     at every round, so a message-passing network (here a small random GIN-like
#     net with sum readout) gives both graphs the same embedding.
two_triangles = to_directed([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
hexagon = to_directed([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])
mlps = [torch.nn.Sequential(torch.nn.Linear(d_in, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16))
        for d_in in [1, 16, 16]]


def graph_embedding(ei, x):
    h = x
    for mlp in mlps:
        h = mlp(h + propagate(h, ei, "sum"))       # GIN update with epsilon = 0
    return h.sum(0)                                 # sum readout


ones = torch.ones(6, 1)
gap = (graph_embedding(two_triangles, ones) - graph_embedding(hexagon, ones)).abs().max()
print(f"\nconstant features: max embedding difference = {gap.item():.2e}")
# Give each node a feature that message passing cannot compute itself:
# whether it lies on a triangle (here, the number of closed 3-walks / 2).
def triangles(ei, n):
    A = torch.zeros(n, n); A[ei[0], ei[1]] = 1
    return (torch.linalg.matrix_power(A, 3).diagonal() / 2)[:, None]


gap = (graph_embedding(two_triangles, triangles(two_triangles, 6))
       - graph_embedding(hexagon, triangles(hexagon, 6))).abs().max()
print(f"triangle-count features: max embedding difference = {gap.item():.2e}")
