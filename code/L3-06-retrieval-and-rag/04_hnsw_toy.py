"""A compact HNSW (hierarchical navigable small world) index in pure Python and
NumPy. Measures recall@10 against exact search and how many distances each query
computes, as ef_search grows, with two ways of choosing each node's links:
the M closest candidates, or the diversity heuristic of Malkov and Yashunin."""
import heapq
import math
import time
import numpy as np


class HNSW:
    def __init__(self, X, M=8, ef_construction=64, heuristic=False, seed=0):
        self.X, self.M, self.efc, self.heuristic = X, M, ef_construction, heuristic
        self.rng = np.random.default_rng(seed)
        self.mL = 1 / math.log(M)              # level multiplier: P(level >= l) = M^-l
        self.graph = []                        # graph[l][node] = list of neighbor ids
        self.entry, self.ndist = None, 0
        for i in range(len(X)):
            self._insert(i)

    def _d(self, q, ids):
        self.ndist += len(ids)
        return 1.0 - self.X[ids] @ q           # cosine distance on unit vectors

    def _search_layer(self, q, entries, ef, layer):
        """Best-first search keeping the ef closest nodes found so far."""
        d0 = self._d(q, entries)
        visited = set(entries)
        cand = [(d, e) for d, e in zip(d0, entries)]          # min-heap by distance
        best = [(-d, e) for d, e in zip(d0, entries)]         # max-heap of results
        heapq.heapify(cand)
        heapq.heapify(best)
        while cand:
            d, c = heapq.heappop(cand)
            if d > -best[0][0]:
                break                                         # nothing closer is reachable
            nbrs = [n for n in self.graph[layer].get(c, []) if n not in visited]
            if not nbrs:
                continue
            visited.update(nbrs)
            for dn, n in zip(self._d(q, nbrs), nbrs):
                if len(best) < ef or dn < -best[0][0]:
                    heapq.heappush(cand, (dn, n))
                    heapq.heappush(best, (-dn, n))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted((-d, n) for d, n in best)

    def _select(self, found):
        """found: sorted (dist, id). Closest-M, or keep a candidate only if it is
        closer to the new node than to every neighbor already kept (diversity)."""
        if not self.heuristic:
            return [n for _, n in found[: self.M]]
        kept = []
        for d, n in found:
            if len(kept) == self.M:
                break
            if not kept or d < self._d(self.X[n], kept).min():
                kept.append(n)
        return kept

    def _insert(self, i):
        level = int(-math.log(1 - self.rng.random()) * self.mL)
        while len(self.graph) <= level:
            self.graph.append({})
        if self.entry is None:
            for l in range(level + 1):
                self.graph[l][i] = []
            self.entry = i
            return
        q, ep = self.X[i], [self.entry]
        top = max(l for l in range(len(self.graph)) if self.entry in self.graph[l])
        for l in range(top, level, -1):                       # greedy descent, ef = 1
            ep = [self._search_layer(q, ep, 1, l)[0][1]]
        for l in range(min(level, top), -1, -1):
            found = self._search_layer(q, ep, self.efc, l)
            cap = 2 * self.M if l == 0 else self.M
            self.graph[l][i] = self._select(found)
            for n in self.graph[l][i]:                        # links go both ways
                self.graph[l][n].append(i)
                if len(self.graph[l][n]) > cap:               # prune to the closest
                    nb = self.graph[l][n]
                    self.graph[l][n] = [nb[j] for j in np.argsort(self._d(self.X[n], nb))[:cap]]
            ep = [n for _, n in found]
        for l in range(level + 1):
            self.graph[l].setdefault(i, [])
        if level > top:
            self.entry = i

    def query(self, q, k, ef):
        ep = [self.entry]
        top = max(l for l in range(len(self.graph)) if self.entry in self.graph[l])
        for l in range(top, 0, -1):
            ep = [self._search_layer(q, ep, 1, l)[0][1]]
        return [n for _, n in self._search_layer(q, ep, max(ef, k), 0)[:k]]


def unit(A):
    return A / np.linalg.norm(A, axis=1, keepdims=True)


rng = np.random.default_rng(0)
N, D, NQ = 5000, 32, 200
centers = rng.normal(size=(100, D))
datasets = {
    "clustered (100 topics)": unit(centers[rng.integers(0, 100, N + NQ)] + 0.5 * rng.normal(size=(N + NQ, D))),
    "structureless Gaussian": unit(rng.normal(size=(N + NQ, D))),
}
for name, V in datasets.items():
    X, Q = V[:N], V[N:]
    truth = np.argsort(-(Q @ X.T), axis=1)[:, :10]          # exact top-10
    print(f"\n{name}, N={N}, d={D}. Exact search computes {N} distances per query.")
    print("  links      build s  ef_search: recall@10 (distances per query)")
    for heuristic in [False, True]:
        t0 = time.time()
        index = HNSW(X, heuristic=heuristic)
        row = f"  {'diverse' if heuristic else 'closest M':9s}  {time.time() - t0:6.1f}  "
        for ef in [10, 40, 160]:
            index.ndist, rec = 0, []
            for qi in range(NQ):
                found = index.query(Q[qi], 10, ef)
                rec.append(len(set(found) & set(truth[qi])) / 10)
            row += f"  {ef}: {np.mean(rec):.3f} ({index.ndist / NQ:4.0f})"
        print(row)
print(f"Nodes per layer (same for every index, set by the random levels): "
      f"{[len(g) for g in index.graph]}")
