"""Exact versus approximate nearest-neighbor search, with an IVF index from scratch.

Exact search scores the query against every vector: one matrix product.
An inverted-file (IVF) index clusters the database with k-means and, at
query time, scans only the vectors in the `nprobe` clusters whose centroids
are closest to the query. Fewer vectors scanned means faster search, at the
price of sometimes missing a true neighbor. We measure that trade-off as
recall@10 and show that it depends on the data having cluster structure.
"""
import time

import numpy as np
from sklearn.cluster import KMeans

rng = np.random.default_rng(0)
N, D, NQ, K = 100_000, 64, 200, 10


def normalize(M):
    return (M / np.linalg.norm(M, axis=1, keepdims=True)).astype(np.float32)


def clustered_data(n):
    # 500 "topics"; each vector is a topic center plus noise.
    centers = rng.normal(size=(500, D))
    return normalize(centers[rng.integers(0, 500, n)] + 0.6 * rng.normal(size=(n, D)))


def exact_topk(db, q, k=K):
    scores = q @ db.T                                    # (NQ, N) cosine scores
    top = np.argpartition(-scores, k, axis=1)[:, :k]     # unordered top k, O(N)
    return top


class IVF:
    def __init__(self, db, nlist):
        km = KMeans(nlist, n_init=1, max_iter=20, random_state=0)
        km.fit(db[rng.choice(len(db), 20_000, replace=False)])  # train on a sample
        self.centroids = km.cluster_centers_.astype(np.float32)
        assign = np.argmax(db @ self.centroids.T, axis=1)
        self.lists = [np.flatnonzero(assign == c) for c in range(nlist)]
        # Store each list's vectors contiguously, as real IVF indexes do.
        self.vecs = [db[ids] for ids in self.lists]

    def search(self, q, nprobe, k=K):
        out, scanned = [], 0
        near = np.argsort(-(q @ self.centroids.T), axis=1)[:, :nprobe]
        for qi, cl in zip(q, near):
            cand = np.concatenate([self.lists[c] for c in cl])
            s = np.concatenate([self.vecs[c] @ qi for c in cl])
            scanned += len(cand)
            out.append(cand[np.argpartition(-s, k)[:k]])
        return out, scanned / len(q)


def recall(found, truth):
    return np.mean([len(set(f) & set(t)) / K for f, t in zip(found, truth)])


for name, db in (("clustered data", clustered_data(N)),
                 ("structureless Gaussian data", normalize(rng.normal(size=(N, D))))):
    q = normalize(db[rng.choice(N, NQ)] + 0.1 * rng.normal(size=(NQ, D)))
    t0 = time.perf_counter()
    truth = exact_topk(db, q)
    t_exact = (time.perf_counter() - t0) / NQ * 1e3
    t0 = time.perf_counter()
    index = IVF(db, nlist=256)
    t_build = time.perf_counter() - t0
    print(f"\n{name}: N={N:,}, d={D}; exact search {t_exact:.3f} ms/query; "
          f"IVF build {t_build:.1f} s")
    print(f"  {'nprobe':>6} {'scanned':>9} {'recall@10':>10} {'ms/query':>9}")
    for nprobe in (1, 4, 16, 64):
        t0 = time.perf_counter()
        found, scanned = index.search(q, nprobe)
        ms = (time.perf_counter() - t0) / NQ * 1e3
        print(f"  {nprobe:>6} {scanned / N:>9.1%} {recall(found, truth):>10.3f} {ms:>9.3f}")
