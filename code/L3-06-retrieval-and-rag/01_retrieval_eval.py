"""Compare BM25, a dense (LSA) retriever and their hybrid (RRF) on the bundled
corpus, overall and by query type, with a paired bootstrap on MRR differences."""
import numpy as np
from rag_lib import BM25, LSA, chunk_by_section, is_relevant, load_docs, load_queries, rank_metrics, rrf

docs = load_docs()
chunks = chunk_by_section(docs)
queries = [q for q in load_queries() if q["answer"]]
bm25, lsa = BM25(chunks), LSA(chunks, dim=32)
print(f"{len(docs)} documents, {len(chunks)} section chunks, {len(queries)} answerable queries")

methods = {
    "BM25": lambda q: bm25.scores(q),
    "dense (LSA-32)": lambda q: lsa.scores(q),
    "hybrid (RRF)": lambda q: rrf([bm25.scores(q), lsa.scores(q)]),
}
per_query = {}
for name, f in methods.items():
    rows = []
    for q in queries:
        rel = np.array([is_relevant(c, q) for c in chunks])
        rows.append(rank_metrics(f(q["question"]), rel))
    per_query[name] = rows

types = ["lexical", "paraphrase", "identifier"]
print("\nmethod           R@1   R@3   R@5  MRR@10 nDCG@10 | MRR by type: "
      + "  ".join(f"{t}({sum(q['type'] == t for q in queries)})" for t in types))
for name, rows in per_query.items():
    m = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
    by_type = [np.mean([r["MRR@10"] for r, q in zip(rows, queries) if q["type"] == t]) for t in types]
    print(f"{name:15s} {m['R@1']:.3f} {m['R@3']:.3f} {m['R@5']:.3f}  {m['MRR@10']:.3f}  "
          f"{m['nDCG@10']:.3f}  |  " + "      ".join(f"{v:.3f}" for v in by_type))

# Paired bootstrap over queries: is the hybrid really better than BM25 here?
rng = np.random.default_rng(0)
a = np.array([r["MRR@10"] for r in per_query["hybrid (RRF)"]])
for other in ["BM25", "dense (LSA-32)"]:
    b = np.array([r["MRR@10"] for r in per_query[other]])
    boot = [(a - b)[rng.integers(0, len(a), len(a))].mean() for _ in range(10_000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"hybrid - {other:15s} MRR diff {np.mean(a - b):+.3f}, 95% bootstrap CI [{lo:+.3f}, {hi:+.3f}]; "
          f"hybrid better on {(a > b).sum()}, worse on {(a < b).sum()} queries")

print("\nQueries where BM25 puts the answer below rank 2:")
for r_b, r_d, q in zip(per_query["BM25"], per_query["dense (LSA-32)"], queries):
    if r_b["MRR@10"] < 0.5:
        top = chunks[int(np.argmax(bm25.scores(q["question"])))].cid
        print(f"  {q['id']} [{q['type']:10s}] BM25 MRR {r_b['MRR@10']:.2f}, dense MRR "
              f"{r_d['MRR@10']:.2f}  top BM25 hit: {top:24s} | {q['question']}")

# The dimension of the dense model is a hyperparameter. Choosing it by looking at
# these numbers would be tuning on the test queries, so they are shown, not used.
print("\nMRR@10 of dense retrieval by LSA dimension:")
for dim in [8, 16, 32, 48, 64]:
    m = LSA(chunks, dim=dim)
    mrr = np.mean([rank_metrics(m.scores(q["question"]),
                                np.array([is_relevant(c, q) for c in chunks]))["MRR@10"] for q in queries])
    print(f"  dim {dim:2d}: {mrr:.3f}")
