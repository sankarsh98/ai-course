"""How chunking changes retrieval: fixed word windows of several sizes, with and
without overlap, versus one chunk per section. Relevance = the chunk contains the
answer string, so an answer cut in half by a boundary counts as unfindable.
Recall is also measured at a fixed budget of words handed to the generator."""
import numpy as np
from rag_lib import BM25, chunk_by_section, chunk_fixed, is_relevant, load_docs, load_queries, rank_metrics

docs = load_docs()
queries = [q for q in load_queries() if q["answer"]]
schemes = [(f"fixed {s} words, overlap {o}", chunk_fixed(docs, s, o))
           for s in [20, 50, 100, 200] for o in [0, s // 4]]
schemes.append(("by section", chunk_by_section(docs)))


def recall_at_budget(scores, rel, lengths, budget):
    """Take chunks in rank order while they fit in `budget` words (at least one)."""
    used = 0
    for i in np.argsort(-scores, kind="stable"):
        if used and used + lengths[i] > budget:
            return 0.0
        used += lengths[i]
        if rel[i]:
            return 1.0
    return 0.0


print(f"{'chunking':27s} chunks words/chunk split  R@1   R@3  words@top3  R@150w  R@300w")
for name, chunks in schemes:
    bm25 = BM25(chunks)
    lengths = np.array([len(c.text.split()) for c in chunks])
    res, split, ctx, b150, b300 = [], 0, [], [], []
    for q in queries:
        rel = np.array([is_relevant(c, q) for c in chunks])
        split += not rel.any()            # no chunk holds the whole answer string
        s = bm25.scores(q["question"])
        res.append(rank_metrics(s, rel))
        ctx.append(lengths[np.argsort(-s)[:3]].sum())
        b150.append(recall_at_budget(s, rel, lengths, 150))
        b300.append(recall_at_budget(s, rel, lengths, 300))
    print(f"{name:27s} {len(chunks):6d} {lengths.mean():11.0f} {split:5d}  "
          f"{np.mean([r['R@1'] for r in res]):.3f} {np.mean([r['R@3'] for r in res]):.3f} "
          f"{np.mean(ctx):9.0f}   {np.mean(b150):.3f}   {np.mean(b300):.3f}")
