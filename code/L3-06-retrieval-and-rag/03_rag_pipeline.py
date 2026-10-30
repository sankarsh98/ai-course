"""A complete small RAG pipeline without external APIs: hybrid retrieval of 10
candidates, a learned reranker (logistic regression, leave-one-query-out), an
extractive answer with a citation, and abstention when the evidence is weak."""
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from rag_lib import (BM25, LSA, chunk_by_section, is_relevant, load_docs, load_queries,
                     norm, rrf, tokenize)

docs = load_docs()
chunks = chunk_by_section(docs)
queries = load_queries()                       # 48 answerable + 8 unanswerable
bm25, lsa = BM25(chunks), LSA(chunks, dim=32)
K = 10


def features(q, cand, s_bm, s_lsa):
    """Cheap query-chunk features. A production reranker would be a cross-encoder."""
    qt = set(tokenize(q))
    w = {t: bm25.idf.get(t, 0.0) for t in qt}
    rows = []
    for rank, i in enumerate(cand):
        ct, ht = set(tokenize(chunks[i].text)), set(tokenize(chunks[i].section))
        rows.append([s_bm[i], s_bm[i] / (s_bm[cand].max() + 1e-9), s_lsa[i],
                     sum(w[t] for t in qt & ct) / (sum(w.values()) + 1e-9),   # idf-weighted coverage
                     sum(w[t] for t in qt & ht) / (sum(w.values()) + 1e-9),   # coverage of the heading
                     1.0 / (rank + 1), float(chunks[i].current)])
    return np.array(rows)


# ---- first stage for every query, and the training data for the reranker ----
data = []
for q in queries:
    s_bm, s_lsa = bm25.scores(q["question"]), lsa.scores(q["question"])
    cand = np.argsort(-rrf([s_bm, s_lsa]), kind="stable")[:K]
    X = features(q["question"], cand, s_bm, s_lsa)
    y = np.array([is_relevant(chunks[i], q) for i in cand], dtype=int)
    data.append((q, cand, X, y))

# ---- leave-one-query-out: each query is scored by a model that never saw it ----
probs = []
for k in range(len(data)):
    Xtr = np.vstack([d[2] for j, d in enumerate(data) if j != k])
    ytr = np.concatenate([d[3] for j, d in enumerate(data) if j != k])
    model = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr)
    probs.append(model.predict_proba(data[k][2])[:, 1])


def best_sentence(q, chunk):
    """Extractive 'generator': the sentence with the largest idf-weighted overlap."""
    qt = set(tokenize(q))
    body = chunk.text.split(". ", 1)[1]                    # drop the "Title > Section" prefix
    sents = re.split(r"(?<=[.!?])\s+", body)
    score = [sum(bm25.idf.get(t, 0) for t in qt & set(tokenize(s))) for s in sents]
    return sents[int(np.argmax(score))]


def answer(k, threshold):
    q, cand, X, y = data[k]
    order = np.argsort(-probs[k], kind="stable")
    top = cand[order[0]]
    if probs[k][order[0]] < threshold:
        return None, top
    return best_sentence(q["question"], chunks[top]), top


ans = [d[0]["answer"] != "" for d in data]
print("Retrieval, 48 answerable queries:     first stage R@1   after rerank R@1   R@10 (ceiling)")
r1_first = np.mean([d[3][0] for d, a in zip(data, ans) if a])
r1_rerank = np.mean([d[3][np.argmax(p)] for d, p, a in zip(data, probs, ans) if a])
r10 = np.mean([d[3].any() for d, a in zip(data, ans) if a])
print(f"{'':38s} {r1_first:.3f}              {r1_rerank:.3f}             {r10:.3f}")
travel = [k for k, d in enumerate(data) if d[0]["doc"] == "travel-policy-2025"]
stale = sum(not chunks[data[k][1][0]].current for k in travel)
stale_rr = sum(not chunks[data[k][1][np.argmax(probs[k])]].current for k in travel)
print(f"Travel questions answered from the superseded 2023 policy: first stage {stale} of "
      f"{len(travel)}, after rerank {stale_rr} of {len(travel)}")

print("\nOf 48 answerable:  correct  wrong: bad chunk  wrong: bad sentence  abstained"
      "  |  abstained on 8 unanswerable")
for th in [0.0, 0.2, 0.3, 0.5, 0.7]:
    c = w_ret = w_ext = ab = ab_u = 0
    for k, (q, *_rest) in enumerate(data):
        sent, top = answer(k, th)
        if not q["answer"]:
            ab_u += sent is None
        elif sent is None:
            ab += 1
        elif norm(q["answer"]) in norm(chunks[top].section + ". " + sent):
            c += 1
        elif is_relevant(chunks[top], q):     # right evidence, wrong sentence picked
            w_ext += 1
        else:                                  # retrieval put a wrong chunk first
            w_ret += 1
    print(f"threshold {th:.1f}:     {c:3d}        {w_ret:3d}              {w_ext:3d}           "
          f"{ab:3d}       |   {ab_u}")

print("\nExamples at threshold 0.3:")
for qid in ["q10", "q21", "q03", "q28", "q52"]:
    k = next(i for i, d in enumerate(data) if d[0]["id"] == qid)
    sent, top = answer(k, 0.3)
    c = chunks[top]
    print(f"  Q: {data[k][0]['question']}")
    print(f"  A: {sent} [{c.cid}: {docs[c.doc]['title']} > {c.section}]" if sent else
          f"  A: Not found in the documents. (best candidate {c.cid}, p={probs[k].max():.2f})")

# What an LLM-based generator would receive instead of the extractive step:
k = next(i for i, d in enumerate(data) if d[0]["id"] == "q10")
top3 = [data[k][1][i] for i in np.argsort(-probs[k])[:3]]
context = "\n".join(f"[{chunks[i].cid}] {chunks[i].text}" for i in top3)
print("\nPrompt for an LLM generator (not run here):\n" + "-" * 60)
print("Answer the question using only the sources below. Cite the source id in brackets\n"
      "after each claim. If the sources do not contain the answer, say so.\n\n"
      f"Sources:\n{context}\n\nQuestion: {data[k][0]['question']}\nAnswer:")
