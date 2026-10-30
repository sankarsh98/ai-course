---
title: "Retrieval and RAG"
---

> **Level 3 · Module 6** · Prerequisites: L0-02, L0-04, L1-01, L2-06 · Time: 7 to 9 hours · Code: `code/L3-06-retrieval-and-rag/`

## What you will be able to do

- Implement BM25 and a dense retriever, and explain when each fails.
- Combine retrievers with reciprocal rank fusion, and explain how HNSW finds neighbors without scanning everything.
- Choose a chunking scheme by measuring retrieval at a fixed context budget.
- Build a retrieve, rerank, answer and cite pipeline that abstains when evidence is weak.
- Evaluate a RAG system by separating retrieval errors from generation errors, with honest uncertainty on small test sets.
- Weigh long context against retrieval using the published evidence.

## 1. Why retrieval, and what RAG is

A language model's knowledge is frozen at training time (L3-01). It does not know your company's policies, and asked anyway it may answer fluently and wrongly. **Retrieval-augmented generation (RAG)** fetches relevant text at question time and puts it in the prompt, so the model answers from evidence it can cite. Lewis et al. coined the name for a seq2seq generator ("parametric memory") combined with a dense index of Wikipedia ("non-parametric memory") [Lewis et al., 2020]; most systems today pair an off-the-shelf retriever with a general LLM. Each of the four stages fails in its own way:

1. **Index:** split documents into chunks and build search structures (sections 2 to 6).
2. **Retrieve:** find candidate chunks for the query, cheaply (sections 2 to 5).
3. **Rerank:** reorder the candidates with a more expensive model (section 7).
4. **Generate:** answer from the top chunks, cite them, or decline (section 8).

Retrieval is also a tool an agent can call; for the agent loop and tool design see the AI Systems course (https://sankarsh98.github.io/ai-systems/agents/the-loop) and L3-07.

## 2. Sparse retrieval and BM25

The classic way to find matching documents is to score shared words, weighting rare words more. **BM25**, from the probabilistic relevance framework [Robertson & Zaragoza, 2009], scores a document $D$ for a query $Q$ as

$$\text{BM25}(D, Q) = \sum_{t \in Q} \text{idf}(t) \cdot \frac{f(t, D)\,(k_1 + 1)}{f(t, D) + k_1\big(1 - b + b\, \frac{|D|}{\text{avgdl}}\big)}, \qquad \text{idf}(t) = \ln\Big(1 + \frac{N - n_t + 0.5}{n_t + 0.5}\Big),$$

where $f(t, D)$ is the count of term $t$ in $D$, $|D|$ the document length, avgdl the average length, $N$ the number of documents and $n_t$ the number containing $t$. This idf form, which never goes negative, is the one Apache Lucene uses, with defaults $k_1 = 1.2$ and $b = 0.75$ (Lucene `BM25Similarity` documentation).

**idf** makes a match on "TG-204" (in 1 of 65 chunks, idf 3.8) count far more than a common word. The **tf** term saturates at $k_1 + 1$, so 20 mentions are not 20 times more relevant. **Length normalization** ($b$) discounts long documents, which match more words by chance. BM25 needs no training, handles identifiers, names and numbers, and is cheap to update. It fails on **vocabulary mismatch**: "holiday" does not match "annual leave". Tokenization matters too; ours stems crudely and keeps codes like `tg-204` as one token.

## 3. Dense retrieval

A **dense retriever** embeds queries and documents into the same vector space and scores by dot product or cosine (L0-02, L2-06). A **bi-encoder** encodes the two sides separately, so document vectors are computed once at indexing time and a query costs one encoder pass plus a nearest-neighbor search. Dense Passage Retrieval trained a pair of BERT encoders contrastively on question-passage pairs with in-batch negatives (the setup of L2-06's contrastive section) and beat a strong Lucene BM25 by 9 to 19 percentage points in top-20 passage retrieval accuracy on open-domain QA benchmarks [Karpukhin et al., 2020].

The advantage is not universal. BEIR evaluated 10 retrievers **zero-shot** on 18 datasets and found BM25 a robust baseline; dense models often trailed, and reranking and late-interaction models did best at a higher compute cost [Thakur et al., 2021]. Dense models tend to handle paraphrase better and exact strings unseen in training (part numbers, error codes) worse. MTEB found that no text embedding method dominated across all tasks (Muennighoff et al., 2022, arXiv:2210.07316). Models change quickly; test on your own data.

**Our stand-in.** This course downloads no models, so script 1 uses **latent semantic analysis (LSA)** (Deerwester et al., 1990): TF-IDF vectors reduced by truncated SVD (L0-02, L1-04) to 32 dimensions. LSA maps co-occurring terms to nearby directions, so it matches some paraphrases, but it learns only from our 65 chunks. Read its results as "dense retrieval trained on almost nothing", not as a verdict on modern embeddings. The `LSA` class is the only thing to replace to try a real encoder.

## 4. Hybrid retrieval and rank fusion

Sparse and dense retrievers make different mistakes, so combining them can help. Their scores live on different scales (BM25 is unbounded, cosine is in $[-1, 1]$), so averaging raw scores is fragile. **Reciprocal rank fusion (RRF)** uses only ranks [Cormack et al., 2009]:

$$\text{RRF}(d) = \sum_{\text{retrievers } i} \frac{1}{k + \text{rank}_i(d)}, \qquad k = 60.$$

A document ranked first by one retriever and tenth by another scores $1/61 + 1/70$. The constant $k$ damps the influence of top ranks, so one retriever's overconfidence cannot dominate.

Script 1 measures BM25, LSA and their RRF hybrid on the 48 answerable test questions, where a chunk is relevant if it contains the question's answer string:

| method | R@1 | R@3 | MRR@10 | MRR lexical (16) | MRR paraphrase (26) | MRR identifier (6) |
|---|---|---|---|---|---|---|
| BM25 | 0.583 | 0.875 | 0.721 | 0.906 | 0.562 | 0.917 |
| dense (LSA-32) | 0.625 | 0.854 | 0.738 | 0.896 | 0.605 | 0.889 |
| hybrid (RRF) | 0.562 | 0.875 | 0.712 | 0.865 | 0.577 | 0.889 |

The largest effect is the query type: paraphrased questions score far below lexical ones for every method, which is vocabulary mismatch in numbers. The hybrid did not help here. Its MRR difference from BM25 is $-0.009$ with a 95% paired bootstrap interval of $[-0.038, +0.016]$ (L0-04): no detectable difference on 48 queries. That is expected, because LSA is built from the same word counts as BM25, so their errors are correlated, and fusion helps only when errors are complementary. The dimension sweep at the end of the script (MRR 0.519 at 8 dimensions, 0.764 at 48, 0.713 at 64) shows how easily you could "win" by tuning on the test questions; the script reports it but does not use it.

## 5. Vector indexes: IVF and HNSW

L2-06 showed exact search is fast enough up to about $10^5$ vectors and built an **inverted file (IVF)** index, which scans only the $k$-means clusters nearest the query. **HNSW (hierarchical navigable small world)** graphs are the other main family [Malkov & Yashunin, 2020]. Each vector is a node linked to about $M$ neighbors. Search is greedy: from an entry node, move to whichever neighbor is closest to the query until none is closer, keeping a beam of the `ef` best nodes found. For long jumps, each node gets a random top layer, $P(\text{level} \ge l) = M^{-l}$ (the paper's $m_L = 1/\ln M$), like a skip list; search descends greedily through the sparse upper layers and runs the beam only on the bottom layer. Insertion is a search followed by linking, so the index grows incrementally.

Linking each node to its $M$ closest candidates sounds natural, but on clustered data every list fills with the node's own cluster and the graph splits into islands. The paper's **neighbor-selection heuristic** keeps a candidate only if it is closer to the new node than to every neighbor already kept, which preserves some long-range links. Script 4 implements both on 5,000 vectors of dimension 32:

| data | links | recall@10 at ef=10 | ef=40 | ef=160 | distances per query at ef=160 |
|---|---|---|---|---|---|
| 100 clusters | closest M | 0.667 | 0.807 | 0.965 | 425 |
| 100 clusters | heuristic | 0.982 | 1.000 | 1.000 | 756 |
| structureless | closest M | 0.443 | 0.796 | 0.983 | 1352 |
| structureless | heuristic | 0.473 | 0.799 | 0.986 | 1380 |

On clustered data the heuristic lifts recall at `ef=10` from 0.667 to 0.982 with 124 distance computations instead of 5,000, mirroring the paper's own clustered-data comparison. On structureless data it barely matters and every setting needs more work. Real embeddings are clustered, which is why graph indexes do well on them; still, measure recall on your data. Production libraries add compression, filtering and compiled speed (see the Faiss paper cited in L2-06).

## 6. Chunking

Retrievers score **chunks**, and the generator's context is limited. Small chunks are precise but can split an answer or lose context; large ones waste the generator's budget. Script 2 compares fixed word windows with one chunk per document section, using BM25:

| chunking | chunks | words/chunk | answers split | R@1 | R@3 | R within 150 words | R within 300 words |
|---|---|---|---|---|---|---|---|
| fixed 20, no overlap | 141 | 23 | 8 | 0.250 | 0.458 | 0.542 | 0.562 |
| fixed 50, overlap 12 | 72 | 52 | 0 | 0.521 | 0.812 | 0.750 | 0.854 |
| fixed 200, overlap 50 | 18 | 173 | 0 | 0.771 | 0.938 | 0.771 | 0.771 |
| by section | 65 | 47 | 0 | 0.583 | 0.875 | 0.833 | 0.917 |

First, **R@k across chunk sizes is not a fair comparison**: three 173-word chunks are about 520 words, a sixth of this corpus. At a fixed budget of words for the generator, section chunks win (0.833 within 150 words, 0.917 within 300). Second, boundaries matter: 20-word windows without overlap cut 8 of 48 answers in half; overlap fixes it at the cost of more chunks. Section chunks also carry their title and heading (`Leave Policy > Carrying leave over.`), words the body lacks. Anthropic's "contextual retrieval" write-up (September 2024) takes this further, using an LLM to write a short context for every chunk before indexing; it reports that this cut top-20 retrieval failures by 35% for embeddings, 49% with contextual BM25 added, and 67% with reranking as well, on its own test sets.

## 7. Reranking

First-stage retrievers are cheap because they encode query and document independently. A **cross-encoder** reads them together through one transformer, so every query word attends to every document word: more accurate and far more expensive, so it is applied only to the top candidates. A BERT cross-encoder improved MRR@10 on the MS MARCO passage leaderboard by 27% relative over the previous best (Nogueira and Cho, 2019, arXiv:1901.04085). **Late interaction** (ColBERT) sits in between: it stores a vector per token and scores by summing, for each query token, its maximum similarity to the document's tokens [Khattab & Zaharia, 2020].

Without a cross-encoder, script 3 trains a small **learned reranker**: logistic regression (L1-02) on seven query-candidate features (BM25 and LSA scores, idf-weighted query coverage, heading coverage, first-stage rank, document status). It is evaluated **leave-one-query-out**: each question is scored by a model trained on the other 55. On 10 hybrid candidates per question, R@1 rises from 0.562 to 0.667 (ceiling 0.938).

It also fixes a failure unrelated to wording. The corpus holds a superseded 2023 travel policy beside the current 2025 one, with different numbers. For 4 of 8 travel questions the first stage ranks the stale policy first, since it matches the words equally well. The reranker learned from the labels that current documents answer, and after reranking none of the 8 uses the 2023 policy. A real system should also filter on metadata (status, date, permissions) before ranking: relevance and validity are different questions.

## 8. Generation, grounding and citations

An LLM generator receives a prompt like the one script 3 prints: instructions to answer only from the numbered sources, cite them, and say when they do not contain the answer, followed by the sources and the question. Our generator is **extractive**: it returns the sentence of the top-ranked chunk with the largest idf-weighted overlap with the question, plus a citation to that chunk. That keeps the pipeline runnable without a model and makes every claim traceable, but it cannot combine facts or rephrase.

**Abstention.** Retrieval always returns something, even for questions the corpus cannot answer (the "always a nearest neighbor" failure of L2-06). Script 3 abstains when the reranker's probability for its top candidate is below a threshold:

| threshold | correct (of 48) | wrong: bad chunk | wrong: bad sentence | abstained | abstained on unanswerable (of 8) |
|---|---|---|---|---|---|
| 0.0 | 23 | 16 | 9 | 0 | 0 |
| 0.3 | 19 | 10 | 7 | 12 | 6 |
| 0.5 | 16 | 1 | 6 | 25 | 8 |

With no threshold it answers everything, including all 8 unanswerable questions; the guest Wi-Fi question gets a confident, cited sentence about password length. At 0.5 it declines every unanswerable question and nearly all retrieval errors, but also 25 answerable ones. Where to sit on this curve depends on the cost of a wrong answer versus no answer, and with 8 unanswerable questions the rates are very uncertain.

**Citations are claims too.** A citation says where an answer came from, not that it is right. Check that the cited chunk supports the claim (extractive answers do by construction; LLM answers may cite one thing and say another). Typical failures: the answer is not in the index; it is there but not retrieved; it is outranked by a stale or near-duplicate document; it is in the context but ignored or mixed with pretraining knowledge; or text planted in a document hijacks the model (prompt injection, L3-07, L4-05).

## 9. Evaluating RAG

Evaluate the stages separately, then end to end.

**Retrieval metrics** need relevance labels. With binary labels and a ranked list: **R@k** (here, "success at $k$") is the fraction of queries with a relevant chunk in the top $k$; **MRR@10** averages $1/\text{rank}$ of the first relevant result (0 if none in the top 10); **nDCG@10** sums $\text{rel}_i / \log_2(i + 1)$ over the top 10 and divides by the best possible value, rewarding relevant results near the top. Labels are expensive. Defining relevance as "contains the answer string", as these scripts do, makes labels cheap and independent of chunking, but it misses answers phrased differently.

**End-to-end metrics** ask whether the answer is correct and supported. Script 3 splits errors into "wrong top chunk" (retrieval) and "right chunk, wrong sentence" (generation): 16 versus 9 at threshold 0, so a better generator alone could fix at most 9 of 25 errors. That decomposition tells you where to work. Without reference answers, LLM judges can rate context focus, faithfulness and answer relevance; Ragas packages such metrics [Es et al., 2023]. Judges have biases (L4-03), so check them against human labels.

**Small test sets.** With 56 questions, differences of a few MRR points are noise; use a paired bootstrap at minimum (script 1). Write questions the way users ask, including paraphrases, identifiers and unanswerable ones. Capstone 3 builds such a set for your own notes.

## 10. Long context versus retrieval

As of October 2026, advertised context windows reach millions of tokens; Meta, for example, announced a 10 million token window for Llama 4 Scout in April 2025 ("The Llama 4 herd", Meta AI blog). Why not put everything in the prompt?

- **Advertised length is not usable length.** RULER found that of 17 long-context models claiming at least 32K tokens, only about half performed satisfactorily at 32K on its tasks (Hsieh et al., 2024, arXiv:2404.06654). Liu et al. found accuracy often highest when the relevant passage is at the start or end of the context and lower in the middle [Liu et al., 2024]. Chroma's "Context Rot" report (Hong, Troynikov and Huber, July 2025) tested 18 models and found performance degraded with input length even on simple tasks, faster when the relevant text is less semantically similar to the question and when similar but irrelevant distractors are present.
- **Cost.** Every token in the prompt is processed for every question, and attention and KV-cache memory grow with context length (L2-05; the AI Systems course explains the KV cache at https://sankarsh98.github.io/ai-systems/inference/kv-cache).
- **Freshness, permissions and citations** are easier with an index.

The evidence is mixed. Li et al. compared the two on public benchmarks with three LLMs of 2024 and found long context better on average when resources allowed, RAG much cheaper, and a router that lets the model choose per query close to long-context quality at much lower cost [Li et al., 2024]. LaRA (Li et al., ICML 2025, arXiv:2502.09977) concluded across 11 models that neither approach is always better: the answer depends on model strength, context length, task type and retrieval quality. Anthropic's contextual retrieval write-up suggests that a knowledge base under about 200,000 tokens can simply go in the prompt, with caching. Measure both on your questions; for large, changing or permissioned collections, retrieval remains necessary.

## Hands-on

The scripts are in `code/L3-06-retrieval-and-rag/` and need NumPy and scikit-learn; there are no external APIs and nothing is downloaded. The corpus in `corpus/` is **fictional, written for this course**: 12 Markdown documents (about 3,000 words) forming the knowledge base of an invented instrument maker, "Brindlewood" (policies, IT security, a VPN guide, a product manual, firmware notes, warranty, support levels, onboarding, facilities, data retention), including a deliberately superseded 2023 travel policy. `corpus/queries.tsv` has 56 questions: 16 lexical, 26 paraphrased, 6 with identifiers, and 8 unanswerable. Each answerable one has a gold document and an answer string found in exactly one section. Reuse it freely.

All scripts run in under 12 seconds (outputs from a 24-core Linux machine; HNSW timings will vary).

### Shared library: `rag_lib.py`

```python
"""Shared pieces for the L3-06 scripts: corpus loading, chunking, BM25, a dense
retriever (LSA: TF-IDF + truncated SVD), reciprocal rank fusion, and metrics.

The corpus in corpus/ is fictional, written for this course (an internal
knowledge base of a made-up instrument maker, "Brindlewood"). It has 12
Markdown documents and 56 test questions with answer strings."""
import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

CORPUS = Path(__file__).parent / "corpus"
STOP = set("""a an and are as at be by can do does for from get has have how i if in is it
its me must my of on or our should so that the their then there this to was we what
when where which who will with you your am any all not no after before up""".split())


@dataclass
class Chunk:
    cid: str        # e.g. "leave-policy#3"
    doc: str
    section: str
    text: str       # what the retriever indexes and the generator sees
    current: bool   # document status is "current" (not superseded)


def stem(w):
    """A crude suffix stripper: enough to match 'reimbursed' and 'reimburse'."""
    for suf, rep in [("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")]:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: len(w) - len(suf)] + rep
    return w


def tokenize(text):
    # keep codes like tg-204, e-47, 0.45 as single tokens
    words = re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", text.lower())
    return [stem(w) for w in words if w not in STOP]


def load_docs():
    docs = {}
    for path in sorted((CORPUS / "docs").glob("*.md")):
        title, sections, cur = None, [], None
        status = "current" in path.read_text().split("\n")[2]   # the "Status:" line
        for line in path.read_text().splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                cur = [line[3:].strip(), []]
                sections.append(cur)
            elif cur and line.strip():
                cur[1].append(line.strip())
        docs[path.stem] = dict(title=title, current=status,
                               sections=[(h, " ".join(b)) for h, b in sections])
    return docs


def chunk_by_section(docs):
    """One chunk per '##' section, prefixed with document title and section heading."""
    out = []
    for name, d in docs.items():
        for i, (head, body) in enumerate(d["sections"]):
            out.append(Chunk(f"{name}#{i}", name, head, f"{d['title']} > {head}. {body}",
                             d["current"]))
    return out


def chunk_fixed(docs, size, overlap=0):
    """Fixed windows of `size` words over each document, ignoring its structure."""
    out = []
    for name, d in docs.items():
        words = " ".join(f"{h}. {b}" for h, b in d["sections"]).split()
        step = max(1, size - overlap)
        for k, i in enumerate(range(0, max(1, len(words) - overlap), step)):
            out.append(Chunk(f"{name}@{k}", name, f"words {i}-{i + size}",
                             f"{d['title']}: " + " ".join(words[i:i + size]), d["current"]))
    return out


def load_queries():
    with open(CORPUS / "queries.tsv") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def norm(s):
    return " ".join(re.findall(r"[a-z0-9$.\-:]+", s.lower()))


def is_relevant(chunk, query):
    """Relevance is defined by the answer string, so it does not depend on chunking."""
    return bool(query["answer"]) and norm(query["answer"]) in norm(chunk.text)


class BM25:
    """Okapi BM25 with the non-negative idf used by Lucene."""

    def __init__(self, chunks, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c.text)) for c in chunks]
        self.len = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg = self.len.mean()
        df = Counter(t for d in self.docs for t in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - m + 0.5) / (m + 0.5)) for t, m in df.items()}

    def scores(self, query):
        s = np.zeros(len(self.docs))
        norm_len = self.k1 * (1 - self.b + self.b * self.len / self.avg)
        for t in set(tokenize(query)):
            if t not in self.idf:
                continue
            tf = np.array([d.get(t, 0) for d in self.docs], dtype=float)
            s += self.idf[t] * tf * (self.k1 + 1) / (tf + norm_len)
        return s


class LSA:
    """Dense vectors from TF-IDF + truncated SVD, compared by cosine.
    A stand-in for a neural embedding model, which we cannot download here."""

    def __init__(self, chunks, dim=64, seed=0):
        self.vec = TfidfVectorizer(analyzer=tokenize, sublinear_tf=True)
        X = self.vec.fit_transform([c.text for c in chunks])
        self.svd = TruncatedSVD(n_components=min(dim, X.shape[0] - 1), random_state=seed)
        self.E = self._unit(self.svd.fit_transform(X))

    @staticmethod
    def _unit(E):
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    def scores(self, query):
        q = self._unit(self.svd.transform(self.vec.transform([query])))
        return self.E @ q[0]


def rrf(score_lists, k=60):
    """Reciprocal rank fusion: sum over retrievers of 1 / (k + rank)."""
    fused = np.zeros(len(score_lists[0]))
    for s in score_lists:
        ranks = np.empty(len(s))
        ranks[np.argsort(-s, kind="stable")] = np.arange(1, len(s) + 1)
        fused += 1.0 / (k + ranks)
    return fused


def rank_metrics(scores, rel, ks=(1, 3, 5)):
    """Binary relevance metrics for one query. rel: boolean array over chunks."""
    order = np.argsort(-scores, kind="stable")
    hits = rel[order]
    out = {f"R@{k}": float(hits[:k].any()) for k in ks}    # "success at k"
    first = np.flatnonzero(hits[:10])
    out["MRR@10"] = 1.0 / (first[0] + 1) if len(first) else 0.0
    gains = hits[:10] / np.log2(np.arange(2, 12))
    ideal = np.sort(rel)[::-1][:10] / np.log2(np.arange(2, 12))
    out["nDCG@10"] = gains.sum() / ideal.sum() if rel.any() else 0.0
    return out
```

### Script 1: BM25, dense and hybrid retrieval

```python
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
```

```
12 documents, 65 section chunks, 48 answerable queries

method           R@1   R@3   R@5  MRR@10 nDCG@10 | MRR by type: lexical(16)  paraphrase(26)  identifier(6)
BM25            0.583 0.875 0.917  0.721  0.771  |  0.906      0.562      0.917
dense (LSA-32)  0.625 0.854 0.896  0.738  0.782  |  0.896      0.605      0.889
hybrid (RRF)    0.562 0.875 0.896  0.712  0.768  |  0.865      0.577      0.889
hybrid - BM25            MRR diff -0.009, 95% bootstrap CI [-0.038, +0.016]; hybrid better on 3, worse on 5 queries
hybrid - dense (LSA-32)  MRR diff -0.026, 95% bootstrap CI [-0.078, +0.025]; hybrid better on 6, worse on 7 queries

Queries where BM25 puts the answer below rank 2:
  q02 [paraphrase] BM25 MRR 0.25, dense MRR 0.00  top BM25 hit: facilities#3             | How much can I spend on a room when visiting Zurich for work?
  q04 [paraphrase] BM25 MRR 0.00, dense MRR 0.12  top BM25 hit: returns-and-warranty#1   | How long after a business trip do I have to ask for my money back?
  q11 [paraphrase] BM25 MRR 0.33, dense MRR 0.50  top BM25 hit: leave-policy#0           | By when must I take the holiday I brought forward?
  q14 [paraphrase] BM25 MRR 0.00, dense MRR 0.00  top BM25 hit: leave-policy#2           | How much paid time off does a new mother get after having a baby?
  q17 [paraphrase] BM25 MRR 0.00, dense MRR 0.00  top BM25 hit: bw40-firmware-notes#0    | How often am I forced to update my login secret?
  q19 [paraphrase] BM25 MRR 0.33, dense MRR 0.33  top BM25 hit: it-security#4            | My work computer was stolen at the station; how quickly must I tell someone?
  q26 [paraphrase] BM25 MRR 0.33, dense MRR 1.00  top BM25 hit: bw40-user-guide#0        | Can the soil logger survive being dropped in a pond?
  q28 [paraphrase] BM25 MRR 0.00, dense MRR 0.00  top BM25 hit: vpn-troubleshooting#1    | What happens when the device has no space left?
  q40 [paraphrase] BM25 MRR 0.33, dense MRR 0.25  top BM25 hit: support-slas#3           | How quickly will someone reply to an ordinary customer question?
  q44 [paraphrase] BM25 MRR 0.33, dense MRR 0.20  top BM25 hit: it-security#2            | How much does it cost to replace a lost building pass?
  q46 [paraphrase] BM25 MRR 0.20, dense MRR 0.50  top BM25 hit: data-retention#0         | How long do we keep customer measurements stored online?

MRR@10 of dense retrieval by LSA dimension:
  dim  8: 0.519
  dim 16: 0.600
  dim 32: 0.738
  dim 48: 0.764
  dim 64: 0.713
```

Reading the misses: q14 ("a new mother") and q17 ("login secret") fail for every method because no word overlaps with "parental leave" or "password". q26 ("dropped in a pond") is the clearest LSA win (rank 1 against BM25's rank 3). q21 shows the stop-word list mattering: "mean" outweighs the rare code in one chunk, so the status-light section ("two red blinks mean") outranks the TG-204 section.

### Script 2: chunking

```python
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
```

```
chunking                    chunks words/chunk split  R@1   R@3  words@top3  R@150w  R@300w
fixed 20 words, overlap 0      141          23     8  0.250 0.458        70   0.542   0.562
fixed 20 words, overlap 5      184          23     4  0.271 0.479        71   0.542   0.646
fixed 50 words, overlap 0       59          50     6  0.521 0.708       154   0.625   0.771
fixed 50 words, overlap 12      72          52     0  0.521 0.812       157   0.750   0.854
fixed 100 words, overlap 0      31          92     4  0.583 0.792       288   0.604   0.750
fixed 100 words, overlap 25     39          91     0  0.625 0.854       293   0.625   0.812
fixed 200 words, overlap 0      18         156     1  0.708 0.938       510   0.708   0.771
fixed 200 words, overlap 50     18         173     0  0.771 0.938       538   0.771   0.771
by section                      65          47     0  0.583 0.875       155   0.833   0.917
```

### Script 3: the full pipeline

```python
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
```

```
Retrieval, 48 answerable queries:     first stage R@1   after rerank R@1   R@10 (ceiling)
                                       0.562              0.667             0.938
Travel questions answered from the superseded 2023 policy: first stage 4 of 8, after rerank 0 of 8

Of 48 answerable:  correct  wrong: bad chunk  wrong: bad sentence  abstained  |  abstained on 8 unanswerable
threshold 0.0:      23         16                9             0       |   0
threshold 0.2:      21         16                7             4       |   2
threshold 0.3:      19         10                7            12       |   6
threshold 0.5:      16          1                6            25       |   8
threshold 0.7:      12          0                4            32       |   8

Examples at threshold 0.3:
  Q: Can I keep unused holiday for next year?
  A: Up to 5 unused days may be carried over into the next year. [leave-policy#1: Leave Policy > Carrying leave over]
  Q: What does TG-204 mean?
  A: Not found in the documents. (best candidate bw40-user-guide#4, p=0.23)
  Q: What is the daily meal allowance when travelling?
  A: Instead of collecting receipts for food, travellers receive a daily meal allowance of $65 for each full day away from home. [travel-policy-2025#2: Travel and Expenses Policy (2025) > Meals and incidentals]
  Q: What happens when the device has no space left?
  A: Not found in the documents. (best candidate vpn-troubleshooting#1, p=0.24)
  Q: What is the guest Wi-Fi password?
  A: Passwords for company accounts must be at least 14 characters long. [it-security#0: IT Security Standard > Passwords and sign-in]

Prompt for an LLM generator (not run here):
------------------------------------------------------------
Answer the question using only the sources below. Cite the source id in brackets
after each claim. If the sources do not contain the answer, say so.

Sources:
[leave-policy#1] Leave Policy > Carrying leave over. Up to 5 unused days may be carried over into the next year. Carried-over days must be used by 31 March; after that date they expire and are not paid out. Days above the carry-over limit are lost at the end of the year, so plan holidays early.
[leave-policy#0] Leave Policy > Annual leave. Full-time employees receive 25 days of paid annual leave per calendar year, plus public holidays. Part-time employees receive a pro-rata amount based on contracted hours. Leave builds up monthly, so in your first year you can take only the days you have earned so far, unless your manager agrees to an advance.
[bw40-firmware-notes#3] BW-40 Firmware Release Notes > Error codes. Error codes appear in the app next to the logger name. - E-12: sensor disconnected. Check that the probe cable is fully pushed in and the connector is dry. - E-31: reading out of range. Usually caused by a probe that is not in contact with soil; reinstall the probe with firm contact. - E-47: memory full. Download the readings and clear the memory, or turn on overwrite mode. - E-58: firmware update failed. Leave the logger within Bluetooth range and run the update again; the logger keeps the previous firmware until an update succeeds.

Question: Can I keep unused holiday for next year?
Answer:
```

The TG-204 and "no space left" questions are answerable, but the reranker is not confident (p = 0.23 and 0.24), so the pipeline declines at threshold 0.3. Declining is better than the alternative for "no space left": the top candidate is a VPN section.

### Script 4: HNSW

```python
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
```

```

clustered (100 topics), N=5000, d=32. Exact search computes 5000 distances per query.
  links      build s  ef_search: recall@10 (distances per query)
  closest M     1.7    10: 0.667 (  90)  40: 0.807 ( 127)  160: 0.965 ( 425)
  diverse       2.8    10: 0.982 ( 124)  40: 1.000 ( 216)  160: 1.000 ( 756)

structureless Gaussian, N=5000, d=32. Exact search computes 5000 distances per query.
  links      build s  ef_search: recall@10 (distances per query)
  closest M     2.5    10: 0.443 ( 192)  40: 0.796 ( 485)  160: 0.983 (1352)
  diverse       2.7    10: 0.473 ( 198)  40: 0.799 ( 494)  160: 0.986 (1380)
Nodes per layer (same for every index, set by the random levels): [5000, 604, 77, 10]
```

## Common mistakes

- **Evaluating with questions copied from the documents.** It hides vocabulary mismatch. Write paraphrases and unanswerable questions.
- **Comparing chunkers or retrievers at equal $k$ instead of equal context budget.** Bigger chunks win R@k for free.
- **Tuning on the test questions.** The LSA dimension sweep in script 1 would have "found" a better setting. Hold out questions, or cross-validate as script 3 does.
- **Trusting the top hit.** Search always returns something. Set and validate an abstention rule, and monitor how often it fires.
- **Ignoring metadata.** Stale, superseded or unauthorized documents match words as well as good ones. Filter on status, date and permissions.
- **Reporting only end-to-end accuracy.** Split errors into retrieval and generation, or you will tune the wrong stage.
- **Assuming a bigger context window removes the need to evaluate.** Long-context accuracy depends on position, distractors and length; measure it.

## Exercises

1. [warm-up] Compute the BM25 score of a one-word query by hand for a chunk of 40 words containing the word twice, with $N = 65$, $n_t = 3$, avgdl $= 47$, $k_1 = 1.2$, $b = 0.75$. Check it with `BM25`.
2. [warm-up] Show that RRF with $k = 0$ lets a single retriever's top result dominate the fused ranking. Why does $k = 60$ help?
3. [core] Rerun script 1 with $b = 0$ and $b = 1$. Explain the change with the length distribution of section chunks.
4. [core] Write 10 new paraphrased questions for the corpus without looking at the documents' wording. How do BM25 and LSA do on them compared with the 26 in the file?
5. [core] Add query expansion to BM25: append the top 5 terms of the top-ranked chunk to the query (pseudo-relevance feedback) and measure MRR by query type.
6. [core] In script 3, add a metadata filter that drops superseded documents before ranking, and remove the `current` feature from the reranker. Compare the results on the travel questions.
7. [core] Compute a bootstrap confidence interval for the abstention rate on unanswerable questions at threshold 0.3. What does it say about the 8-question set?
8. [core] In script 4, vary $M \in \{4, 8, 16\}$ with the heuristic and plot recall@10 against distances per query.
9. [stretch] Replace LSA with a real sentence-embedding model if you can install one, keeping everything else fixed. Report MRR by query type with bootstrap intervals, and say whether hybrid now helps.
10. [stretch] Add an LLM generator (local or API) using the printed prompt, and measure faithfulness: for each answer, does the cited chunk contain the claimed fact? Compare with the extractive generator.

### Hints and answers

1. idf $= \ln(1 + 62.5/3.5) \approx 2.94$; the length factor is $1.2 (0.25 + 0.75 \cdot 40/47) \approx 1.066$; tf part $= 2 \cdot 2.2 / (2 + 1.066) \approx 1.435$; score $\approx 4.22$. (Stemming and stop words decide what "the word" is; check the tokens first.)
2. With $k = 0$ rank 1 contributes 1 and rank 2 only 0.5, so one retriever's first place beats a document ranked second by both ($1 + 1/\text{large}$ versus $0.5 + 0.5$). Larger $k$ flattens the curve, so agreement between retrievers matters more than any single top rank.
3. Section chunks vary in length from one sentence to a long list. With $b = 1$ long sections are penalized fully. While writing, MRR moved from 0.721 to 0.743 with $b = 0$ and 0.710 with $b = 1$, differences within the noise of 48 questions.
4. Expect both to drop compared with lexical questions. If LSA holds up better, check whether your paraphrases reuse words that co-occur with the answer in the corpus.
5. Feedback helps when the first hit is right and hurts when it is wrong (query drift), so expect gains on lexical questions and mixed results on paraphrases.
6. The filter removes the 2023 chunks entirely, so stale answers become impossible regardless of the reranker, and the reranker no longer needs the `current` feature. Filtering is the more robust fix; the learned feature only works because the labels happened to teach it.
7. With 6 of 8, the exact (Clopper-Pearson) 95% interval is about 0.35 to 0.97, and a percentile bootstrap gives 0.375 to 1.0 (L0-04). The set can detect only gross failures.
8. Larger $M$ raises recall at a given `ef` and costs more distances per hop and more memory; expect diminishing returns.
9. Expect dense to improve most on paraphrases and to stay weaker on identifiers, with hybrid helping because the errors are now less correlated. Whether that happens on 48 questions is what the bootstrap interval will tell you.
10. Count answers whose key fact (number, name) is absent from the cited chunk. Extractive answers score 100% on this by construction; generative ones can fall short even when they sound right.

## Check your understanding

**1. Why does BM25 still matter when dense retrievers exist?** It needs no training, is robust out of domain (BEIR), and is strong on exact identifiers, names and numbers, which embedding models trained on general text handle poorly. Dense retrieval wins on paraphrase. Their errors differ, which is why hybrids exist.

**2. Why did the hybrid not help in script 1?** Fusion helps when retrievers fail on different queries. LSA is computed from the same word counts as BM25 on the same tiny corpus, so it fails on largely the same paraphrases, and the observed difference was within the bootstrap interval.

**3. What does the HNSW neighbor heuristic buy, and why does it matter more on clustered data?** It keeps some links that point away from the node's own cluster, so greedy search can travel between clusters. With only closest-neighbor links, clustered data form islands and the search gets stuck; in script 4 recall at `ef=10` rose from 0.667 to 0.982.

**4. Why compare chunking schemes at a fixed context budget?** Because a larger chunk contains more text, so it is more likely to contain the answer at any $k$ while costing more of the generator's context and adding distractors. Equal budgets compare what the generator actually receives.

**5. A RAG system answers 48% of test questions correctly. What should you find out next?** Whether the errors are retrieval or generation: is the answer in the index, in the top $k$, in the top-ranked chunk, and did the generator use it? Also what the system does on unanswerable questions, and how wide the confidence interval is for the test set size.

## Further reading

- **Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze, "Introduction to Information Retrieval"** (2008, Cambridge University Press). https://nlp.stanford.edu/IR-book/
  *Read for:* inverted indexes, tf-idf scoring, evaluation of ranked retrieval and latent semantic indexing, explained from scratch. Free online edition. *Level:* intro.
- **Daniel Jurafsky and James H. Martin, "Speech and Language Processing", 3rd edition draft, chapter 11 "Information Retrieval and RAG"** (draft of August 19, 2026). https://web.stanford.edu/~jurafsky/slp3/
  *Read for:* a current textbook treatment of sparse and dense retrieval and RAG. Free PDF. *Level:* intro.
- **Stephen Robertson and Hugo Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond"** (2009, *Foundations and Trends in Information Retrieval* 3(4), 333 to 389). https://doi.org/10.1561/1500000019
  *Read for:* where BM25 comes from and what $k_1$ and $b$ mean. A free author copy is at https://staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf. *Level:* advanced.
- **Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih, "Dense Passage Retrieval for Open-Domain Question Answering"** (2020, EMNLP; arXiv:2004.04906). https://arxiv.org/abs/2004.04906
  *Read for:* training a bi-encoder with in-batch negatives, and its comparison with BM25. *Level:* intermediate.
- **Patrick Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"** (2020, NeurIPS; arXiv:2005.11401). https://arxiv.org/abs/2005.11401
  *Read for:* the original RAG model and its parametric versus non-parametric memory framing. *Level:* intermediate.
- **Gordon V. Cormack, Charles L. A. Clarke, Stefan Büttcher, "Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods"** (2009, SIGIR, 758 to 759). https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
  *Read for:* the two-page paper behind RRF. *Level:* intro.
- **Yu. A. Malkov and D. A. Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs"** (2020, *IEEE TPAMI* 42(4), 824 to 836; arXiv:1603.09320). https://arxiv.org/abs/1603.09320
  *Read for:* HNSW's layers, the $m_L = 1/\ln M$ choice, and the neighbor-selection heuristic. *Level:* advanced.
- **Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna Gurevych, "BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models"** (2021, NeurIPS Datasets and Benchmarks; arXiv:2104.08663). https://arxiv.org/abs/2104.08663
  *Read for:* how sparse, dense, late-interaction and reranking models generalize out of domain. *Level:* intermediate.
- **Omar Khattab and Matei Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"** (2020, SIGIR; arXiv:2004.12832). https://arxiv.org/abs/2004.12832
  *Read for:* late interaction as a middle ground between bi-encoders and cross-encoders. *Level:* advanced.
- **Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang, "Lost in the Middle: How Language Models Use Long Contexts"** (2024, TACL; arXiv:2307.03172). https://arxiv.org/abs/2307.03172
  *Read for:* the position effect in long contexts and what it means for ordering retrieved passages. *Level:* intermediate.
- **Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu Mei, Michael Bendersky, "Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach"** (2024, EMNLP Industry Track; arXiv:2407.16833). https://arxiv.org/abs/2407.16833
  *Read for:* a controlled comparison of RAG and long context, and the Self-Route router. *Level:* intermediate.
- **Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert, "Ragas: Automated Evaluation of Retrieval Augmented Generation"** (2023, arXiv:2309.15217). https://arxiv.org/abs/2309.15217
  *Read for:* reference-free metrics for context quality, faithfulness and answer quality, and their assumptions. *Level:* intermediate.
