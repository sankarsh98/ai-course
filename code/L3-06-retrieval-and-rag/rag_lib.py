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
