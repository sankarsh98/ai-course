"""Contamination: what n-gram overlap checks catch, what they miss, and how a
leak inflates scores even on a "decontaminated" subset.

A 300-item benchmark of templated word problems; a synthetic pretraining
corpus into which some items leak verbatim and some leak paraphrased.
"""
import re

import numpy as np

rng = np.random.default_rng(4)
NAMES = ["Ana", "Ben", "Chen", "Dara", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun"]
THINGS = ["apples", "stamps", "marbles", "coins", "shells", "cards", "beads", "pencils"]
TEMPLATES = [
    "{n} has {a} {t} and buys {b} more each week for {c} weeks. How many {t} does {n} have now?",
    "{n} shares {a} {t} equally among {b} friends and keeps the {c} left over. How many does each friend get?",
    "A shelf holds {a} boxes with {b} {t} in each box, and {n} removes {c} {t}. How many {t} remain on the shelf?",
]
SYNONYMS = {"has": "owns", "buys": "gets", "more": "extra", "each": "every", "week": "weekly batch",
            "How": "Tell me how", "many": "much", "now": "at the end", "holds": "carries",
            "remain": "are left", "shares": "splits", "friends": "pals", "keeps": "holds on to"}


def make_item(rng):
    k = rng.integers(len(TEMPLATES))
    a, b, c = (int(v) for v in rng.integers(2, 40, size=3))
    n, t = rng.choice(NAMES), rng.choice(THINGS)
    if k == 1:
        a, c = a * b + c % b, c % b          # make the division come out even
    ans = [a + b * c, (a - c) // b, a * b - c][k]
    return dict(text=TEMPLATES[k].format(n=n, a=a, b=b, c=c, t=t), answer=ans)


def paraphrase(text):
    """Word substitutions plus moving the question to the front. Same meaning,
    different surface form."""
    words = [SYNONYMS.get(w, w) for w in text.split()]
    s = " ".join(words)
    body, q = s.split(". ", 1) if ". " in s else (s, "")
    return f"{q} Context: {body}."


bench = [make_item(rng) for _ in range(300)]
leak_verbatim = set(range(0, 30))      # items 0-29 appear word for word in the corpus
leak_para = set(range(30, 60))         # items 30-59 appear paraphrased

FILLER = ("the of and to in is was for on that with as by at from it this be are or "
          "an which one all were we when there can more if will their has would time "
          "people who about other into some could them see only new these two may").split()


def filler(n_words):
    return " ".join(rng.choice(FILLER, n_words))


corpus = [filler(rng.integers(50, 200)) for _ in range(2000)]
for i in leak_verbatim:
    corpus.append(f"{filler(30)} Q: {bench[i]['text']} A: {bench[i]['answer']} {filler(30)}")
for i in leak_para:
    corpus.append(f"{filler(30)} {paraphrase(bench[i]['text'])} Answer {bench[i]['answer']} {filler(30)}")
# Clean items still share template text with many web pages about word problems.
for _ in range(40):
    corpus.append(f"{filler(20)} {make_item(rng)['text']} {filler(20)}")


def tokens(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def ngrams(toks, n):
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def flagged(n):
    """Items sharing at least one n-gram with any corpus document."""
    corpus_grams = set().union(*(ngrams(tokens(d), n) for d in corpus))
    return {i for i, it in enumerate(bench) if ngrams(tokens(it["text"]), n) & corpus_grams}


print("1. n-gram overlap detection (300 items: 30 leaked verbatim, 30 paraphrased, 240 clean)")
print("    n   verbatim caught   paraphrased caught   clean items flagged")
for n in [5, 8, 13]:
    f = flagged(n)
    clean = set(range(300)) - leak_verbatim - leak_para
    print(f"   {n:2d}   {len(f & leak_verbatim):9d}/30   {len(f & leak_para):12d}/30   "
          f"{len(f & clean):14d}/240")

# ---- 2. What the leak does to scores -------------------------------------------
# A model with genuine accuracy 0.60 that recalls the stored answer 90% of the
# time when it has seen the item (in either form) in training. We use expected
# scores rather than sampled ones, so the numbers below carry no sampling noise.
seen = leak_verbatim | leak_para


def expected_score(item_ids, perturbed=False):
    p_seen = 0.6 if perturbed else 0.9 + 0.1 * 0.6
    return np.mean([p_seen if i in seen else 0.6 for i in item_ids])


clean13 = sorted(set(range(300)) - flagged(13))
print("\n2. expected scores of a model with true skill 0.60 that memorized the leaked items")
print(f"   full benchmark:                           {expected_score(range(300)):.3f}")
print(f"   '13-gram clean' subset ({len(clean13)} items):        {expected_score(clean13):.3f}"
      f"   ({len(set(clean13) & seen)} leaked items survive the filter)")
print(f"   truly clean items (240):                  {expected_score(range(60, 300)):.3f}")
print(f"   same items with new numbers and names:    {expected_score(range(300), perturbed=True):.3f}")
