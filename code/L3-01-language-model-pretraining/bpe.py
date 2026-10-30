"""Byte-level BPE from scratch: train, encode, decode, and measure.

Run it to train on Romeo and Juliet (the corpus from L2-05) and print what
the tokenizer learned. Other scripts import train_bpe, BPETokenizer.
"""
import pathlib
import re
import time
from collections import Counter, defaultdict

# Pre-tokenizer in the spirit of GPT-2: split text into chunks (a word with
# its leading space, a single digit, a run of punctuation, or whitespace) and
# never merge across chunks. Python's re has no \p{L}, so [^\W\d_] stands in
# for "any Unicode letter". Digits are split one per chunk, a design choice.
PRETOKEN = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d| ?[^\s\w]+|\s+(?!\S)|\s+")

CORPUS = (pathlib.Path(__file__).resolve().parents[1]
          / "L2-05-transformers" / "data" / "romeo_and_juliet.txt")


def pretokenize(text):
    return PRETOKEN.findall(text)


def train_bpe(text, n_merges):
    """Learn n_merges merges. Returns a list of ((a, b), new_id) in order."""
    # Count each distinct chunk once; work on its bytes (ids 0..255).
    chunk_counts = Counter(pretokenize(text))
    seqs = [list(c.encode("utf-8")) for c in chunk_counts]
    freqs = list(chunk_counts.values())

    # pair -> total count, and pair -> which chunks contain it. Updating
    # these incrementally (only for chunks that change) is what makes this
    # fast; recounting everything after every merge is the naive version.
    pair_count = Counter()
    where = defaultdict(set)
    for i, s in enumerate(seqs):
        for p in zip(s, s[1:]):
            pair_count[p] += freqs[i]
            where[p].add(i)

    merges = []
    for k in range(n_merges):
        if not pair_count:
            break
        # Most frequent pair; ties broken by the smaller pair, for determinism.
        best = max(pair_count, key=lambda p: (pair_count[p], -p[0], -p[1]))
        if pair_count[best] < 2:
            break
        new_id = 256 + k
        merges.append((best, new_id))
        for i in list(where[best]):
            s, f = seqs[i], freqs[i]
            for p in zip(s, s[1:]):          # remove this chunk's old pairs
                pair_count[p] -= f
                if pair_count[p] <= 0:
                    del pair_count[p]
                where[p].discard(i)
            seqs[i] = s = merge_seq(s, best, new_id)
            for p in zip(s, s[1:]):          # add its new pairs
                pair_count[p] += f
                where[p].add(i)
        where.pop(best, None)
    return merges


def merge_seq(s, pair, new_id):
    """Replace every non-overlapping occurrence of pair in s, left to right."""
    out, i = [], 0
    while i < len(s):
        if i + 1 < len(s) and s[i] == pair[0] and s[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(s[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges):
        self.rank = {pair: r for r, (pair, _) in enumerate(merges)}
        self.new_id = dict(merges)
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in merges:
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        self.cache = {}

    def encode_chunk(self, chunk):
        if chunk in self.cache:
            return self.cache[chunk]
        s = list(chunk.encode("utf-8"))
        # Apply merges in the order they were learned: repeatedly merge the
        # adjacent pair with the lowest rank until no learned pair remains.
        while len(s) > 1:
            pair = min(zip(s, s[1:]), key=lambda p: self.rank.get(p, 1e18))
            if pair not in self.rank:
                break
            s = merge_seq(s, pair, self.new_id[pair])
        self.cache[chunk] = s
        return s

    def encode(self, text):
        return [t for c in pretokenize(text) for t in self.encode_chunk(c)]

    def decode(self, ids):
        # errors="replace": a token sequence cut mid-character is not valid
        # UTF-8; a whole encoded text always is.
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


def bytes_per_token(tok, text):
    return len(text.encode("utf-8")) / len(tok.encode(text))


if __name__ == "__main__":
    text = CORPUS.read_text(encoding="utf-8")
    n_train = int(0.9 * len(text))
    train, val = text[:n_train], text[n_train:]
    print(f"train {len(train):,} chars, val {len(val):,} chars; "
          f"{len(set(pretokenize(train))):,} distinct pre-tokens")

    t0 = time.time()
    merges = train_bpe(train, 2000)
    print(f"learned {len(merges)} merges in {time.time() - t0:.1f} s")

    tok = BPETokenizer(merges)
    show = lambda i: repr(tok.vocab[i].decode("utf-8", errors="replace"))
    print("first 12 merges:", ", ".join(show(i) for _, i in merges[:12]))
    print("merges 1990-1999:", ", ".join(show(i) for _, i in merges[1990:2000]))

    # Round trip must be exact on any string, including unseen characters.
    for s in [val, "naïve café, 東京 🙂", "x = f(3.14159)\n\tprint(x)"]:
        assert tok.decode(tok.encode(s)) == s
    print("round trip exact on validation text, accents, CJK, emoji, code")

    # Compression on held-out text as the vocabulary grows. Truncating the
    # merge list gives exactly the tokenizer we would have had with fewer merges.
    print("\nmerges  vocab  bytes/token (validation)")
    for m in [0, 100, 250, 500, 1000, 2000]:
        print(f"{m:6d}  {256 + m:5d}  {bytes_per_token(BPETokenizer(merges[:m]), val):.2f}")

    sentence = "Wherefore art thou Romeo? Deny thy father in 1597."
    ids = tok.encode(sentence)
    print(f"\n{sentence!r} -> {len(ids)} tokens:")
    print(" | ".join(tok.vocab[i].decode() for i in ids))

    # An English-only tokenizer on other text: same meaning, very different cost.
    samples = {
        "English": "The weather is very nice today, so we are going to the park.",
        "French": "Il fait très beau aujourd'hui, donc nous allons au parc.",
        "German": "Das Wetter ist heute sehr schön, also gehen wir in den Park.",
        "Hindi": "आज मौसम बहुत अच्छा है, इसलिए हम पार्क जा रहे हैं।",
        "Python": "for i in range(10):\n    total += prices[i] * 1.08",
    }
    print("\ntext      chars  bytes  tokens  tokens/char")
    for name, s in samples.items():
        n = len(tok.encode(s))
        print(f"{name:8s}  {len(s):5d}  {len(s.encode()):5d}  {n:6d}  {n / len(s):.2f}")
