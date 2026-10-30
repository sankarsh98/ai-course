# Language Model Pretraining

> **Level 3 · Module 1** · Prerequisites: L0-04, L0-05, L2-02, L2-04, L2-05 · Time: 10 to 12 hours · Code: `code/L3-01-language-model-pretraining/`

## What you will be able to do

- Implement byte-level BPE from scratch, and explain how vocabulary size, pre-tokenization and language change sequence length and cost.
- State the next-token objective, and compare models with different tokenizers in bits per byte.
- Describe a modern pretraining data pipeline and why each step exists.
- Compute training FLOPs with $C \approx 6ND$, derive a compute-optimal allocation from a fitted scaling law, and explain why practice deviates from it.
- Explain decoder-only and mixture-of-experts designs and the main long-context techniques well enough to read a technical report critically.
- Judge claims of emergent abilities by separating what a model can do from what a metric shows.

## 1. What pretraining is

A **base model** is trained on one task: predict the next token. For tokens $x_1, \dots, x_T$ the model defines $p_\theta(x_1, \dots, x_T) = \prod_{t} p_\theta(x_t \mid x_{<t})$, and training minimizes the average negative log-likelihood $-\frac{1}{T}\sum_t \log p_\theta(x_t \mid x_{<t})$, the cross-entropy of L0-04. L2-05 trained exactly this on one play; pretraining runs it on trillions of tokens. Documents are joined with an end-of-text token and cut into fixed-length windows, so nothing is wasted on padding.

Why does one objective teach so much? Predicting the next token of a proof, a translation or a bug fix is easier if you model the mathematics, both languages or the program. GPT-2 framed this as unsupervised multitask learning [Radford et al., 2019]. GPT-3 (175B parameters) showed that a large base model can perform new tasks from a few examples in its prompt, with no gradient updates: **in-context learning** (Brown et al., 2020, "Language Models are Few-Shot Learners", arXiv:2005.14165). A base model still only continues text. Turning it into an assistant is L3-02.

**Units.** Loss is in nats per token and perplexity is $e^{\text{loss}}$. Tokens differ between tokenizers, so the comparable unit is **bits per byte**: total bits for a text divided by its UTF-8 length. Script 2 shows why it matters.

## 2. Tokenization

Characters make sequences long (attention cost grows with length, L2-05 section 9). Whole words make the vocabulary unbounded and leave unseen words unwritable. **Subword** tokenizers sit between: frequent strings become one token, rare ones are spelled from pieces.

**Byte Pair Encoding (BPE)** [Sennrich et al., 2016]: start from single symbols, count adjacent pairs, merge the most frequent pair into a new symbol, repeat until the vocabulary is the size you want. The ordered merge list *is* the tokenizer. With word counts `the` 6, `then` 3, `than` 2, `an` 4:

| step | most frequent pair | count | words afterwards |
|---|---|---|---|
| 1 | `t h` | 6+3+2 = 11 | `th e`, `th e n`, `th a n`, `a n` |
| 2 | `th e` | 6+3 = 9 | `the`, `the n`, `th a n`, `a n` |
| 3 | `a n` | 2+4 = 6 | `the`, `the n`, `th an`, `an` |

To encode new text, apply the merges in the order learned. Recounting all pairs after every merge is slow; `bpe.py` keeps an index from each pair to the words containing it and updates only what changed, so 2,000 merges take about a second.

**Byte-level BPE.** Over Unicode characters the base vocabulary would be over 130,000 symbols, by GPT-2's count, and still miss unseen characters. GPT-2 ran BPE on UTF-8 bytes instead, so the base vocabulary is exactly 256 and any string can be encoded, and it blocked merges across character categories (letters, digits, punctuation, with an exception for a leading space) so that `dog`, `dog.` and `dog!` do not become separate tokens; its vocabulary was 50,257 [Radford et al., 2019]. That splitting is done by a **pre-tokenizer** regex, a real design decision: `bpe.py` splits digits one at a time, so `1597` is always four tokens.

**Unigram LM** [Kudo, 2018] goes the other way: start from a large candidate vocabulary, repeatedly drop the tokens whose removal hurts corpus likelihood least, and segment words with the Viterbi algorithm. Being probabilistic, it can also sample segmentations during training (subword regularization). SentencePiece implements both algorithms on raw text, spaces included (Kudo and Richardson, 2018, arXiv:1808.06226).

**Trade-offs, measured in script 1.** Bytes per token on held-out text rise from 1.00 to 2.30 at 500 merges and 3.04 at 2,000, with diminishing returns, while the embedding and output layers grow as $V \times d$. Text unlike the training data costs more: the same sentence takes 20 tokens in English, 31 to 33 in French and German, and 125 in Hindi (one per byte, because no Devanagari merge was learned). Multilingual tokenizers narrow this but do not close it: Petrov et al. found up to 15-fold differences in token length for the same content across languages ("Language Model Tokenizers Introduce Unfairness Between Languages", NeurIPS 2023, arXiv:2305.15425). Prices, context limits and latency are per token, so this is a cost and fairness issue. Tokenization also explains failures such as counting letters in a word the model only sees as two tokens. Some work removes the fixed tokenizer: the Byte Latent Transformer groups bytes into variable-length patches by how predictable they are, and reports matching token-based models in a compute-controlled study up to 8B parameters (Pagnoni et al., 2024, arXiv:2412.09871). Whether this displaces tokenizers at the largest scales is open.

## 3. Data: sources and curation

Corpora are mostly web text, plus code, books, papers, encyclopedias and math. Raw crawls are mostly unusable; much of a lab's edge is in cleaning them. The pipeline documented for FineWeb [Penedo et al., 2024] and Llama 3 [Grattafiori et al., 2024]:

1. **Extract** main text from HTML; **identify language**; apply **heuristic filters** (too short, mostly symbols, repeated lines, blocked domains).
2. **Deduplicate**, exactly and approximately (for example MinHash over n-grams). Lee et al. found one 61-word sentence repeated over 60,000 times in C4; deduplication made models emit memorized text far less often and train in fewer steps ("Deduplicating Training Data Makes Language Models Better", ACL 2022, arXiv:2107.06499).
3. **Filter for quality with a model**: a classifier scores pages and the best are kept. FineWeb (15T tokens from 96 Common Crawl snapshots) released FineWeb-Edu, 1.3T tokens chosen by an educational-quality classifier [Penedo et al., 2024]. DataComp-LM's controlled comparisons also found model-based filtering to be the key step; its 7B model, trained on 2.6T filtered tokens, scored 64% on MMLU with 6.6 times less compute than Llama 3 8B (Li et al., 2024, arXiv:2406.11794).
4. **Remove** personal information and some harmful content, and **decontaminate**: strip benchmark test sets so scores are not memorization (L4-03).
5. **Mix** sources. Llama 3's mix was about 50% general knowledge, 25% math and reasoning, 17% code, 8% multilingual [Grattafiori et al., 2024], tuned on small proxy models.

High-quality text is finite. Muennighoff et al. found up to about 4 epochs of repetition nearly as good as fresh data, with returns decaying toward zero beyond that ("Scaling Data-Constrained Language Models", arXiv:2305.16264). Many recipes end with **annealing**: decay the learning rate while upweighting the best data. Llama 3 reports that annealing a half-trained 8B model on 40B high-quality tokens raised GSM8K and MATH validation scores by 24.0% and 6.4%, with negligible gains for the 405B model [Grattafiori et al., 2024]. Copyright and consent questions about training data belong to L4-05.

## 4. Scaling laws and the compute-optimal debate

**Counting compute.** A forward pass costs about $2N$ FLOPs per token for $N$ parameters (one multiply-add per weight) and the backward pass about twice that, so training on $D$ tokens costs $C \approx 6ND$. Check: Llama 3 405B on 15.6T tokens gives $3.79 \times 10^{25}$, matching the reported $3.8 \times 10^{25}$ [Grattafiori et al., 2024]. (Attention's $T^2$ term is ignored; L4-02 does the full accounting.)

**Kaplan et al. (2020)** found loss falling as a power law in model size, data and compute over more than seven orders of magnitude, with depth versus width mattering little [Kaplan et al., 2020]. Their allocation favored very large models on relatively little data, stopped early: $N_{\text{opt}} \propto C^{0.73}$.

**Chinchilla (2022).** Hoffmann et al. trained over 400 models (70M to 16B parameters, 5B to 500B tokens) and estimated the split three ways; all said to grow parameters and tokens about equally, $N_{\text{opt}} \propto C^{a}$ with $a = 0.50, 0.49, 0.46$ [Hoffmann et al., 2022]. Chinchilla, 70B parameters on 1.4T tokens, used the same compute as the 280B Gopher, beat it, and reached 67.5% on MMLU. Twenty tokens per parameter became the rule of thumb. The third method fits

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},$$

an irreducible term $E$ plus penalties for too few parameters and too little data. Minimizing it subject to $6ND = C$ gives $N_{\text{opt}} = G\,(C/6)^{\beta/(\alpha+\beta)}$ with $G = (\alpha A / \beta B)^{1/(\alpha+\beta)}$ and $D_{\text{opt}} = C/6N_{\text{opt}}$.

**Why did the two disagree?** Pearce and Song attribute most of the gap to Kaplan counting only non-embedding parameters and working at small scale ("Reconciling Kaplan and Chinchilla Scaling Laws", TMLR 2024, arXiv:2406.12907). An exponent is a property of a fit and its setup, not of nature.

**The replication.** Besiroglu et al. rebuilt the data behind Chinchilla's parametric fit from the paper's figure and found the published fit inconsistent with the other two methods, poorly fitting, and with confidence intervals so narrow they would need over 600,000 experiments [Besiroglu et al., 2024]. Script 3 compares the fits. At Chinchilla's budget ($5.76 \times 10^{23}$), the published constants recommend 40B parameters on 2.4T tokens (59 tokens per parameter) and a ratio rising to 85 at $3.8 \times 10^{25}$; the refit recommends 72B on 1.3T (18 per parameter), close to what Chinchilla actually did, and stays near 16 to 22. The main conclusion survives; a widely copied set of constants did not.

**The optimum is flat.** At fixed compute, half or twice the optimal size costs only about 0.005 nats in script 3. Llama 3's scaling law predicted 402B parameters on 16.55T tokens; the authors chose 405B, noting the flat isoFLOP minima [Grattafiori et al., 2024].

**Why practice ignores it.** "Compute-optimal" minimizes *training* compute for a given loss, but a deployed model also costs about $2N$ FLOPs per generated token. Sardana et al. added inference to the scaling law and concluded that anyone expecting heavy use should train smaller models on more data ("Beyond Chinchilla-Optimal", ICML 2024, arXiv:2401.00448). In script 3, matching the optimal model's loss at $10^{24}$ FLOPs with half the parameters takes 1.26 times the training compute; with a quarter, 4.47 times and about 1,300 tokens per parameter. Llama 3's smaller models were trained "much longer than is compute-optimal" in the authors' words, and Qwen3's report describes training all its models, 0.6B to 235B parameters, on a dataset of about 36T tokens (Yang et al., 2025, arXiv:2505.09388). In the open reports I checked for this module (2024 to early 2026), training far past 20 tokens per parameter is the norm.

**What the laws do not say.** They predict loss on the training distribution, not capabilities (section 7), and assume enough unique data, which is becoming a real constraint for the largest runs; how binding it is, is debated. Llama 3 did forecast a benchmark (ARC Challenge) in two steps, compute to log-likelihood of correct answers, then log-likelihood to accuracy, across four orders of magnitude with a slight underestimate [Grattafiori et al., 2024]. That is the kind of evidence a scaling claim should come with.

## 5. Architecture choices

Encoder-only models (BERT style) see the whole input and suit classification and embeddings (L2-06); encoder-decoders (T5 style) suit input-to-output tasks; large general models are decoder-only (L2-05), since one causal stack handles any task phrased as text and the loss uses every position.

**The modern dense block** differs from L2-05's GPT in small ways: RMSNorm, gated MLPs (SwiGLU), rotary positions, no biases, and **grouped-query attention**, where several query heads share a key-value head to shrink the KV cache. Raschka's survey of ten open-weight releases from January and February 2026 shows the variants then in use: grouped-query or multi-head latent attention, sliding-window layers mixed with global ones (often 3 to 1), QK-norm and gated attention for stability, and hybrids that replace most attention layers with linear-attention layers such as Gated DeltaNet [Raschka, 2026]. He concludes that data and training recipes explain more of the differences between models than architecture does.

**Mixture of experts (MoE).** Replace each MLP with many expert MLPs and a small **router** that sends each token to its top $k$. Parameters grow with the number of experts, compute per token only with $k$. The idea at scale dates to Shazeer et al.'s sparsely gated MoE layer (2017, arXiv:1701.06538); Switch Transformers simplified routing and reported up to 7 times faster pretraining than dense T5 at equal compute (Fedus et al., 2021, arXiv:2101.03961). Documented examples: DeepSeek-V3, 671B total and 37B active parameters, 14.8T tokens (DeepSeek-AI, 2024, arXiv:2412.19437); Kimi K2, 1T total and 32B active, 15.5T tokens (Kimi Team, 2025, arXiv:2507.20534); Qwen3-235B-A22B, 235B total and 22B active. In Raschka's early-2026 survey MoE is the default for large open models. The costs: every expert must sit in memory, so serving needs more memory and communication (L4-02); routers can collapse onto a few experts and need load balancing; and MoE training has a history of instability. For "1T parameters", ask how many are active per token.

**Closed models.** OpenAI, Anthropic and Google do not publish architecture, size or data for their frontier models. Figures that circulate are rumors or estimates; do not repeat them as facts.

## 6. Long context

Attention FLOPs grow as $T^2$ and genuinely long training documents are scarce, so models are pretrained short and extended at the end. Llama 3 went from 8K to 128K tokens in six stages using about 800B tokens [Grattafiori et al., 2024]; Qwen3 used a final stage at 32,768 tokens. With rotary embeddings, positions past the training length give rotation angles the model never saw; **position interpolation** squeezes the longer range into the trained one, and Chen et al. extended LLaMA models to 32,768 tokens with under 1,000 fine-tuning steps ("Extending Context Window of Large Language Models via Positional Interpolation", arXiv:2306.15595). Sliding-window and linear-attention layers (section 5) cut the cost instead.

**A context window is a claim, not a measurement.** Needle-in-a-haystack tests are easy; on RULER's harder tasks, only about half of 17 models claiming 32K or more performed satisfactorily at 32K (Hsieh et al., COLM 2024, arXiv:2404.06654). Models also use information at the start and end of the context better than in the middle (Liu et al., "Lost in the Middle", TACL, arXiv:2307.03172). Long context versus retrieval is L3-06.

## 7. Emergent abilities and the debate

Wei et al. call an ability **emergent** if it is "not present in smaller models but is present in larger models", and documented many tasks where performance sat near chance and then rose sharply with scale [Wei et al., 2022]. If abilities appear abruptly, small experiments cannot warn you, which matters for forecasting and safety.

Schaeffer et al. argued that much of this comes from the metric [Schaeffer et al., 2023]. If per-token accuracy $p$ improves smoothly, a metric requiring all $L$ tokens right scores about $p^L$, which stays near zero and then shoots up. With continuous metrics the same models improved smoothly, and a sharp metric produced apparent emergence even in vision models. The paper won a NeurIPS 2023 outstanding paper award.

Du et al. pushed back: downstream performance tracks pretraining loss across model sizes, and some tasks stay at chance until loss crosses a threshold and then improve, even under continuous metrics ("Understanding Emergent Abilities of Language Models from the Loss Perspective", NeurIPS 2024, arXiv:2403.15796). Llama 3's two-step forecast (section 4) is the practical version: capabilities can be predicted from loss if you model the nonlinear step to accuracy.

Script 4 shows both effects. For a 57,744-parameter model learning 3-digit addition, exact match jumps from 0.7% to 81.5% between steps 300 and 600 while per-digit accuracy goes from 0.49 to 0.95, and exact match tracks $p^4$, as Schaeffer's model predicts. But the smooth metrics have a plateau too: the smaller model's loss sits near 1.73 from step 150 to 450, then falls fast, a genuine change in what it computes. And the small model breaks the $p^L$ argument: 77% per-digit accuracy but only 9.7% exact match, far below $0.774^4 = 0.36$, because its errors are not independent. It solves three digit positions and is at chance on the tens digit. Some reported emergence is a measurement artifact, some is a real nonlinear transition; you have to look at the task and metric to tell which.

## 8. What pretraining at scale involves

Llama 3 405B trained on up to 16K H100 GPUs. In one 54-day period there were 466 job interruptions, 419 unexpected, with GPU problems behind 58.7% of the unexpected ones [Grattafiori et al., 2024]. At this scale failure is routine: frequent checkpoints, detection of slow or faulty nodes, automatic restarts. Parallelism is L4-02. **Loss spikes**, where the loss jumps and may diverge and force a rollback, are the other risk; DeepSeek-V3 reports no irrecoverable spikes or rollbacks, and Kimi K2 reports none at all, crediting its MuonClip optimizer (both the authors' own claims). Checkpoints are evaluated throughout, because a bad data mix found at 5% of a run is far cheaper than one found at the end. The product is a base model: a strong text predictor with no particular wish to help.

## Hands-on

Run from the repository root, for example `.venv/bin/python code/L3-01-language-model-pretraining/bpe.py`. Outputs come from Python 3.12.3, PyTorch 2.15.0 (CPU) and NumPy 2.5.4 on Linux (WSL2), Intel i7-13700HX, on a machine shared with other heavy jobs (load average 4 to 10). Seeded results reproduced exactly across reruns; timings did not. Scripts 2 and 4 import `gpt.py` from L2-05 (shown in full there), and scripts 1 and 2 read L2-05's copy of Romeo and Juliet (Project Gutenberg eBook #1513, public domain).

### Script 1: byte-level BPE from scratch, `bpe.py`

```python
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
```

```text
train 128,052 chars, val 14,229 chars; 4,170 distinct pre-tokens
learned 2000 merges in 1.3 s
first 12 merges: ' t', 'he', ' a', ' s', 'ou', ' m', 'in', ' w', 'ha', 're', ' b', 'nd'
merges 1990-1999: ' ut', ' ur', ' id', ' ord', ' You', ' Then', ' When', ' iron', "!'", "'."
round trip exact on validation text, accents, CJK, emoji, code

merges  vocab  bytes/token (validation)
     0    256  1.00
   100    356  1.63
   250    506  1.93
   500    756  2.30
  1000   1256  2.68
  2000   2256  3.04

'Wherefore art thou Romeo? Deny thy father in 1597.' -> 18 tokens:
Where | fore |  art |  thou |  Romeo | ? |  D | en | y |  thy |  father |  in |   | 1 | 5 | 9 | 7 | .

text      chars  bytes  tokens  tokens/char
English      60     60      20  0.33
French       56     57      33  0.59
German       60     61      31  0.52
Hindi        49    125     125  2.55
Python       49     49      29  0.59
```

The first merges are what English frequency predicts. By merge 2,000 it is learning whole words from only 4,170 distinct pre-tokens, close to memorizing this corpus, yet "Deny" is still three tokens. The round trip is exact for emoji and Chinese characters the corpus never contains, the point of byte-level BPE. Hindi's 2.55 tokens per character is one token per UTF-8 byte (Devanagari characters are 3 bytes, and combining marks count as characters).

### Script 2: same training tokens, two tokenizers, `tokens_vs_chars.py`

```python
"""Same model, same compute, two tokenizers: which predicts the text better?

Trains the L2-05 GPT twice on Romeo and Juliet, once on characters and once
on BPE tokens, with the same number of steps, batch and context length in
tokens (so about the same training FLOPs). Loss per token is not comparable
across tokenizers, so both are converted to bits per byte of text.
"""
import math
import pathlib
import sys
import time

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.dont_write_bytecode = True  # don't leave a __pycache__ in L2-05's folder
sys.path.insert(0, str(HERE.parent / "L2-05-transformers"))  # reuse L2-05's model
from gpt import GPT, GPTConfig, count_params  # noqa: E402
from bpe import CORPUS, BPETokenizer, train_bpe  # noqa: E402

torch.set_num_threads(8)
text = CORPUS.read_text(encoding="utf-8")
n_train = int(0.9 * len(text))
train_text, val_text = text[:n_train], text[n_train:]
val_bytes = len(val_text.encode("utf-8"))

STEPS, BATCH, BLOCK, LR = 1000, 32, 64, 3e-3


def char_tokenizer():
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    return len(chars), lambda s: [stoi[c] for c in s]


def bpe_tokenizer(n_merges=500):
    tok = BPETokenizer(train_bpe(train_text, n_merges))  # trained on train only
    return 256 + n_merges, tok.encode


def run(name, vocab_size, encode):
    torch.manual_seed(0)
    tr = torch.tensor(encode(train_text))
    va = torch.tensor(encode(val_text))
    model = GPT(GPTConfig(vocab_size=vocab_size, block_size=BLOCK,
                          n_layer=3, n_head=4, n_embd=64))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

    def batch(src, g=None):
        ix = torch.randint(len(src) - BLOCK - 1, (BATCH,), generator=g)
        return (torch.stack([src[i:i + BLOCK] for i in ix]),
                torch.stack([src[i + 1:i + BLOCK + 1] for i in ix]))

    t0 = time.time()
    for step in range(STEPS):
        for gr in opt.param_groups:      # cosine decay, no warmup needed here
            gr["lr"] = LR * 0.5 * (1 + math.cos(math.pi * step / STEPS))
        _, loss = model(*batch(tr))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Validation: nats per token, averaged over fixed batches.
    model.eval()
    g = torch.Generator().manual_seed(123)
    with torch.no_grad():
        nats = sum(model(*batch(va, g))[1].item() for _ in range(40)) / 40
    # Total bits for the validation text = bits/token * tokens; divide by bytes.
    bpb = nats / math.log(2) * len(va) / val_bytes
    print(f"{name:5s} vocab {vocab_size:4d}  params {count_params(model):7,}  "
          f"val tokens {len(va):6,}  bytes/token {val_bytes / len(va):.2f}  "
          f"context {BLOCK} tok = {BLOCK * val_bytes / len(va):5.0f} bytes  "
          f"loss {nats:.3f} nats/token = {bpb:.3f} bits/byte  ({time.time() - t0:.0f} s)")


print(f"{STEPS} steps x {BATCH} sequences x {BLOCK} tokens for both runs\n")
run("char", *char_tokenizer())
run("bpe", *bpe_tokenizer())
```

```text
1000 steps x 32 sequences x 64 tokens for both runs

char  vocab   66  params 158,400  val tokens 14,229  bytes/token 1.00  context 64 tok =    64 bytes  loss 1.820 nats/token = 2.625 bits/byte  (19 s)
bpe   vocab  756  params 202,560  val tokens  6,185  bytes/token 2.30  context 64 tok =   147 bytes  loss 3.722 nats/token = 2.334 bits/byte  (25 s)
```

Per token the BPE model looks much worse (3.72 nats against 1.82), which means nothing, since its tokens carry 2.3 bytes. Per byte it wins, 2.33 against 2.63 bits. Each 64-token window covers 147 bytes instead of 64, and the model need not spell common words. Caveats: one seed; the larger output layer gives the BPE model 28% more parameters and roughly 29% more FLOPs per token here (far less in a large model); and it passes over its 55K training tokens about 37 times, against 16 for the character model, so it may overfit more (exercise 4). L2-05's larger character model reached 2.46 bits per byte. The whole script took 46 s under load and 33 s on a quieter rerun.

### Script 3: compute and scaling-law arithmetic, `scaling_laws.py`

```python
"""Compute accounting and the Chinchilla compute-optimal debate, in numbers.

Uses the parametric loss L(N, D) = E + A / N^alpha + B / D^beta, with N
parameters and D training tokens, and training compute C = 6 N D FLOPs.
Two published fits of the same functional form:
  Hoffmann et al. (2022), eq. 4, as reported;
  Besiroglu et al. (2024), Table 1, a refit of data extracted from Hoffmann's
  Figure 4.
The absolute losses are for Chinchilla's data and tokenizer; only the
trade-offs transfer to other setups.
"""
FITS = {
    "Hoffmann 2022": dict(E=1.6934, A=406.4, B=410.7, alpha=0.3392, beta=0.2849),
    "Besiroglu 2024": dict(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658),
}


def loss(f, N, D):
    return f["E"] + f["A"] / N ** f["alpha"] + f["B"] / D ** f["beta"]


def compute_optimal(f, C):
    """Minimize L subject to 6 N D = C (closed form, Hoffmann eq. 4)."""
    a, b = f["alpha"], f["beta"]
    G = (a * f["A"] / (b * f["B"])) ** (1 / (a + b))
    N = G * (C / 6) ** (b / (a + b))
    return N, C / (6 * N)


def fmt(x):
    for unit, s in [(1e12, "T"), (1e9, "B"), (1e6, "M")]:
        if x >= unit:
            return f"{x / unit:.1f}{s}"
    return f"{x:.0f}"


# 1. Compute accounting: about 6 FLOPs per parameter per token
#    (2 for the forward multiply-add, 4 for the backward pass).
print("1. Training compute C = 6 N D")
for name, N, D, reported in [("Chinchilla", 70e9, 1.4e12, "budget 5.76e23"),
                             ("Llama 3 405B", 405e9, 15.6e12, "3.8e25")]:
    print(f"   {name:13s} N={fmt(N):>6s} D={fmt(D):>6s}  6ND = {6 * N * D:.2e}"
          f"  (paper: {reported})")

# 2. Where should a compute budget go? Same functional form, two fits.
print("\n2. Compute-optimal allocation  N* (params), D* (tokens), D*/N*")
print("   budget C   " + "".join(f"{k:>34s}" for k in FITS))
for C in [1e21, 1e23, 5.76e23, 1e25, 3.8e25]:
    row = f"   {C:8.2e}  "
    for f in FITS.values():
        N, D = compute_optimal(f, C)
        row += f"   N*={fmt(N):>7s} D*={fmt(D):>7s} ratio={D / N:5.1f}"
    print(row)
for k, f in FITS.items():
    a = f["beta"] / (f["alpha"] + f["beta"])
    print(f"   {k}: N* grows as C^{a:.3f}, D* as C^{1 - a:.3f}")

# 3. How flat is the optimum? Loss for models of other sizes at C = 5.76e23.
f = FITS["Besiroglu 2024"]
C = 5.76e23
N_opt, _ = compute_optimal(f, C)
L_opt = loss(f, N_opt, C / (6 * N_opt))
print(f"\n3. Same budget C = {C:.2e}, other model sizes (Besiroglu fit)")
for k in [0.25, 0.5, 1, 2, 4]:
    N = k * N_opt
    D = C / (6 * N)
    print(f"   N = {k:4.2f} x N* = {fmt(N):>6s}, D = {fmt(D):>6s}: "
          f"loss {loss(f, N, D):.4f} (+{loss(f, N, D) - L_opt:.4f})")

# 4. Over-training: a smaller model trained on far more tokens than
#    compute-optimal. What extra training compute buys a cheaper model to serve?
print("\n4. Over-training (Besiroglu fit): reach the loss of the compute-optimal model")
print("   at C = 1e24 with a smaller model by training it on more tokens")
C = 1e24
N0, D0 = compute_optimal(f, C)
target = loss(f, N0, D0)
print(f"   compute-optimal: N={fmt(N0)}, D={fmt(D0)}, loss {target:.4f}")
for shrink in [0.5, 0.25]:
    N = shrink * N0
    gap = target - f["E"] - f["A"] / N ** f["alpha"]   # what the data term may cost
    if gap <= 0:
        print(f"   N={fmt(N)}: cannot reach the target with any amount of data")
        continue
    D = (f["B"] / gap) ** (1 / f["beta"])
    extra = 6 * N * D / C
    print(f"   N={fmt(N):>6s}: needs D={fmt(D):>6s} ({D / N:5.0f} tokens/param), "
          f"training compute x{extra:.2f}, inference cost per token x{shrink}")
```

```text
1. Training compute C = 6 N D
   Chinchilla    N= 70.0B D=  1.4T  6ND = 5.88e+23  (paper: budget 5.76e23)
   Llama 3 405B  N=405.0B D= 15.6T  6ND = 3.79e+25  (paper: 3.8e25)

2. Compute-optimal allocation  N* (params), D* (tokens), D*/N*
   budget C                        Hoffmann 2022                    Besiroglu 2024
   1.00e+21     N*=   2.2B D*=  75.3B ratio= 34.0   N*=   2.8B D*=  60.0B ratio= 21.6
   1.00e+23     N*=  18.1B D*= 919.5B ratio= 50.7   N*=  29.4B D*= 566.0B ratio= 19.2
   5.76e+23     N*=  40.3B D*=   2.4T ratio= 59.1   N*=  72.2B D*=   1.3T ratio= 18.4
   1.00e+25     N*= 148.3B D*=  11.2T ratio= 75.7   N*= 312.1B D*=   5.3T ratio= 17.1
   3.80e+25     N*= 272.9B D*=  23.2T ratio= 85.1   N*= 618.7B D*=  10.2T ratio= 16.5
   Hoffmann 2022: N* grows as C^0.456, D* as C^0.544
   Besiroglu 2024: N* grows as C^0.513, D* as C^0.487

3. Same budget C = 5.76e+23, other model sizes (Besiroglu fit)
   N = 0.25 x N* =  18.1B, D =   5.3T: loss 1.9939 (+0.0195)
   N = 0.50 x N* =  36.1B, D =   2.7T: loss 1.9793 (+0.0048)
   N = 1.00 x N* =  72.2B, D =   1.3T: loss 1.9744 (+0.0000)
   N = 2.00 x N* = 144.5B, D = 664.4B: loss 1.9793 (+0.0049)
   N = 4.00 x N* = 289.0B, D = 332.2B: loss 1.9942 (+0.0198)

4. Over-training (Besiroglu fit): reach the loss of the compute-optimal model
   at C = 1e24 with a smaller model by training it on more tokens
   compute-optimal: N=95.9B, D=1.7T, loss 1.9597
   N= 47.9B: needs D=  4.4T (   91 tokens/param), training compute x1.26, inference cost per token x0.5
   N= 24.0B: needs D= 31.1T ( 1296 tokens/param), training compute x4.47, inference cost per token x0.25
```

Chinchilla's 5.88e23 is slightly above the paper's budget because 70B and 1.4T are rounded. The two fits share a functional form and nearly the same $\alpha$, but $B$ and $\beta$ differ enough to triple the recommended tokens per parameter. The losses are for Chinchilla's data and tokenizer; only the shapes transfer.

### Script 4: smooth and sharp metrics, `emergence_metrics.py`

```python
"""Is a capability 'emergent', or is the metric sharp? A small test.

Trains small L2-05 GPTs on 3-digit addition and, at checkpoints along
training (the compute axis), scores each on held-out problems three ways:
loss on the answer tokens and per-digit accuracy (smooth, partial credit),
and exact match of the whole 4-digit answer (all or nothing).
"""
import math
import pathlib
import sys
import time

import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True  # don't leave a __pycache__ in L2-05's folder
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "L2-05-transformers"))
from gpt import GPT, GPTConfig, count_params  # noqa: E402

torch.set_num_threads(8)
PLUS, EQ = 10, 11            # tokens 0-9 are digits
ANS = 4                      # answer digits, written least significant first


def is_test(a, b):           # 1 in 7 of all (a, b) pairs is held out
    return (a * 1000 + b) % 7 == 0


def make_problems(n, g, test=False):
    a = torch.randint(0, 1000, (n,), generator=g)
    b = torch.randint(0, 1000, (n,), generator=g)
    if test:
        b = b - (a * 1000 + b) % 7          # move each pair onto a test pair
        b = torch.where(b < 0, b + 7, b)    # (stays a test pair, mod 7)
    else:
        b = torch.where(is_test(a, b), (b + 1) % 1000, b)  # off the test set
    return a, b


def encode(a, b):
    """'123+456=' then the sum reversed ('9750' for 579), as digit tokens."""
    digits = lambda x, k: torch.stack([(x // 10 ** i) % 10 for i in range(k)], 1)
    # Operands most significant digit first, answer least significant first:
    # then each answer digit depends only on digits already seen plus a carry.
    return torch.cat([digits(a, 3).flip(1), torch.full_like(a, PLUS)[:, None],
                      digits(b, 3).flip(1), torch.full_like(a, EQ)[:, None],
                      digits(a + b, ANS)], 1)   # length 3+1+3+1+4 = 12


def split(seq):
    x, y = seq[:, :-1], seq[:, 1:].clone()
    y[:, :7] = -100           # no loss on predicting the question itself
    return x, y


def evaluate(model, test_seq):
    model.eval()
    with torch.no_grad():
        x, y = split(test_seq)
        logits, _ = model(x)
        ans_loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
        # Teacher-forced greedy digits at the 4 answer positions. For exact
        # match this equals free generation: the first wrong digit fails both.
        correct = logits[:, 7:].argmax(-1) == test_seq[:, 8:]
    model.train()
    return (ans_loss.item(), correct.float().mean().item(),
            correct.all(1).float().mean().item(), correct.float().mean(0))


g = torch.Generator().manual_seed(0)
a, b = make_problems(4000, g, test=True)
assert is_test(a, b).all()
test_seq = encode(a, b)
STEPS, BATCH, LR, EVERY = 1500, 128, 3e-3, 150

for width in [24, 48]:
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=12, block_size=11, n_layer=2,
                          n_head=2, n_embd=width))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    print(f"\nwidth {width}, {count_params(model):,} parameters; "
          f"{BATCH} fresh training problems per step")
    print("  step  answer loss  digit acc  exact match  digit_acc^4")
    t0 = time.time()
    for step in range(STEPS + 1):
        if step % EVERY == 0:
            loss_, dacc, exact, by_pos = evaluate(model, test_seq)
            print(f"{step:6d}  {loss_:11.3f}  {dacc:9.3f}  {exact:11.3f}  {dacc ** 4:11.3f}")
        if step == STEPS:
            break
        for gr in opt.param_groups:
            gr["lr"] = LR * 0.5 * (1 + math.cos(math.pi * step / STEPS))
        x, y = split(encode(*make_problems(BATCH, g)))
        logits, _ = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    print("  final accuracy by answer digit (ones, tens, hundreds, thousands):",
          " ".join(f"{v:.3f}" for v in by_pos.tolist()), f"({time.time() - t0:.1f} s)")
```

```text

width 24, 15,048 parameters; 128 fresh training problems per step
  step  answer loss  digit acc  exact match  digit_acc^4
     0        2.494      0.051        0.000        0.000
   150        1.738      0.326        0.001        0.011
   300        1.733      0.320        0.000        0.011
   450        1.704      0.339        0.001        0.013
   600        1.293      0.485        0.016        0.055
   750        0.779      0.742        0.091        0.303
   900        0.672      0.758        0.087        0.329
  1050        0.630      0.767        0.093        0.345
  1200        0.614      0.773        0.095        0.356
  1350        0.609      0.774        0.099        0.359
  1500        0.608      0.774        0.097        0.358
  final accuracy by answer digit (ones, tens, hundreds, thousands): 1.000 0.098 0.996 1.000 (7.6 s)

width 48, 57,744 parameters; 128 fresh training problems per step
  step  answer loss  digit acc  exact match  digit_acc^4
     0        2.507      0.073        0.000        0.000
   150        1.731      0.330        0.001        0.012
   300        1.434      0.491        0.007        0.058
   450        0.654      0.755        0.226        0.324
   600        0.160      0.952        0.815        0.821
   750        0.060      0.988        0.952        0.953
   900        0.024      0.997        0.987        0.987
  1050        0.018      0.997        0.989        0.990
  1200        0.012      0.999        0.995        0.995
  1350        0.011      0.999        0.996        0.996
  1500        0.010      0.999        0.996        0.996
  final accuracy by answer digit (ones, tens, hundreds, thousands): 1.000 0.999 0.997 1.000 (17.0 s)
```

Writing the answer least-significant digit first lets each digit be computed from digits already produced plus a carry (exercise 7 reverses it). Section 7 discusses the results. The two runs took 7.6 s and 17.0 s under load, and about 5 s and 8 s in an earlier, quieter run.

## Common mistakes

- **Comparing perplexities across tokenizers.** Use bits per byte, or compare only models with the same tokenizer.
- **Fitting the tokenizer on test data**, or on data unlike deployment. The tokenizer is part of the model.
- **Assuming encode-decode is lossless.** Normalization (lowercasing, Unicode normalization, stripped spaces) can break it; test the round trip.
- **Quoting "20 tokens per parameter" as how models are trained.** It is a training-compute optimum under one setup, since refit; deployed models are trained far past it.
- **Treating a scaling exponent as a constant.** It depends on parameter counting, schedules, data and range.
- **Confusing total and active MoE parameters**, or a context-window number with long-context ability.
- **Declaring emergence from an exact-match plot.** Plot a continuous metric too, and check whether errors are independent before using $p^L$.

## Exercises

1. [warm-up] Run BPE by hand on `low` 5, `lower` 2, `newest` 6, `widest` 3 for three merges; check with `train_bpe` on the words joined by spaces.
2. [warm-up] A 7B model is trained on 2T tokens. How many FLOPs, and how many days on 1,000 GPUs sustaining $4 \times 10^{14}$ FLOP/s each?
3. [warm-up] Under each fit in script 3, what are $N_{\text{opt}}$ and $D_{\text{opt}}$ for $C = 10^{22}$?
4. [core] In script 2, also report training loss in bits per byte, and try 250 and 1,000 merges. Does BPE overfit more?
5. [core] Change `PRETOKEN` to group digit runs (`\d+`). How do years and prices tokenize, and why might one digit per token help arithmetic?
6. [core] Train `bpe.py` on the play plus a few paragraphs of another language. How does each language's tokens per character change with the mix?
7. [core] In script 4, write the answer most-significant digit first. What changes, and why, given what a causal model can see?
8. [core] Why is the optimum in script 3, part 3 flat? Answer using the derivative of $L$ along $6ND = C$.
9. [stretch] Run script 4 for widths 16 to 64 with three seeds each. Does exact match rise more sharply with width than per-digit accuracy, beyond seed noise?
10. [stretch] Implement a small unigram tokenizer (candidates: substrings up to length 6 seen 5 or more times; Viterbi segmentation; prune 20% per round, keeping all bytes) and compare its compression with BPE at equal vocabulary size.

### Hints and answers

1. `e s` (9, tied with `s t`), then `es t` (9), then `l o` (7, tied with `o w`). The code breaks ties toward smaller bytes and gives `es`, `est`, `lo`; the leading-space pairs (` l` has count 6) do not change these.
2. $8.4 \times 10^{22}$ FLOPs; at $4 \times 10^{17}$ FLOP/s, $2.1 \times 10^5$ s or about 2.4 days at perfect utilization, so budget more.
3. Hoffmann: about 6.3B parameters, 263B tokens. Besiroglu: about 9.0B, 184B.
4. Expect a larger train-validation gap for BPE, since it sees each token more often; fewer merges shrink both its advantage and its overfitting.
5. With `\d+`, frequent numbers become single tokens and rare ones split unpredictably, so a digit's token depends on its neighbors. One digit per token gives a consistent place-value code.
6. A language that gets few merges stays near one token per byte; as its share grows it earns merges and English compression drops a little. The vocabulary is a shared budget.
7. The first answer digit then depends on carries from every lower position, which must be computed before anything is written. Expect slower learning and lower exact match.
8. At the optimum, $dL/d\log N$ along the constraint is zero (the parameter penalty falls as fast as the data penalty rises), so a factor of 2 costs only a second-order amount.
9. Seed variation is large at this scale; in an exploratory run with a different setup, width 32 did worse than width 24. Compare the spread across seeds with the differences between widths (L1-01).
10. Compression is usually similar to BPE; unigram's advantage is probabilistic segmentation.

## Check your understanding

1. **Why can byte-level BPE encode any string, and what does that cost?** The base vocabulary is all 256 byte values, so the worst case is one token per byte. The cost is length: text the merges did not anticipate falls back toward bytes and uses more context, compute and money, as the Hindi example shows.

2. **Two models report validation perplexities of 8 and 12. Which is better?** You cannot tell without their tokenizers and test text. Perplexity is per token, and a model with longer tokens can have higher perplexity but lower bits per byte. Compare bits per byte on the same text.

3. **Why do most released small models ignore the compute-optimal ratio?** It minimizes training compute for a given loss, but serving costs about $2N$ FLOPs per token. A smaller model trained on more tokens costs more to train and less to serve, and with heavy use the total is lower.

4. **Did the replication overturn Chinchilla?** No. It found the published parametric fit inconsistent with the paper's other methods, poorly fitting and with implausibly narrow intervals; the corrected fit agrees with the other methods and with roughly 20 tokens per parameter. A widely used set of constants was wrong; the main conclusion was not.

5. **Is emergence real?** Partly. With all-or-nothing metrics, smooth per-token progress looks like a jump, and some emergence disappears under continuous metrics. But learning can also have plateaus and fast transitions visible in the loss itself, and some tasks stay at chance until loss crosses a threshold. Check continuous metrics and the independence of errors before deciding.

## Further reading

- **Rico Sennrich, Barry Haddow, Alexandra Birch, "Neural Machine Translation of Rare Words with Subword Units"** (2016, ACL; arXiv:1508.07909). https://arxiv.org/abs/1508.07909
  *Read for:* the original BPE-for-NLP algorithm, with a short reference implementation. *Level:* intro to intermediate.
- **Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, "Language Models are Unsupervised Multitask Learners"** (2019, OpenAI report). https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  *Read for:* section 2.2 on byte-level BPE, and the case for next-token prediction as multitask learning. *Level:* intermediate.
- **Taku Kudo, "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates"** (2018, ACL; arXiv:1804.10959). https://arxiv.org/abs/1804.10959
  *Read for:* the unigram language model tokenizer and sampled segmentations. *Level:* intermediate.
- **Andrej Karpathy, "minbpe"** (GitHub repository, MIT license). https://github.com/karpathy/minbpe
  *Read for:* compact byte-level BPE code whose regex tokenizer reproduces GPT-4's, with a linked lecture video. *Level:* intro.
- **Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei, "Scaling Laws for Neural Language Models"** (2020, arXiv:2001.08361). https://arxiv.org/abs/2001.08361
  *Read for:* the power laws that made extrapolation the basis of planning. *Level:* intermediate.
- **Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al. (22 authors), "Training Compute-Optimal Large Language Models"** (2022, arXiv:2203.15556). https://arxiv.org/abs/2203.15556
  *Read for:* the three estimation methods, isoFLOP curves and the Chinchilla result. *Level:* intermediate to advanced.
- **Tamay Besiroglu, Ege Erdil, Matthew Barnett, Josh You, "Chinchilla Scaling: A replication attempt"** (2024, arXiv:2404.10102). https://arxiv.org/abs/2404.10102
  *Read for:* a short, careful check of a famous fit, with corrected constants. *Level:* intermediate.
- **Jason Wei, Yi Tay, Rishi Bommasani, et al. (16 authors), "Emergent Abilities of Large Language Models"** (2022, TMLR; arXiv:2206.07682). https://arxiv.org/abs/2206.07682
  *Read for:* the definition and the catalogue of tasks that started the debate. *Level:* intro to intermediate.
- **Rylan Schaeffer, Brando Miranda, Sanmi Koyejo, "Are Emergent Abilities of Large Language Models a Mirage?"** (2023, NeurIPS; arXiv:2304.15004). https://arxiv.org/abs/2304.15004
  *Read for:* the metric-choice argument and its simple mathematical model. *Level:* intro to intermediate.
- **Aaron Grattafiori et al. (Llama Team, Meta), "The Llama 3 Herd of Models"** (2024, arXiv:2407.21783). https://arxiv.org/abs/2407.21783
  *Read for:* the most detailed public account of a frontier-scale run: data, scaling-law use, context extension, annealing and failures at 16K GPUs. *Level:* intermediate to advanced.
- **Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf, "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale"** (2024, arXiv:2406.17557). https://arxiv.org/abs/2406.17557
  *Read for:* each curation step with ablations, and the classifier behind FineWeb-Edu. *Level:* intro to intermediate.
- **Sebastian Raschka, "A Dream of Spring for Open-Weight LLMs: 10 Architectures from Jan-Feb 2026"** (Ahead of AI newsletter, February 25, 2026). https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight
  *Read for:* side-by-side diagrams of recent open-weight architectures. A secondary source; check details against each technical report. *Level:* intro.
