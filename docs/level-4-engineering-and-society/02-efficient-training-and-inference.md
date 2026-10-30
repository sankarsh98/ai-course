---
title: "Efficient Training and Inference"
---

> **Level 4 · Module 2** · Prerequisites: L0-01, L0-05, L2-02, L2-05, L3-01 · Time: 7 to 9 hours · Code: `code/L4-02-efficient-training-and-inference/`

## What you will be able to do

- Account for the memory of a training run (weights, gradients, optimizer state, activations) and predict what mixed precision, activation checkpointing and sharding each save.
- Derive the $6ND$ training-compute rule, state its assumptions, and turn FLOPs into a run time through model FLOPs utilization.
- Explain fp32, fp16, bf16 and fp8, and why mixed precision keeps fp32 master weights and (for fp16) scales the loss.
- Compare data, tensor, pipeline and expert parallelism by the memory each removes and the communication each adds.
- Apply distillation and pruning to a small model and judge compression claims.
- Describe the main LLM serving techniques (continuous batching, paged KV cache, chunked prefill, disaggregation, prefix caching) and what each trades.

## 1. Scope, and what AI Systems already covers

Training is limited by memory capacity, arithmetic and communication between devices; inference is paid on every request. For inference this module leans on the external "AI Systems" course, Track B, which explains why decoding is limited by memory bandwidth and teaches the KV cache, quantization and speculative decoding with runnable code: [why inference is hard](https://sankarsh98.github.io/ai-systems/inference/prefill-and-decode), [the KV cache](https://sankarsh98.github.io/ai-systems/inference/kv-cache), [quantization](https://sankarsh98.github.io/ai-systems/inference/quantization), [speculative decoding](https://sankarsh98.github.io/ai-systems/inference/speculative-decoding). Read them before section 7; they are not repeated here.

This module covers what they skip: memory accounting (section 2), FLOPs (3), numeric formats (4), parallelism (5), distillation and pruning (6), and serving at a survey level (7). Its central tool is a memory-and-FLOPs calculator, checked byte for byte against the GPT from L2-05.

Notation: $L$ layers, width $d$, $H$ heads, vocabulary $V$, sequence length $T$, batch of $B$ sequences, $N$ parameters, $D$ training tokens. $1\text{ GB} = 10^9$ bytes.

## 2. Memory accounting for training

### 2.1 Model state

Each parameter needs a weight, a gradient, and the optimizer's state; Adam (L0-05) keeps two running averages. In fp32 that is $4 + 4 + 8 = 16$ bytes per parameter. **Mixed precision** runs forward and backward in 16 bits but updates an fp32 master copy. ZeRO's accounting for Adam: 2 bytes of 16-bit weights, 2 of 16-bit gradients, 12 of fp32 master weights, momentum and variance, so $16\Psi$ bytes for $\Psi$ parameters; a 1.5B-parameter GPT-2 needs at least 24 GB for this, against 3 GB for its 16-bit weights [Rajbhandari et al., 2019].

So **mixed precision does not shrink the model state**. Script 1 gives 115.87 GB for a 7.24B-parameter model in both recipes. What shrinks it: dropping the fp32 copy (risky, section 4), a lighter optimizer, or sharding (2.4).

### 2.2 Activations

The backward pass needs forward values (L0-03, L2-01), which autograd stores. Per token and layer, a pre-LN GPT block like L2-05's stores the two layer-norm inputs ($2d$ numbers, plus 4 scalars of statistics), the attention input ($d$), queries, keys and values ($3d$), the attention output ($d$), the MLP input ($d$), and the MLP hidden layer before and after GELU ($4d$ each): $16d$ numbers. A fused attention kernel (L2-05) adds one log-sum-exp per head instead of the $T \times T$ probabilities. In 16 bits:

$$M_{\text{act}} \approx 32\,d \cdot L \cdot B \cdot T \text{ bytes}.$$

Korthikanti et al.'s formula for Megatron's GPT, $sbh(34 + 5as/h)$ bytes per layer ($s = T$, $b = B$, $h = d$, $a = H$), is the same count plus 1-byte dropout masks ($2sbh$) and, for unfused attention, the stored softmax, its mask and its dropout output ($5as^2b$) [Korthikanti et al., 2022]. With those terms switched on, script 1 matches their equation for the GPT-2 shape (89.7 MB per layer per 1,024-token sequence).

Activations scale with $B \cdot T$, and without a fused kernel the $T^2$ term dominates: for a Mistral-7B shape at 4,096 tokens, 18.9 GB per sequence fused versus 105.8 GB unfused with dropout. Watch the **logits** too: $V$ fp32 numbers per token, 206 MB per sequence for the GPT-2 shape, 40% of its 0.51 GB.

**Validation.** Script 2 registers `torch.autograd.graph.saved_tensors_hooks`, which sees every tensor saved for backward, and sums their storage. For L2-05's model ($L = 4$, $d = 96$, $B = 8$, $T = 128$) measurement and calculator agree to the byte: 26,379,268 in fp32. Under bf16 autocast both say 16,070,020, only 39% less, because autocast keeps the residual stream in fp32 and saves a bf16 copy of each weight matrix for backward. Accounting rules miss such details unless checked.

### 2.3 Activation checkpointing

**Activation checkpointing** stores only some activations and recomputes the rest in the backward pass. Chen et al. reduced memory to $O(\sqrt{n})$ for $n$ layers at the cost of about one extra forward pass [Chen et al., 2016]. Checkpointing each transformer block's input leaves $L \cdot B \cdot T \cdot d$ values, plus one block's activations during recomputation. Since backward costs about twice forward, the extra forward adds roughly a third. In script 2, checkpointing cut saved bytes from 26.4 MB to 2.66 MB (as predicted), gave bitwise-identical gradients, and slowed a step by 1.20 to 1.31 times over five runs. Korthikanti et al. recompute only the cheap, large pieces and, with sequence parallelism, report 5 times less activation memory, over 90% less recomputation overhead, and 54.2% versus 42.1% model FLOPs utilization on a 530B model [Korthikanti et al., 2022].

### 2.4 Sharding: ZeRO and FSDP

Plain data parallelism (section 5) replicates the state on every device. ZeRO shards it over $N_d$ devices in stages: optimizer state ($4 + 12/N_d$ bytes per parameter), then gradients ($2 + 14/N_d$), then weights ($16/N_d$). Stages 1 and 2 keep data parallelism's communication volume of $2\Psi$ per step; stage 3 needs $3\Psi$ because weights are gathered for both passes [Rajbhandari et al., 2019]. PyTorch's FSDP implements the idea (Zhao et al., 2023, arXiv:2304.11277). For the 7B shape over 8 devices, script 1 gives 39.8, 27.2 and 14.5 GB per device. ZeRO also warns about temporary buffers (6 GB for a flattened fp32 gradient of 1.5B parameters) and fragmentation [Rajbhandari et al., 2019]: leave headroom.

## 3. FLOPs accounting and $6ND$

### 3.1 Derivation

Count a multiply-add as 2 FLOPs, so a vector times an $m \times n$ matrix costs $2mn$. Each linear-layer weight is used once per token forward: **$2N$ per token**. Backward computes two products per forward product, the input gradient and the weight gradient (L2-01): **$4N$**. Hence $C \approx 6ND$. Attention's $QK^\top$ and weighted sum cost $2Td$ each per token and layer forward; counted over the full context as in PaLM's appendix, training costs

$$6N + 12LTd \text{ FLOPs per token}$$

[Chowdhery et al., 2022]. Kaplan et al. write the forward context term as $2 n_{\text{layer}} n_{\text{ctx}} d_{\text{attn}}$, half of PaLM's, and drop it because their models had $d \gg T/12$ [Kaplan et al., 2020]; conventions differ by two, so state yours. With $N \approx 12Ld^2$, attention's share is about $T/(6d)$: 22% for GPT-2 small at $T = 1{,}024$.

### 3.2 Assumptions behind $6ND$

1. $N$ counts weights used in matrix products: embedding lookups are free, while a tied output head costs $2Vd$ forward.
2. Attention is ignored, which needs $T \ll 6d$.
3. No recomputation; full checkpointing makes it about $8N$. Megatron's formula $96BsLh^2(1 + s/6h + V/16Lh)$ includes recomputation [Narayanan et al., 2021].
4. Dense models; for mixture of experts use the active parameters.
5. Elementwise work (norms, softmax, optimizer) is left out: small in FLOPs, not in time.

**Validation.** `FlopCounterMode` measures 3,461,480,448 FLOPs for one training step of L2-05's model on 1,024 tokens (script 2). The matrix products match the calculator exactly. The 3% gap is attention's backward, counted at 2.5 times its forward because the fused kernel recomputes the scores. Plain $6ND$ with all 466,176 parameters is 17% low, mostly because $T/(6d) = 0.22$.

### 3.3 From FLOPs to time

**Model FLOPs utilization** (MFU) is achieved tokens per second divided by what peak FLOP/s would give if the step needed only the model's FLOPs, without recomputation. PaLM reported 46.2% for its 540B model and computed 21.3% for GPT-3, 32.5% for Gopher and 30.2% for Megatron-Turing NLG [Chowdhery et al., 2022]. So

$$\text{time} \approx \frac{(6N + 12LTd)\,D}{\text{devices} \times \text{peak FLOP/s} \times \text{MFU}}.$$

Script 1: a 7B shape on 20 tokens per parameter needs $7.1 \times 10^{21}$ FLOPs, about 206 device-days at 40% MFU on an illustrative $10^{15}$ FLOP/s device. L3-01 checks $6ND$ against Llama 3.

## 4. Numeric formats and mixed precision

A float spends bits on exponent (range) and mantissa (precision). From script 3:

| format | exp/mantissa bits | epsilon | max | min normal |
|---|---|---|---|---|
| fp32 | 8/23 | $1.2 \times 10^{-7}$ | $3.4 \times 10^{38}$ | $1.2 \times 10^{-38}$ |
| fp16 | 5/10 | $9.8 \times 10^{-4}$ | 65,504 | $6.1 \times 10^{-5}$ |
| bf16 | 8/7 | $7.8 \times 10^{-3}$ | $3.4 \times 10^{38}$ | $1.2 \times 10^{-38}$ |
| fp8 E4M3 | 4/3 | 0.125 | 448 | $1.6 \times 10^{-2}$ |
| fp8 E5M2 | 5/2 | 0.25 | 57,344 | $6.1 \times 10^{-5}$ |

**fp16 lacks range.** In script 3, 44.7% of a million log-uniform values between $10^{-12}$ and $10^{-2}$ become zero in fp16. Micikevicius et al. made fp16 training work with fp32 master weights, **loss scaling** (multiply the loss before backward so small gradients survive, divide afterwards) and fp32 accumulation [Micikevicius et al., 2018]. Scaling by $2^{16}$ rescues all those values but sends gradients of 1 and 10 to infinity, so real scalers adapt the factor and skip overflowing steps.

**bf16 lacks precision.** It has fp32's exponent, so nothing underflows; Kalamkar et al. matched fp32 results across many tasks with no hyperparameter changes [Kalamkar et al., 2019], and PyTorch's CPU bf16 recipe uses autocast without `GradScaler` [PyTorch, AMP]. But an update smaller than half the spacing of representable numbers vanishes: near 1.0 that spacing is $2^{-7}$, so adding $10^{-4}$ a thousand times leaves a bf16 weight at 1.0. With Adam at learning rate $10^{-4}$ on weights near 1 (the late, decayed phase of training), fp32 reaches loss $2.4 \times 10^{-11}$, bf16 compute with fp32 master weights $2.1 \times 10^{-4}$, and pure bf16 barely moves from $6.8 \times 10^{-3}$. Sums behave the same way: a 4,096-term dot product accumulated in bf16 gave $-51.0$ against an exact $-52.25$. Hence 16-bit multiplies with fp32 accumulators and master weights.

**fp8.** Micikevicius et al. (2022) proposed E4M3 for weights and activations and E5M2 for gradients, with scale factors, and matched 16-bit quality up to 175B parameters [Micikevicius et al., 2022]. DeepSeek-V3's report describes FP8 mixed-precision training of that model (DeepSeek-AI, 2024, arXiv:2412.19437). NVIDIA's NVFP4 paper calls FP8 training "widely adopted" and reports a 12B model trained on 10 trillion tokens in a 4-bit format, comparable to an FP8 baseline (NVIDIA, 2025, arXiv:2509.25149). These are the authors' claims; I could not independently check industry-wide practice as of October 2026.

**Low precision is fast only on hardware built for it.** On this laptop's CPU (Intel i7-13700HX: AVX2 and AVX-VNNI, no AVX-512 bf16 or AMX), a 1024-by-1024 matrix product took 4.3 ms in fp32 and 1,553 ms in bf16, presumably a slow fallback path. On accelerators with 16-bit or 8-bit matrix units, the ratio reverses. Measure on the target.

## 5. Parallelism and communication

**Collectives.** All-reduce gives everyone the sum of everyone's tensors; reduce-scatter gives each device the sum of one shard; all-gather gives everyone all shards; all-to-all sends a different piece to each device. A ring all-reduce is a reduce-scatter then an all-gather, each $n - 1$ steps of $S/n$ bytes, so each device sends

$$2\,\frac{n-1}{n}\,S \text{ bytes},$$

nearly independent of $n$. Time is bytes over bandwidth plus a per-message latency, so frameworks bucket gradients and overlap communication with the backward pass.

**Data parallelism.** Replicas process different slices of the batch and average gradients by all-reduce; with equal slices this equals the full-batch gradient. Script 4 checks it with four real processes (gloo backend): difference $1.6 \times 10^{-7}$, float32 rounding. For 7.24B parameters with bf16 gradients, each device sends about 25 GB per step, a quarter second at an illustrative 100 GB/s (script 1). Sharding (2.4) removes the replicated state.

**Tensor parallelism** splits weight matrices. Megatron-LM cuts the MLP's first matrix by columns, so each device applies GELU to its own slice, and the second by rows, so one all-reduce sums the partial outputs; attention splits by heads. A layer needs two all-reduces of $B \cdot T \cdot d$ activations forward and two backward [Shoeybi et al., 2019]. In script 4 the split MLP matches the full one to $7.7 \times 10^{-7}$, while a row split of the first matrix with GELU applied per device is off by up to 1.75, because GELU of a sum is not the sum of GELUs. The traffic sits on every layer's critical path, so Narayanan et al. advise tensor parallelism up to the GPUs in one server and pipeline parallelism across servers [Narayanan et al., 2021].

**Pipeline parallelism** gives each device a block of layers; only boundary activations move. GPipe splits a batch into $m$ micro-batches, runs all forwards then all backwards (Huang et al., 2018, arXiv:1811.06965), and leaves a **bubble** of $(p-1)/m$ of the ideal time for $p$ stages [Narayanan et al., 2021], or $(p-1)/(m+p-1)$ of the total. Script 4's simulated schedule matches: 43% idle with 4 stages and 4 micro-batches, 16% with 16. The PipeDream-Flush ("1F1B") schedule keeps the bubble but holds activations for at most $p$ micro-batches instead of $m$; interleaving several smaller stages per device divides the bubble by their number at the cost of more communication [Narayanan et al., 2021].

**Expert parallelism** places a mixture-of-experts layer's experts (L3-01) on different devices; tokens travel to their experts and back by all-to-all (Fedus et al., 2021, arXiv:2101.03961; Lepikhin et al., 2020, arXiv:2006.16668). Compute per token stays small, but all experts occupy memory, load imbalance stalls the all-to-all, and Switch Transformers names communication costs and instability as obstacles. **Sequence parallelism** splits the parts of a layer outside the tensor-parallel matrix products (norms, dropout) along the sequence dimension, cutting their activation memory [Korthikanti et al., 2022].

Large runs combine them: tensor parallelism within servers, pipelines across, sharded data parallelism over the rest. Narayanan et al. reached 502 petaFLOP/s (52% of peak) on 3,072 GPUs for a 1-trillion-parameter configuration [Narayanan et al., 2021]. The Ultra-Scale Playbook works through these choices with measurements [Hugging Face, 2025].

## 6. Distillation and pruning

Quantization is AI Systems lesson 3.

**Distillation** trains a small student to match a teacher's outputs. Hinton et al. soften both with a temperature $T$, $p_i \propto \exp(z_i/T)$, so the student learns which wrong answers are plausible, and multiply the soft-target loss by $T^2$ because its gradients scale as $1/T^2$ [Hinton et al., 2015]. DistilBERT reported a model 40% smaller than BERT keeping 97% of its language understanding and 60% faster (Sanh et al., 2019, arXiv:1910.01108). In LLM post-training, "distillation" often means fine-tuning on a stronger model's outputs (L3-02).

Script 5 uses scikit-learn's digits: 300 labeled images, 800 unlabeled, 697 for testing. The teacher (301,066 parameters) scores 0.940; a 1,482-parameter student trained on labels, 0.908 (mean of 5 seeds). Distilling on the same 300 images helps a little and inconsistently (0.919, range 0.895 to 0.930). Letting the teacher label the 800 extra images is what matters: hard pseudo-labels give 0.930 and soft targets 0.936, near the teacher at one two-hundredth of the size. Here the **transfer set** does most of the work and soft targets add a little; one small dataset cannot show how general that is.

**Pruning** removes weights: **unstructured** (individual weights, often the smallest in magnitude) or **structured** (whole neurons, heads, channels, which dense hardware can skip). In script 5, removing 90% of the teacher's weights drops accuracy to 0.763 and 20 epochs of fine-tuning restore 0.940; at 95%, 0.902; at 98%, 0.257, with 7,035 nonzero parameters, nearly five times the dense student that scored 0.936. Stored densely, 50%, 90% or 99% zeros multiply in the same 1.1 ms as none; a CSR sparse format was slower at 50% and faster at 90% and above on this CPU. GPUs exploit unstructured sparsity poorly; NVIDIA's Ampere supports a 2:4 pattern at twice dense math throughput (Mishra et al., 2021, arXiv:2104.08378). SparseGPT prunes 175B-parameter models to 50 to 60% sparsity in one shot with little perplexity loss (Frantar and Alistarh, 2023, arXiv:2301.00774), and the lottery ticket hypothesis proposes that dense networks contain sparse subnetworks that train well from their initial weights (Frankle and Carbin, ICLR 2019, arXiv:1803.03635).

Be skeptical of pruning claims: Blalock et al. reviewed 81 papers and found comparisons so inconsistent that progress was hard to measure [Blalock et al., 2020]. Ask about the baseline (including a small dense model), whether speed was measured or inferred from FLOPs, and whether the hardware accelerates the sparsity pattern.

## 7. Serving at a survey level

Generation has a compute-bound **prefill** over the prompt and a bandwidth-bound **decode** of one token per step (AI Systems lesson 1). Metrics: **time to first token** (TTFT), **time per output token** (TPOT), and throughput. The main serving ideas:

- **Continuous batching.** Static batches run until their longest request ends. Orca's iteration-level scheduling lets requests join and leave at every decode step; it reported 36.9 times FasterTransformer's throughput on GPT-3 175B at equal latency [Yu et al., 2022].
- **Paged KV cache.** Reserving KV memory for the maximum length wastes most of it: Kwon et al. found only 20.4% to 38.2% of KV memory held real token states in the systems they profiled. PagedAttention allocates fixed-size blocks on demand, as an OS pages memory, nearly eliminating waste and letting sequences share blocks; vLLM reported 2 to 4 times the throughput of FasterTransformer and Orca at equal latency, and preempts the latest requests when memory runs out [Kwon et al., 2023].
- **Chunked prefill.** Sarathi-Serve splits long prefills into chunks mixed with decodes so they do not stall them, reporting 2.6 times vLLM's serving capacity for Mistral-7B on one A100 under tail-latency limits (Agrawal et al., 2024, arXiv:2403.02310).
- **Disaggregation.** DistServe puts prefill and decode on separate GPUs, reporting 7.4 times more requests or 12.6 times tighter latency targets than its baselines (Zhong et al., OSDI 2024, arXiv:2401.09670).
- **Prefix caching.** Requests sharing a prefix reuse its KV cache; SGLang keeps prefixes in a radix tree (RadixAttention) and reports up to 6.4 times higher throughput across agent, few-shot and chat workloads from this and its other optimizations (Zheng et al., 2023, arXiv:2312.07104).
- **Fewer bytes, fewer passes:** quantization, grouped-query attention and speculative decoding (AI Systems lessons 2 to 4); fused attention (L2-05); tensor parallelism across GPUs for models that do not fit on one.

All figures are the authors' benchmarks against the baselines of their day; gains against current engines, which combine most of these, are smaller and not additive. As of October 2026, PyPI lists vLLM 0.32.0 (October 18, 2026) and SGLang 0.5.23 (October 30, 2026); I have not benchmarked either.

## Hands-on

Run from `code/L4-02-efficient-training-and-inference/` in the course environment. Scripts 2 and 3 import `gpt.py` and the corpus from `code/L2-05-transformers/` unchanged. Timings are from an Intel i7-13700HX laptop at load average 0.3 to 2.1; counts and bytes should match exactly on any machine with the same library versions.

### Script 1: the calculator, `calculator.py`

The Mistral-7B shape takes Table 1 of Jiang et al. (2023, arXiv:2310.06825) and adds assumptions the table does not state (gated MLP, RMSNorm, no biases, untied head), so its 7.24B is an estimate for a model the paper calls "7-billion-parameter". The gated-MLP activation term has not been checked against a real model (exercise 5).

```python
"""Memory, FLOPs and communication calculator for decoder-only transformers.

Pure arithmetic: no model is built. validate_calculator.py checks these formulas
against a real PyTorch model (the GPT from L2-05).

Symbols: L layers, d model width, H heads, V vocabulary, T sequence length,
B sequences per batch, N parameters. Bytes, not bits, everywhere.
"""
from dataclasses import dataclass


@dataclass
class Shape:
    n_layer: int
    d_model: int
    n_head: int
    vocab: int
    seq_len: int
    d_ff: int = 0             # 0 means 4 * d_model
    n_kv_head: int = 0        # 0 means n_head (no grouped-query attention)
    gated_mlp: bool = False   # SwiGLU-style MLP: three matrices instead of two
    bias: bool = True         # biases in linear layers
    layernorm: bool = True    # LayerNorm (weight + bias) vs RMSNorm (weight only)
    tied: bool = True         # output head shares the token-embedding matrix
    learned_pos: bool = True  # learned position table (False for rotary)

    def __post_init__(self):
        self.d_ff = self.d_ff or 4 * self.d_model
        self.n_kv_head = self.n_kv_head or self.n_head


def params(s):
    """Parameter count by component. 'matmul' = weights used in a matmul per token."""
    d, hd = s.d_model, s.d_model // s.n_head
    kv = 2 * d * s.n_kv_head * hd                    # key and value projections
    attn = d * d + kv + d * d                        # query, key/value, output
    mlp = (3 if s.gated_mlp else 2) * d * s.d_ff
    norm = 2 * d if s.layernorm else d
    biases = (d + 2 * s.n_kv_head * hd + d + s.d_ff + d) if s.bias else 0
    layer = attn + mlp + 2 * norm + biases
    emb = s.vocab * d + (s.seq_len * d if s.learned_pos else 0)
    head = 0 if s.tied else s.vocab * d
    total = emb + s.n_layer * layer + norm + head
    matmul = s.n_layer * (attn + mlp) + s.vocab * d  # head matmul runs either way
    return dict(embeddings=emb, per_layer=layer, total=total, matmul=matmul)


# Bytes per parameter for model state (weights + gradients + optimizer state).
RECIPES = {
    "fp32 weights, fp32 Adam": dict(weights=4, grads=4, optim=8),
    "bf16 weights+grads, fp32 master+Adam": dict(weights=2, grads=2, optim=12),
    "bf16 everything (no fp32 master)": dict(weights=2, grads=2, optim=4),
    "fp32 SGD with momentum": dict(weights=4, grads=4, optim=4),
}


def state_bytes(n, recipe, zero_stage=0, n_dp=1):
    """Model-state bytes per device. ZeRO shards optimizer (1), +grads (2), +weights (3)."""
    r = RECIPES[recipe]
    w, g, o = r["weights"], r["grads"], r["optim"]
    if zero_stage >= 1:
        o /= n_dp
    if zero_stage >= 2:
        g /= n_dp
    if zero_stage >= 3:
        w /= n_dp
    return n * (w + g + o)


def activation_bytes(s, batch, act=2, res=None, flash=True, dropout=False,
                     checkpoint=False, autocast_copies=False):
    """Bytes autograd keeps between forward and backward.

    act: bytes per element of matmul inputs/outputs; res: bytes of the residual
    stream (layer-norm inputs), defaults to act. Per token per layer we save:
    two norm inputs (2d at res bytes) plus their mean and 1/std (4 fp32 numbers),
    and 14d at act bytes: norm-1 output, q, k, v, attention output, norm-2
    output, MLP hidden before and after the nonlinearity (4d each).
    flash=True: the attention kernel keeps one fp32 log-sum-exp per head and
    row instead of the T x T probability matrix.
    """
    res = act if res is None else res
    tok = batch * s.seq_len
    d = s.d_model
    hd = d // s.n_head
    # q (d) plus k and v (shrunk by grouped-query attention) instead of 3d
    qkv = d + 2 * s.n_kv_head * hd
    mlp = (3 if s.gated_mlp else 2) * s.d_ff     # SwiGLU keeps gate and up outputs
    per_tok = 2 * d * res + 16 + act * (d + qkv + d + d + mlp)
    if flash:
        per_tok += 4 * s.n_head                     # fp32 log-sum-exp per head
    else:
        per_tok += s.n_head * s.seq_len * act       # softmax probabilities
    if dropout:   # 1-byte masks after attention and MLP, as in Korthikanti et al.
        per_tok += 2 * d
        if not flash:   # attention-probability dropout: mask + its 16-bit output
            per_tok += s.n_head * s.seq_len * (1 + act)
    layer = per_tok * tok
    # After the last block: final-norm input + stats, head input, fp32 logits,
    # plus the int64 token ids, targets and positions, and one fp32 scalar
    # kept by the loss reduction.
    tail = tok * (d * res + 8 + d * act + 4 * s.vocab) + 8 * (2 * tok + s.seq_len) + 4
    if checkpoint:   # keep only each block's input; recompute the rest
        saved = s.n_layer * tok * d * res + tail
        peak = saved + layer                       # one block re-materialized
    else:
        saved = s.n_layer * layer + tail
        peak = saved
    if autocast_copies:  # autocast saves a 16-bit copy of every matmul weight
        saved += 2 * params(s)["matmul"]
        peak += 2 * params(s)["matmul"]
    return dict(saved=saved, peak=peak, per_layer=layer)


def korthikanti_per_layer(s, batch):
    """Eq. (1) of Korthikanti et al. (2022): s*b*h*(34 + 5*a*s/h) bytes, 16-bit."""
    t, b, h, a = s.seq_len, batch, s.d_model, s.n_head
    return t * b * h * (34 + 5 * a * t / h)


def flops_per_token(s, train=True, recompute=False):
    """Matmul FLOPs per token: 2 per multiply-add. Attention scores and the
    weighted sum add 4*T*d per layer forward (no saving from the causal mask)."""
    fwd = 2 * params(s)["matmul"] + 4 * s.n_layer * s.seq_len * s.d_model
    if not train:
        return fwd
    return fwd * (4 if recompute else 3)   # backward ~ 2x forward, +1 forward if recomputing


def allreduce_bytes(payload, n):
    """Bytes each device sends in a ring all-reduce (reduce-scatter + all-gather)."""
    return 2 * (n - 1) / n * payload


def pipeline_bubble(p, m):
    """Idle fraction of a GPipe-style schedule with p stages and m micro-batches."""
    return (p - 1) / (m + p - 1)


def gb(x):
    return f"{x / 1e9:8.2f} GB"


SHAPES = {
    # GPT-2 small layout (Radford et al., 2019): learned positions, tied head.
    "GPT-2-small shape": Shape(n_layer=12, d_model=768, n_head=12, vocab=50257, seq_len=1024),
    # Table 1 of Jiang et al. (2023), Mistral 7B. Assumed (not in the table):
    # SwiGLU MLP, RMSNorm, no biases, untied head, rotary positions.
    "Mistral-7B shape": Shape(n_layer=32, d_model=4096, n_head=32, n_kv_head=8, vocab=32000,
                              seq_len=4096, d_ff=14336, gated_mlp=True, bias=False,
                              layernorm=False, tied=False, learned_pos=False),
}

if __name__ == "__main__":
    PEAK = 1e15        # illustrative accelerator peak, FLOP/s (not a specific chip)
    LINK = 100e9       # illustrative per-device network bandwidth, bytes/s
    for name, s in SHAPES.items():
        p = params(s)
        n = p["total"]
        print(f"== {name}: L={s.n_layer} d={s.d_model} H={s.n_head} V={s.vocab} T={s.seq_len}")
        print(f"parameters {n:,}  (embeddings {p['embeddings']:,}, matmul {p['matmul']:,})")
        print("model state per device (weights + grads + optimizer):")
        for r in RECIPES:
            print(f"  {r:<38}{gb(state_bytes(n, r))}")
        mp = "bf16 weights+grads, fp32 master+Adam"
        for z in (1, 2, 3):
            print(f"  same, ZeRO stage {z} over 8 devices      {gb(state_bytes(n, mp, z, 8))}")
        print("activations for ONE sequence, bf16, fused attention:")
        a = activation_bytes(s, 1)
        ac = activation_bytes(s, 1, checkpoint=True)
        an = activation_bytes(s, 1, flash=False, dropout=True)
        print(f"  stored, no recompute{gb(a['saved'])}   with per-layer checkpointing"
              f"{gb(ac['saved'])} (peak {gb(ac['peak']).strip()})")
        print(f"  unfused attention + dropout{gb(an['saved'])}"
              f"   one layer: {an['per_layer'] / 1e6:.1f} MB vs Korthikanti eq. 1:"
              f" {korthikanti_per_layer(s, 1) / 1e6:.1f} MB")
        f = flops_per_token(s)
        print(f"training FLOPs per token {f:.3e}  vs 6N = {6 * n:.3e}  (ratio {f / (6 * n):.3f})")
        tokens = 20 * n
        days = tokens * f / (PEAK * 0.4) / 86400
        print(f"{tokens:.2e} tokens (20 per parameter): {tokens * f:.2e} FLOPs,"
              f" {days:,.1f} device-days at 40% of {PEAK:.0e} FLOP/s")
        grad = 2 * n   # bf16 gradients
        for dp in (8, 64):
            t = allreduce_bytes(grad, dp) / LINK
            print(f"  data-parallel all-reduce of bf16 grads over {dp:>2}: "
                  f"{allreduce_bytes(grad, dp) / 1e9:6.2f} GB sent per device, {t * 1e3:5.0f} ms at 100 GB/s")
        print()
    print("pipeline bubble (p-1)/(m+p-1):")
    for p_ in (4, 8):
        print(f"  p={p_}: " + "  ".join(f"m={m}: {pipeline_bubble(p_, m):.0%}" for m in (1, 4, 16, 64)))
```

```text
== GPT-2-small shape: L=12 d=768 H=12 V=50257 T=1024
parameters 124,439,808  (embeddings 39,383,808, matmul 123,532,032)
model state per device (weights + grads + optimizer):
  fp32 weights, fp32 Adam                   1.99 GB
  bf16 weights+grads, fp32 master+Adam      1.99 GB
  bf16 everything (no fp32 master)          1.00 GB
  fp32 SGD with momentum                    1.49 GB
  same, ZeRO stage 1 over 8 devices          0.68 GB
  same, ZeRO stage 2 over 8 devices          0.47 GB
  same, ZeRO stage 3 over 8 devices          0.25 GB
activations for ONE sequence, bf16, fused attention:
  stored, no recompute    0.51 GB   with per-layer checkpointing    0.23 GB (peak 0.25 GB)
  unfused attention + dropout    1.29 GB   one layer: 89.7 MB vs Korthikanti eq. 1: 89.7 MB
training FLOPs per token 8.544e+08  vs 6N = 7.466e+08  (ratio 1.144)
2.49e+09 tokens (20 per parameter): 2.13e+18 FLOPs, 0.1 device-days at 40% of 1e+15 FLOP/s
  data-parallel all-reduce of bf16 grads over  8:   0.44 GB sent per device,     4 ms at 100 GB/s
  data-parallel all-reduce of bf16 grads over 64:   0.49 GB sent per device,     5 ms at 100 GB/s

== Mistral-7B shape: L=32 d=4096 H=32 V=32000 T=4096
parameters 7,241,732,096  (embeddings 131,072,000, matmul 7,110,393,856)
model state per device (weights + grads + optimizer):
  fp32 weights, fp32 Adam                 115.87 GB
  bf16 weights+grads, fp32 master+Adam    115.87 GB
  bf16 everything (no fp32 master)         57.93 GB
  fp32 SGD with momentum                   86.90 GB
  same, ZeRO stage 1 over 8 devices         39.83 GB
  same, ZeRO stage 2 over 8 devices         27.16 GB
  same, ZeRO stage 3 over 8 devices         14.48 GB
activations for ONE sequence, bf16, fused attention:
  stored, no recompute   18.86 GB   with per-layer checkpointing    1.67 GB (peak 2.24 GB)
  unfused attention + dropout  105.82 GB   one layer: 3288.4 MB vs Korthikanti eq. 1: 3254.8 MB
training FLOPs per token 4.910e+10  vs 6N = 4.345e+10  (ratio 1.130)
1.45e+11 tokens (20 per parameter): 7.11e+21 FLOPs, 205.8 device-days at 40% of 1e+15 FLOP/s
  data-parallel all-reduce of bf16 grads over  8:  25.35 GB sent per device,   253 ms at 100 GB/s
  data-parallel all-reduce of bf16 grads over 64:  28.51 GB sent per device,   285 ms at 100 GB/s

pipeline bubble (p-1)/(m+p-1):
  p=4: m=1: 75%  m=4: 43%  m=16: 16%  m=64: 4%
  p=8: m=1: 88%  m=4: 64%  m=16: 30%  m=64: 10%
```

### Script 2: checking the calculator against a real model, `validate_calculator.py`

```python
"""Check calculator.py against a real model: the small GPT from L2-05.

1. Parameter counts for several shapes, against model.parameters().
2. Weight, gradient and Adam-state bytes, measured after one optimizer step.
3. Activation bytes, measured with saved-tensor hooks: every tensor autograd
   stores for the backward pass passes through `pack`, so we can add up their
   storage (counting shared storage once and skipping the parameters).
   Done for fp32, bf16 autocast, and per-block activation checkpointing.
4. FLOPs, against PyTorch's FlopCounterMode.
5. The time cost of checkpointing. Timings depend on the machine and its load.
"""
import sys
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.flop_counter import FlopCounterMode

sys.dont_write_bytecode = True   # leave no __pycache__ in the L2-05 folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "L2-05-transformers"))
from gpt import GPT, GPTConfig  # noqa: E402  (the L2-05 model, unchanged)

from calculator import Shape, activation_bytes, flops_per_token, params, state_bytes

torch.manual_seed(0)
torch.set_num_threads(4)


def shape_of(cfg):
    return Shape(n_layer=cfg.n_layer, d_model=cfg.n_embd, n_head=cfg.n_head,
                 vocab=cfg.vocab_size, seq_len=cfg.block_size)


class Checkpointed(torch.nn.Module):
    """Run a block without saving its internals; recompute them in backward.
    use_reentrant=True saves the block input with save_for_backward, so our
    hook sees it; the non-reentrant variant hides it from outer hooks."""
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return checkpoint(self.block, x, use_reentrant=True)


def saved_bytes(model, x, y, autocast=False):
    param_ptrs = {p.untyped_storage().data_ptr() for p in model.parameters()}
    store = {}

    def pack(t):
        s = t.untyped_storage()
        if s.data_ptr() not in param_ptrs:       # parameters are not activations
            store[s.data_ptr()] = s.nbytes()     # views share storage: count once
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            _, loss = model(x, y)
    return sum(store.values()), loss


def state_measured(model, opt):
    w = sum(p.numel() * p.element_size() for p in model.parameters())
    g = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters())
    o = sum(t.numel() * t.element_size() for st in opt.state.values()
            for t in st.values() if torch.is_tensor(t))
    return w, g, o


print("1. parameter counts")
for kw in [dict(vocab_size=66, block_size=128, n_layer=4, n_head=4, n_embd=96),
           dict(vocab_size=1000, block_size=64, n_layer=2, n_head=8, n_embd=256),
           dict(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768)]:
    cfg = GPTConfig(**kw)
    real = sum(p.numel() for p in GPT(cfg).parameters())
    print(f"  L={cfg.n_layer:>2} d={cfg.n_embd:>3} V={cfg.vocab_size:>5}: model {real:>11,}"
          f"  calculator {params(shape_of(cfg))['total']:>11,}")

cfg = GPTConfig(vocab_size=66, block_size=128, n_layer=4, n_head=4, n_embd=96)
s, B = shape_of(cfg), 8
x = torch.randint(0, 66, (B, cfg.block_size))
y = torch.randint(0, 66, (B, cfg.block_size))

print("\n2. model state after one AdamW step (fp32), bytes")
torch.manual_seed(0)
model = GPT(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
_, loss = model(x, y)
loss.backward()
opt.step()
w, g, o = state_measured(model, opt)
n = params(s)["total"]
print(f"  weights {w:,}  grads {g:,}  Adam state {o:,}  total {w + g + o:,}")
print(f"  calculator: {state_bytes(n, 'fp32 weights, fp32 Adam'):,.0f}"
      f"  (the difference is Adam's per-tensor step counters: {o - 8 * n} bytes)")

print("\n3. activation bytes saved for backward, B=8, T=128")
grads = {}
for mode in ("fp32", "bf16 autocast", "checkpointed"):
    torch.manual_seed(0)
    model = GPT(cfg)
    if mode == "checkpointed":
        model.blocks = torch.nn.ModuleList(Checkpointed(b) for b in model.blocks)
    measured, loss = saved_bytes(model, x, y, autocast=(mode == "bf16 autocast"))
    if mode == "fp32":
        pred = activation_bytes(s, B, act=4)["saved"]
    elif mode == "bf16 autocast":   # residual stream stays fp32; matmul tensors are bf16
        pred = activation_bytes(s, B, act=2, res=4, autocast_copies=True)["saved"]
    else:
        pred = activation_bytes(s, B, act=4, checkpoint=True)["saved"]
    loss.backward()
    grads[mode] = torch.cat([p.grad.flatten() for p in model.parameters()])
    print(f"  {mode:<14} measured {measured:>11,}  calculator {pred:>11,.0f}  loss {loss.item():.6f}")
diff = (grads["fp32"] - grads["checkpointed"]).abs().max().item()
print(f"  max |grad difference| fp32 vs checkpointed: {diff:.1e}")

print("\n4. FLOPs for one training step (forward + backward), B=8, T=128")
torch.manual_seed(0)
model = GPT(cfg)
with FlopCounterMode(display=False) as fc:
    _, loss = model(x, y)
    loss.backward()
counts = {str(k): v for k, v in fc.get_flop_counts()["Global"].items()}
for k, v in sorted(counts.items()):
    print(f"  {k:<58}{v:>14,}")
tokens = B * cfg.block_size
print(f"  total measured {fc.get_total_flops():,}")
print(f"  calculator     {flops_per_token(s) * tokens:,.0f}")
print(f"  6 N D          {6 * n * tokens:,}")

print("\n5. wall time per training step, B=32, T=128 (median of 5)")
xb = torch.randint(0, 66, (32, cfg.block_size))
yb = torch.randint(0, 66, (32, cfg.block_size))
for mode in ("plain", "checkpointed"):
    torch.manual_seed(0)
    model = GPT(cfg)
    if mode == "checkpointed":
        model.blocks = torch.nn.ModuleList(Checkpointed(b) for b in model.blocks)
    times = []
    for _ in range(6):
        t0 = time.perf_counter()
        _, loss = model(xb, yb)
        loss.backward()
        times.append(time.perf_counter() - t0)
    print(f"  {mode:<13}{sorted(times[1:])[2] * 1e3:7.1f} ms")
```

```text
1. parameter counts
  L= 4 d= 96 V=   66: model     466,176  calculator     466,176
  L= 2 d=256 V= 1000: model   1,852,416  calculator   1,852,416
  L=12 d=768 V=50257: model 124,439,808  calculator 124,439,808

2. model state after one AdamW step (fp32), bytes
  weights 1,864,704  grads 1,864,704  Adam state 3,729,616  total 7,459,024
  calculator: 7,458,816  (the difference is Adam's per-tensor step counters: 208 bytes)

3. activation bytes saved for backward, B=8, T=128
  fp32           measured  26,379,268  calculator  26,379,268  loss 4.224212
  bf16 autocast  measured  16,070,020  calculator  16,070,020  loss 4.224245
  checkpointed   measured   2,655,236  calculator   2,655,236  loss 4.224212
  max |grad difference| fp32 vs checkpointed: 0.0e+00

4. FLOPs for one training step (forward + backward), B=8, T=128
  aten._scaled_dot_product_flash_attention_for_cpu             201,326,592
  aten._scaled_dot_product_flash_attention_for_cpu_backward    503,316,480
  aten.addmm                                                   905,969,664
  aten.mm                                                    1,850,867,712
  total measured 3,461,480,448
  calculator     3,360,817,152
  6 N D          2,864,185,344

5. wall time per training step, B=32, T=128 (median of 5)
  plain           66.7 ms
  checkpointed    84.0 ms
```

The 208-byte state gap is Adam's step counters, one 4-byte tensor for each of the model's 52 parameter tensors. Other runs of part 5 gave 58.2/72.5, 62.1/74.3, 60.8/78.4 and 58.3/76.6 ms.

### Script 3: numeric formats, `precision.py`

```python
"""Numeric formats and why mixed precision is built the way it is.

1. Range and precision of fp32, fp16, bf16 and the two fp8 formats.
2. Small gradients underflow in fp16; loss scaling rescues them.
3. Small updates vanish when weights are stored in 16 bits.
4. Long sums need a wide accumulator.
5. A late-training regime (small learning rate) with Adam, three ways:
   fp32; bf16 compute with fp32 master weights; everything in bf16.
6. Is bf16 faster? Only on hardware that supports it.
"""
import time

import torch

torch.manual_seed(0)
torch.set_num_threads(4)

print("1. formats")
print(f"  {'dtype':<16}{'bits':>5}{'eps':>11}{'max':>11}{'min normal':>12}{'min subnormal':>15}")
for dt in (torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
    fi = torch.finfo(dt)
    mant = round(-torch.log2(torch.tensor(fi.eps)).item())   # mantissa bits
    sub = fi.smallest_normal * 2.0 ** -mant                   # smallest subnormal
    print(f"  {str(dt)[6:]:<16}{fi.bits:>5}{fi.eps:>11.2e}{fi.max:>11.2e}"
          f"{fi.smallest_normal:>12.2e}{sub:>15.2e}")

print("\n2. gradient underflow: 1M values log-uniform in [1e-12, 1e-2]")
g = 10 ** (torch.rand(1_000_000, dtype=torch.float64) * -10 - 2)
for name, conv in [("fp16", lambda t: t.half()),
                   ("fp16, loss scale 2^16", lambda t: (t * 2**16).half().double() / 2**16),
                   ("bf16", lambda t: t.bfloat16())]:
    q = conv(g).double()
    zero = (q == 0).float().mean().item()
    rel = ((q - g).abs() / g)[q != 0].median().item()
    print(f"  {name:<24} flushed to zero {zero:6.1%}   median relative error of the rest {rel:.1e}")
big = torch.tensor([1e-1, 1.0, 10.0], dtype=torch.float64)
print(f"  but scaled by 2^16, gradients of 1e-1, 1, 10 become {(big * 2**16).half().tolist()}"
      " in fp16 (overflow)")

print("\n3. adding an update of 1e-4 to a weight of 1.0, 1000 times")
for dt in (torch.float32, torch.bfloat16, torch.float16):
    w = torch.tensor(1.0, dtype=dt)
    for _ in range(1000):
        w = w + torch.tensor(1e-4, dtype=dt)
    print(f"  stored in {str(dt)[6:]:<9} final weight {w.item():.6f}   (exact: 1.100000)")

print("\n4. sum of 4096 products x_i*y_i, x, y ~ N(0,1)")
x, y = torch.randn(4096, dtype=torch.float64), torch.randn(4096, dtype=torch.float64)
exact = (x * y).sum().item()
xb, yb = x.bfloat16(), y.bfloat16()
acc = torch.tensor(0.0, dtype=torch.bfloat16)
for a, b in zip(xb, yb):        # every partial sum rounded to bf16
    acc = acc + a * b
wide = (xb.double() * yb.double()).sum().item()   # bf16 inputs, wide accumulator
print(f"  exact {exact:.4f}  bf16 inputs + fp64 accumulator {wide:.4f}"
      f"  bf16 accumulator {acc.item():.4f}")

print("\n5. Adam, lr 1e-4, on a regression whose weights are near 1 (2000 steps)")
X = torch.randn(512, 64)
w_true = 1 + 0.1 * torch.randn(64, 1)
Y = X @ w_true
w_start = w_true + 0.01 * torch.randn(64, 1)     # already close: the late phase of training


def run(mode, steps=2000, lr=1e-4):
    dt = torch.bfloat16 if mode == "pure bf16" else torch.float32
    w = w_start.clone().to(dt).requires_grad_()
    opt = torch.optim.Adam([w], lr=lr)            # its state has w's dtype
    for _ in range(steps):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=(mode != "fp32")):
            loss = ((X.to(dt) @ w - Y.to(dt)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return ((X @ w.detach().float() - Y) ** 2).mean().item()   # judged in fp32


print(f"  starting loss {((X @ w_start - Y) ** 2).mean().item():.2e}")
for mode in ("fp32", "bf16 compute, fp32 master", "pure bf16"):
    print(f"  {mode:<26} final loss {run(mode):.2e}")

print("\n6. 1024x1024 matmul on this CPU (median of 5)")
for dt in (torch.float32, torch.bfloat16):
    a, b = torch.randn(1024, 1024).to(dt), torch.randn(1024, 1024).to(dt)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        a @ b
        ts.append(time.perf_counter() - t0)
    ms = sorted(ts)[2] * 1e3
    print(f"  {str(dt)[6:]:<9}{ms:7.2f} ms   {2 * 1024**3 / (ms / 1e3) / 1e9:6.1f} GFLOP/s")
```

```text
1. formats
  dtype            bits        eps        max  min normal  min subnormal
  float32            32   1.19e-07   3.40e+38    1.18e-38       1.40e-45
  float16            16   9.77e-04   6.55e+04    6.10e-05       5.96e-08
  bfloat16           16   7.81e-03   3.39e+38    1.18e-38       9.18e-41
  float8_e4m3fn       8   1.25e-01   4.48e+02    1.56e-02       1.95e-03
  float8_e5m2         8   2.50e-01   5.73e+04    6.10e-05       1.53e-05

2. gradient underflow: 1M values log-uniform in [1e-12, 1e-2]
  fp16                     flushed to zero  44.7%   median relative error of the rest 6.3e-04
  fp16, loss scale 2^16    flushed to zero   0.0%   median relative error of the rest 2.3e-04
  bf16                     flushed to zero   0.0%   median relative error of the rest 1.4e-03
  but scaled by 2^16, gradients of 1e-1, 1, 10 become [6552.0, inf, inf] in fp16 (overflow)

3. adding an update of 1e-4 to a weight of 1.0, 1000 times
  stored in float32   final weight 1.100017   (exact: 1.100000)
  stored in bfloat16  final weight 1.000000   (exact: 1.100000)
  stored in float16   final weight 1.000000   (exact: 1.100000)

4. sum of 4096 products x_i*y_i, x, y ~ N(0,1)
  exact -52.2534  bf16 inputs + fp64 accumulator -52.2452  bf16 accumulator -51.0000

5. Adam, lr 1e-4, on a regression whose weights are near 1 (2000 steps)
  starting loss 6.81e-03
  fp32                       final loss 2.36e-11
  bf16 compute, fp32 master  final loss 2.08e-04
  pure bf16                  final loss 6.63e-03

6. 1024x1024 matmul on this CPU (median of 5)
  float32     4.28 ms    501.3 GFLOP/s
  bfloat16 1552.91 ms      1.4 GFLOP/s
```

### Script 4: data, tensor and pipeline parallelism, `parallelism.py`

Four processes talk over localhost port 29517; change `MASTER_PORT` if it is taken.

```python
"""Data, tensor and pipeline parallelism, small enough to run on one CPU.

1. Data parallelism with real processes (torch.distributed, gloo backend):
   4 workers each take a quarter of the batch, all-reduce their gradients,
   and must end up with exactly the gradient of the full batch.
2. Tensor parallelism (Megatron-style MLP split) in one process: a column
   split of the first matrix and a row split of the second need a single
   sum (all-reduce) of partial outputs.
3. Pipeline parallelism: simulate a GPipe schedule and measure the bubble.
"""
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

WORLD, D, HID, BATCH = 4, 64, 256, 32


def make_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(D, HID), torch.nn.GELU(), torch.nn.Linear(HID, 1))


def make_data():
    g = torch.Generator().manual_seed(1)
    return torch.randn(BATCH, D, generator=g), torch.randn(BATCH, 1, generator=g)


def worker(rank, queue):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29517")
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    torch.set_num_threads(1)
    model = make_model()                      # same seed: identical replicas
    x, y = make_data()
    shard = slice(rank * BATCH // WORLD, (rank + 1) * BATCH // WORLD)
    F.mse_loss(model(x[shard]), y[shard]).backward()
    flat = torch.cat([p.grad.flatten() for p in model.parameters()])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)   # every rank gets the sum
    flat /= WORLD                                  # mean of shard means = full-batch mean
    if rank == 0:
        queue.put(flat)
    dist.destroy_process_group()


def data_parallel():
    model = make_model()
    x, y = make_data()
    F.mse_loss(model(x), y).backward()
    ref = torch.cat([p.grad.flatten() for p in model.parameters()])
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, queue)) for r in range(WORLD)]
    for p in procs:
        p.start()
    got = queue.get()
    for p in procs:
        p.join()
    n = ref.numel()
    print(f"1. data parallel, {WORLD} processes, {n:,} gradient values")
    print(f"   max |all-reduced grad - full-batch grad| = {(got - ref).abs().max():.1e}")
    print(f"   ring all-reduce traffic per worker: 2(W-1)/W x {n * 4:,} bytes"
          f" = {2 * (WORLD - 1) / WORLD * n * 4:,.0f} bytes per step")


def tensor_parallel(t=4, tokens=8):
    torch.manual_seed(0)
    x = torch.randn(tokens, D)
    A, B = torch.randn(D, HID) / D**0.5, torch.randn(HID, D) / HID**0.5
    full = F.gelu(x @ A) @ B
    cols = A.chunk(t, dim=1)                  # device i holds a block of A's columns ...
    rows = B.chunk(t, dim=0)                  # ... and the matching block of B's rows
    partial = [F.gelu(x @ a) @ b for a, b in zip(cols, rows)]   # no communication
    out = sum(partial)                        # the one all-reduce in the forward pass
    print(f"\n2. tensor parallel MLP over {t} devices")
    print(f"   max |sum of partials - full| = {(out - full).abs().max():.1e}")
    # Splitting A by rows instead gives partial pre-activations that must be
    # summed BEFORE the GELU, an extra synchronization point:
    xs, As = x.chunk(t, dim=1), A.chunk(t, dim=0)
    wrong = sum(F.gelu(xi @ ai) for xi, ai in zip(xs, As))
    print(f"   row-split A, GELU applied per device: max error {(wrong - F.gelu(x @ A)).abs().max():.2f}")


def pipeline(p, m, show=False):
    """GPipe: all forwards, then all backwards; backward takes 2 time units."""
    busy = {}                                 # (stage, time) -> label
    t_free = [0] * p
    fwd_done = {}
    for j in range(m):                        # forward of micro-batch j through stages
        for s in range(p):
            start = max(t_free[s], fwd_done.get((s - 1, j), 0))
            busy.update({(s, start): f"F{j}"})
            t_free[s] = fwd_done[(s, j)] = start + 1
    bwd_done = {}
    for j in range(m):                        # backward runs from the last stage down
        for s in reversed(range(p)):
            ready = fwd_done[(p - 1, m - 1)] if s == p - 1 else bwd_done[(s + 1, j)]
            start = max(t_free[s], ready)
            busy.update({(s, start): f"B{j}", (s, start + 1): f"B{j}"})
            t_free[s] = bwd_done[(s, j)] = start + 2
    total = max(t_free)
    idle = 1 - (3 * m * p) / (total * p)
    if show:
        for s in range(p):
            print("   stage", s, " ".join(busy.get((s, t), "..").rjust(3) for t in range(total)))
    return total, idle


if __name__ == "__main__":
    data_parallel()
    tensor_parallel()
    print("\n3. pipeline (GPipe) schedule, p=4 stages, m=4 micro-batches; F=1 unit, B=2 units")
    pipeline(4, 4, show=True)
    for p_, m_ in [(4, 1), (4, 4), (4, 16), (8, 16), (8, 64)]:
        total, idle = pipeline(p_, m_)
        print(f"   p={p_} m={m_:>2}: time {total:>3} units, idle {idle:5.1%},"
              f" formula (p-1)/(m+p-1) = {(p_ - 1) / (m_ + p_ - 1):5.1%}")
```

```text
1. data parallel, 4 processes, 16,897 gradient values
   max |all-reduced grad - full-batch grad| = 1.6e-07
   ring all-reduce traffic per worker: 2(W-1)/W x 67,588 bytes = 101,382 bytes per step

2. tensor parallel MLP over 4 devices
   max |sum of partials - full| = 7.7e-07
   row-split A, GELU applied per device: max error 1.75

3. pipeline (GPipe) schedule, p=4 stages, m=4 micro-batches; F=1 unit, B=2 units
   stage 0  F0  F1  F2  F3  ..  ..  ..  ..  ..  ..  ..  ..  ..  B0  B0  B1  B1  B2  B2  B3  B3
   stage 1  ..  F0  F1  F2  F3  ..  ..  ..  ..  ..  ..  B0  B0  B1  B1  B2  B2  B3  B3  ..  ..
   stage 2  ..  ..  F0  F1  F2  F3  ..  ..  ..  B0  B0  B1  B1  B2  B2  B3  B3  ..  ..  ..  ..
   stage 3  ..  ..  ..  F0  F1  F2  F3  B0  B0  B1  B1  B2  B2  B3  B3  ..  ..  ..  ..  ..  ..
   p=4 m= 1: time  12 units, idle 75.0%, formula (p-1)/(m+p-1) = 75.0%
   p=4 m= 4: time  21 units, idle 42.9%, formula (p-1)/(m+p-1) = 42.9%
   p=4 m=16: time  57 units, idle 15.8%, formula (p-1)/(m+p-1) = 15.8%
   p=8 m=16: time  69 units, idle 30.4%, formula (p-1)/(m+p-1) = 30.4%
   p=8 m=64: time 213 units, idle  9.9%, formula (p-1)/(m+p-1) =  9.9%
```

### Script 5: distillation and pruning, `compress.py`

```python
"""Distillation and pruning on scikit-learn's digits (8x8 images, 10 classes).

1. Distillation: a small student learns from a big teacher's soft targets
   [Hinton et al., 2015], on the labeled set alone or on a larger transfer
   set of unlabeled images, compared with training on labels, over 5 seeds.
2. Magnitude pruning of the teacher, with and without fine-tuning.
3. Zeros in a dense matrix cost the same as any other number: dense vs sparse matmul.
Timings depend on the machine and its load.
"""
import time
import warnings

import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(4)
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32)
y = torch.tensor(y)
# 300 labeled images, 800 unlabeled ones (labels thrown away), 697 for testing.
Xl, Xrest, yl, yrest = train_test_split(X, y, train_size=300, random_state=0, stratify=y)
Xu, Xte, _, yte = train_test_split(Xrest, yrest, train_size=800, random_state=0, stratify=yrest)
Xall = torch.cat([Xl, Xu])
n_lab = len(Xl)


def mlp(width):
    return torch.nn.Sequential(torch.nn.Linear(64, width), torch.nn.ReLU(),
                               torch.nn.Linear(width, width), torch.nn.ReLU(),
                               torch.nn.Linear(width, 10))


def train(model, Xs, loss_fn, epochs=150, lr=3e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for idx in torch.randperm(len(Xs), generator=g).split(64):
            aug = Xs[idx] + 0.05 * torch.randn(len(idx), 64, generator=g)   # light noise
            opt.zero_grad()
            loss_fn(model(aug), idx).backward()
            opt.step()
    return model


@torch.no_grad()
def acc(model):
    return (model(Xte).argmax(1) == yte).float().mean().item()


def nparams(model):
    return sum((p != 0).sum().item() for p in model.parameters())


def label_loss(out, idx):          # indices < n_lab are labeled examples
    return F.cross_entropy(out, yl[idx])


torch.manual_seed(0)
teacher = train(mlp(512), Xl, label_loss)
with torch.no_grad():
    t_logits = teacher(Xall)       # the teacher labels everything, including Xu
print(f"teacher: {nparams(teacher):,} parameters, trained on {n_lab} labels,"
      f" test accuracy {acc(teacher):.3f}")


def kd_loss(T, alpha=0.9):
    def loss(out, idx):
        soft = F.kl_div(F.log_softmax(out / T, 1), F.softmax(t_logits[idx] / T, 1),
                        reduction="batchmean") * T * T      # T^2 keeps gradient scale
        lab = idx < n_lab                                   # hard loss only where labels exist
        hard = F.cross_entropy(out[lab], yl[idx[lab]]) if lab.any() else 0.0
        return alpha * soft + (1 - alpha) * hard
    return loss


def pseudo_label_loss(out, idx):   # teacher's argmax as if it were a label
    return F.cross_entropy(out, t_logits[idx].argmax(1))


print("\n1. student with 16 hidden units per layer, 5 seeds: mean (min-max)")
for name, Xs, fn in [("labels only (300)", Xl, label_loss),
                     ("distilled, T=4, same 300", Xl, kd_loss(4.0)),
                     ("hard pseudo-labels, 1100", Xall, pseudo_label_loss),
                     ("distilled, T=1, 1100", Xall, kd_loss(1.0)),
                     ("distilled, T=4, 1100", Xall, kd_loss(4.0))]:
    accs = []
    for seed in range(5):
        torch.manual_seed(seed)
        accs.append(acc(train(mlp(16), Xs, fn, seed=seed)))
    t = torch.tensor(accs)
    print(f"   {name:<26} {t.mean():.3f}  ({t.min():.3f}-{t.max():.3f})")
print(f"   student size: {nparams(mlp(16)):,} parameters")

print("\n2. global magnitude pruning of the teacher's weight matrices")
state = {k: v.clone() for k, v in teacher.state_dict().items()}
weights = [m.weight for m in teacher if isinstance(m, torch.nn.Linear)]
for frac in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99):
    teacher.load_state_dict(state)
    allw = torch.cat([w.detach().abs().flatten() for w in weights])
    cut = allw.kthvalue(int(frac * allw.numel())).values
    masks = [(w.detach().abs() > cut).float() for w in weights]
    with torch.no_grad():
        for w, mk in zip(weights, masks):
            w.mul_(mk)
    before = acc(teacher)

    # fine-tune 20 epochs, re-applying the mask after every step via a hook
    hooks = [w.register_hook(lambda g, mk=mk: g * mk) for w, mk in zip(weights, masks)]
    train(teacher, Xl, label_loss, epochs=20, lr=1e-3)
    for h in hooks:
        h.remove()
    print(f"   {frac:4.0%} of weights removed: accuracy {before:.3f} -> {acc(teacher):.3f}"
          f" after fine-tuning   ({nparams(teacher):,} nonzero parameters)")

print("\n3. 2048 x 2048 weight times 2048 x 64 activations (median of 5)")


def timeit(f):
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[2] * 1e3


torch.manual_seed(0)
W, A = torch.randn(2048, 2048), torch.randn(2048, 64)
print(f"   dense, no zeros            {timeit(lambda: W @ A):7.2f} ms")
for frac in (0.5, 0.9, 0.99):
    Wz = W * (torch.rand_like(W) > frac)
    Ws = Wz.to_sparse_csr()
    print(f"   {frac:4.0%} zeros: dense {timeit(lambda: Wz @ A):6.2f} ms   "
          f"sparse CSR {timeit(lambda: Ws @ A):6.2f} ms")
```

```text
teacher: 301,066 parameters, trained on 300 labels, test accuracy 0.940

1. student with 16 hidden units per layer, 5 seeds: mean (min-max)
   labels only (300)          0.908  (0.900-0.915)
   distilled, T=4, same 300   0.919  (0.895-0.930)
   hard pseudo-labels, 1100   0.930  (0.927-0.933)
   distilled, T=1, 1100       0.935  (0.931-0.940)
   distilled, T=4, 1100       0.936  (0.934-0.940)
   student size: 1,482 parameters

2. global magnitude pruning of the teacher's weight matrices
    50% of weights removed: accuracy 0.940 -> 0.937 after fine-tuning   (151,050 nonzero parameters)
    80% of weights removed: accuracy 0.925 -> 0.937 after fine-tuning   (61,041 nonzero parameters)
    90% of weights removed: accuracy 0.763 -> 0.940 after fine-tuning   (31,038 nonzero parameters)
    95% of weights removed: accuracy 0.125 -> 0.902 after fine-tuning   (16,036 nonzero parameters)
    98% of weights removed: accuracy 0.100 -> 0.257 after fine-tuning   (7,035 nonzero parameters)
    99% of weights removed: accuracy 0.100 -> 0.205 after fine-tuning   (4,035 nonzero parameters)

3. 2048 x 2048 weight times 2048 x 64 activations (median of 5)
   dense, no zeros               1.15 ms
    50% zeros: dense   1.13 ms   sparse CSR   1.85 ms
    90% zeros: dense   1.12 ms   sparse CSR   0.30 ms
    99% zeros: dense   1.12 ms   sparse CSR   0.04 ms
```

## Common mistakes

- **Counting only weights.** "7B in bf16 is 14 GB" is inference. Training with Adam needs about 16 bytes per parameter before activations.
- **Expecting mixed precision to halve memory.** It leaves the model state alone and, under autocast, saved 39% of activations in script 2.
- **Forgetting the logits**, $B \cdot T \cdot V$ fp32 numbers. Computing the loss in chunks avoids holding them all.
- **Using $6ND$ at long context or with recomputation** without adding $12LTd$ or the extra forward.
- **Quoting hardware FLOP utilization as MFU.** Recomputation raises the former without producing tokens.
- **Benchmarking low precision on hardware without support for it.**
- **Too few micro-batches**: with $m = p$, nearly half a pipeline is idle.
- **Counting FLOPs saved by unstructured sparsity as speed.** Time it.
- **Comparing a compressed model only with its original**, not with a small dense model given the same budget.
- **Adding up serving papers' speedups.** Their baselines lack each other's techniques.

## Exercises

1. [warm-up] Model-state bytes for 13B parameters with fp32 Adam, with mixed precision, and with ZeRO stage 3 over 16 devices?
2. [warm-up] bf16 activations for one 2,048-token sequence through 24 layers of width 2,048 with fused attention, ignoring logits?
3. [warm-up] For $N \approx 12Ld^2$ and $d = 4{,}096$, at what $T$ is the attention term 25% of $6N$?
4. [core] Run `validate_calculator.py` with $B = 16$ and $T = 64$. Predict each number first.
5. [core] Write a SwiGLU variant of the L2-05 MLP and compare its saved bytes with `activation_bytes` for a `Shape` with `gated_mlp=True`. Fix the formula if needed.
6. [core] Repeat part 5 of `precision.py` with learning rates $10^{-3}$ and $10^{-2}$. When does pure bf16 stop stalling?
7. [core] Add the 1F1B schedule to `parallelism.py` and show that the bubble is unchanged while at most $p$ micro-batches are in flight per stage.
8. [core] In `compress.py`, prune hidden units with the smallest weight norms, shrink the matrices, fine-tune, and time the smaller dense model. Compare with magnitude pruning at equal parameter count.
9. [stretch] Estimate the per-step all-reduce time for 70B parameters over 512 devices, and how much can hide behind a 2 s backward pass if gradients become ready uniformly.
10. [stretch] Write a toy server loop with a cost of 20 ms plus 0.5 ms per running sequence per decode step, and compare static batching with continuous batching, and contiguous KV reservation with 16-token pages, at several arrival rates.

### Hints and answers

1. $13 \times 10^9 \times 16 = 208$ GB for both; 13 GB per device with stage 3.
2. $32 \times 2{,}048 \times 24 \times 2{,}048 \approx 3.2$ GB.
3. $T/(6d) = 0.25$ gives $T = 1.5d = 6{,}144$.
4. The position table loses $64 \times 96 = 6{,}144$ parameters (16 bytes of state each). Activations depend on $B \cdot T = 1{,}024$, unchanged except a 512-byte smaller positions tensor. Attention FLOPs halve; matmul FLOPs do not change.
5. SiLU saves its input and the product saves both inputs: about $3 d_{ff}$ per token. If gate and up projections are fused, check for shared storage.
6. Adam's step is about the learning rate; stalling ends roughly when it exceeds half the bf16 spacing near the weights ($2^{-8} \approx 0.004$ near 1). Expect movement at $10^{-2}$; test $10^{-3}$.
7. Track forwards minus backwards per stage over time; it never exceeds $p - s$ for stage $s$.
8. Structured pruning speeds up any hardware; on this toy expect lower accuracy than unstructured at equal size.
9. About $2 \times 140 \times 511/512 \approx 280$ GB per device; gradients ready late in backward cannot overlap fully.
10. With contiguous reservation, concurrency is capped at memory divided by maximum length, whatever the real lengths; paging lifts the cap until memory is truly full, then needs preemption.

## Check your understanding

**1. Why does mixed-precision Adam still need about 16 bytes per parameter?** The passes use 16-bit weights and gradients (4 bytes), but the update keeps fp32 master weights, momentum and variance (12 bytes), because small updates are lost in 16 bits. The savings are in activations and, on suitable hardware, speed.

**2. Where does the 6 in $6ND$ come from, and when is it wrong?** Each matmul weight is used once per token forward at 2 FLOPs ($2N$), and backward needs gradients for both inputs and weights ($4N$). It omits attention ($12LTd$ per token), recomputation and elementwise work, and for mixture-of-experts models $N$ must be the active parameters.

**3. What does checkpointing buy and cost?** Saving only block inputs cuts activations from about $32d$ bytes per token per layer (16-bit) to $2d$, plus one block during recomputation, for about one extra forward pass; script 2 measured a 20 to 31% slower step and identical gradients.

**4. Why keep tensor parallelism inside a server and spread data parallelism across the cluster?** Tensor parallelism all-reduces activations twice per layer in each direction, on the critical path; data parallelism communicates gradients once per step and can overlap with backward.

**5. A paper claims 90% sparsity, no accuracy loss, 10 times faster. What do you ask?** Was speed measured on hardware or computed from FLOPs? Is the sparsity a pattern the hardware accelerates? Is there a small dense baseline with the same budget? How much fine-tuning, on how many tasks and seeds [Blalock et al., 2020]?

## Further reading

- **sankarsh98 (GitHub), "AI Systems", Track B: Inference efficiency** (web course, free). <https://sankarsh98.github.io/ai-systems/inference/prefill-and-decode>
  *Read for:* prefill and decode, the KV cache, quantization and speculative decoding, with runnable NumPy code. *Level:* intro.
- **Hugging Face nanotron team, "The Ultra-Scale Playbook: Training LLMs on GPU Clusters"** (2025, Hugging Face). <https://huggingface.co/spaces/nanotron/ultrascale-playbook>
  *Read for:* memory formulas, every kind of parallelism and ZeRO, explained with measurements from thousands of runs. *Level:* intro to intermediate.
- **PyTorch developers, "Automatic Mixed Precision package - torch.amp"** (PyTorch 2.15 documentation). <https://docs.pytorch.org/docs/2.15/amp.html>
  *Read for:* which operations autocast runs in 16 bits or fp32, and when `GradScaler` is needed. *Level:* intro.
- **Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He, "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"** (2019, arXiv:1910.02054). <https://arxiv.org/abs/1910.02054>
  *Read for:* the $16\Psi$ accounting and the memory and communication of each sharding stage. *Level:* intermediate.
- **Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, Bryan Catanzaro, "Reducing Activation Recomputation in Large Transformer Models"** (2022, arXiv:2205.05198). <https://arxiv.org/abs/2205.05198>
  *Read for:* the activation derivation of section 2.2, sequence parallelism and selective recomputation. *Level:* intermediate to advanced.
- **Paulius Micikevicius, Sharan Narang, Jonah Alben, et al. (11 authors), "Mixed Precision Training"** (2018, ICLR; arXiv:1710.03740). <https://arxiv.org/abs/1710.03740>
  *Read for:* master weights, loss scaling and fp32 accumulation. *Level:* intro to intermediate.
- **Paulius Micikevicius, Dusan Stosic, Neil Burgess, et al. (15 authors), "FP8 Formats for Deep Learning"** (2022, arXiv:2209.05433). <https://arxiv.org/abs/2209.05433>
  *Read for:* the E4M3 and E5M2 encodings and evidence that fp8 training can match 16-bit. *Level:* intermediate.
- **Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro, "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"** (2019, arXiv:1909.08053). <https://arxiv.org/abs/1909.08053>
  *Read for:* tensor parallelism's column and row splits, in a few lines of PyTorch. *Level:* intermediate.
- **Deepak Narayanan, Mohammad Shoeybi, Jared Casper, et al. (12 authors), "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"** (2021, SC21; arXiv:2104.04473). <https://arxiv.org/abs/2104.04473>
  *Read for:* combining tensor, pipeline and data parallelism, pipeline schedules and their bubbles, and the FLOP formula. *Level:* advanced.
- **Geoffrey Hinton, Oriol Vinyals, Jeff Dean, "Distilling the Knowledge in a Neural Network"** (2015, NIPS 2014 Deep Learning Workshop; arXiv:1503.02531). <https://arxiv.org/abs/1503.02531>
  *Read for:* temperature, soft targets and the $T^2$ factor. *Level:* intro.
- **Davis Blalock, Jose Javier Gonzalez Ortiz, Jonathan Frankle, John Guttag, "What is the State of Neural Network Pruning?"** (2020, MLSys; arXiv:2003.03033). <https://arxiv.org/abs/2003.03033>
  *Read for:* a meta-analysis of 81 pruning papers and a checklist for fair comparisons. *Level:* intermediate.
- **Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al. (9 authors), "Efficient Memory Management for Large Language Model Serving with PagedAttention"** (2023, SOSP; arXiv:2309.06180). <https://arxiv.org/abs/2309.06180>
  *Read for:* why KV memory is wasted, and paging, sharing and preemption in vLLM. *Level:* intermediate.

Also cited, each verified on arXiv or the venue's site: Chen et al., "Training Deep Nets with Sublinear Memory Cost" (2016, arXiv:1604.06174); Kaplan et al., "Scaling Laws for Neural Language Models" (2020, arXiv:2001.08361); Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways" (2022, arXiv:2204.02311); Kalamkar et al., "A Study of BFLOAT16 for Deep Learning Training" (2019, arXiv:1905.12322); Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" (OSDI 2022, <https://www.usenix.org/conference/osdi22/presentation/yu>).
