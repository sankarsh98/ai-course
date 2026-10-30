# Training Deep Networks

> **Level 2 · Module 2** · Prerequisites: L0-05, L1-01, L2-01 · Time: 6 to 8 hours · Code: `code/L2-02-training-deep-networks/`

## What you will be able to do

- Write a complete PyTorch training loop with validation, checkpointing and early stopping.
- Derive why the initial weight scale must depend on fan-in and activation, and detect a network whose signal dies at initialization.
- Use batch norm and layer norm correctly, including train and eval modes, and summarize the debate about why batch norm helps.
- Apply dropout, weight decay and early stopping, and predict when each matters.
- Choose a learning rate and schedule, and debug a network that will not train with a fixed sequence of checks.
- Make a run reproducible and measure seed-to-seed variation.

## 1. From NumPy to PyTorch

L2-01 wrote an MLP's forward and backward passes by hand. PyTorch gives you the same arrays plus automatic differentiation, so you write only the forward pass.

A **tensor** follows NumPy's rules for shape, dtype and broadcasting (L0-01), with two early surprises: the default float dtype is float32, and `torch.from_numpy` shares memory with the NumPy array (script 1).

**Autograd** is L0-03's reverse-mode engine at the level of tensors. Operations on a tensor created with `requires_grad=True` (a **leaf**) record a graph, which the documentation says is recreated from scratch at every iteration [PyTorch, Autograd mechanics], and `loss.backward()` fills each leaf's `.grad`. Three behaviors shape every training loop:

- **Gradients accumulate.** A second `backward()` adds to `.grad` (2, then 4, in script 1), so you clear them every step with `optimizer.zero_grad()`.
- **A graph is used once.** `backward()` frees its saved tensors; a second call raises an error.
- **`torch.no_grad()` stops recording** and `.detach()` cuts a tensor out of the graph. Both are separate from `model.eval()`, which the documentation calls "completely orthogonal" to gradient mode: it changes how dropout and batch norm behave, nothing else [PyTorch, Autograd mechanics].

Given the same float64 weights and batch as L2-01's NumPy MLP, PyTorch matches its loss to 12 digits and its gradients to $6 \times 10^{-17}$.

## 2. Modules, data and the training loop

An **`nn.Module`** is a container for parameters and a `forward` method. Assigning an `nn.Linear` as an attribute registers its parameters, so `model.parameters()` finds them and `model.state_dict()` can save them. A **`Dataset`** needs only `__len__` and `__getitem__`; a **`DataLoader`** batches and shuffles it. The loop itself has four lines that never change:

```python
opt.zero_grad()                  # 1. clear old gradients
loss = loss_fn(model(xb), yb)    # 2. forward
loss.backward()                  # 3. backward
opt.step()                       # 4. update
```

Around it go the parts that make results trustworthy (script 3): a validation set split off the training data, evaluation under `model.eval()` and `torch.no_grad()`, a copy of the best weights by validation loss, a stopping rule, and one final evaluation on the test set. `nn.CrossEntropyLoss` expects raw logits and integer class labels, and is equivalent to log-softmax followed by negative log-likelihood [PyTorch, CrossEntropyLoss]; do not put a softmax at the end of the model.

On digits (the same split as L1-02 and L2-01, now with 1,005 training, 252 validation and 540 test images), the one-hidden-layer MLP stops at epoch 85, restores epoch 75's weights, and scores 0.972 on test, in line with softmax regression (0.972) and L2-01's NumPy MLP (0.970 to 0.975). The rest of this module uses the shorter `fit` helper in `common.py`, which is the same loop.

## 3. Initialization

A deep network multiplies its input by many weight matrices. If each layer scales the signal by $a$, the output scales by $a^L$, which vanishes for $a < 1$ and explodes for $a > 1$; gradients flowing back suffer the same fate. Initialization chooses the weight scale so that $a \approx 1$.

**The variance argument.** For one unit $z = \sum_{i=1}^{n} w_i h_i$ with fan-in $n$ and independent weights of mean 0 and variance $\sigma^2$,

$$\mathrm{Var}(z) = n\,\sigma^2\,\mathbb{E}[h^2].$$

For tanh near its linear region, $\mathbb{E}[h^2] \approx \mathrm{Var}(z_{\text{prev}})$, so keeping the variance constant needs $\sigma^2 = 1/n$. [Glorot & Bengio, 2010] balanced the same condition for the backward pass, which involves fan-out, and proposed $\sigma^2 = 2/(n_{\text{in}} + n_{\text{out}})$ ("Xavier" initialization). ReLU zeroes half of a symmetric input, so $\mathbb{E}[h^2] = \mathrm{Var}(z_{\text{prev}})/2$ and the condition becomes $\sigma^2 = 2/n$, the initialization of [He et al., 2015] ("He" or "Kaiming").

**PyTorch's default is not He.** `nn.Linear` draws weights and biases from $U(-\sqrt{k}, \sqrt{k})$ with $k = 1/\text{fan-in}$ [PyTorch, nn.Linear], a weight variance of $1/(3n)$ (script 4 measures standard deviation 0.0362 for $n = 256$; predicted 0.0361). In a ReLU stack the argument predicts the signal shrinks by $\sqrt{1/6} \approx 0.41$ per layer.

Script 4 measures how much each unit's output varies **across inputs** in a 50-layer network of width 256, which is the information later layers can use. With He and ReLU it is still 0.15 at layer 50; with the default, $10^{-9}$; with $N(0, 0.01^2)$, exactly 0. I first measured the overall standard deviation, and the default looked healthy at 0.02. That was the randomly initialized biases: every input produced the same output. Measure the quantity you care about.

Training a 20-hidden-layer ReLU MLP with SGD and momentum makes the point concrete. With $N(0, 0.01^2)$ or the default, the loss sits at $\ln 10 = 2.303$ for 20 epochs (chance accuracy). Xavier reaches 1.795; He reaches 0.017 and validation accuracy 0.964.

> **Note:** The default is fine for shallow networks, and most deep models add normalization or residual connections (L2-03) that reduce sensitivity to initialization. For a deep plain stack, set the initialization yourself with `torch.nn.init`.

## 4. Normalization

The alternative is to **normalize activations inside the network**. **Batch normalization** [Ioffe & Szegedy, 2015] standardizes each feature over the minibatch and applies a learned scale $\gamma$ and shift $\beta$:

$$\hat z = \frac{z - \mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}}, \qquad \text{output} = \gamma \hat z + \beta.$$

At test time there may be no batch, so the layer keeps running averages during training and uses them in eval mode. PyTorch updates them as $(1 - m)\,\text{running} + m \cdot \text{batch statistic}$ with $m = 0.1$, and warns that this "momentum" is unrelated to optimizer momentum [PyTorch, BatchNorm1d]. **Layer normalization** [Ba et al., 2016] standardizes over the features of each example instead, so it computes the same thing in training and testing and ignores the batch. It is the usual choice in sequence models and transformers (L2-04, L2-05).

Added to the 20-layer network that failed with default initialization (script 5), batch norm reaches validation accuracy 0.976 at lr 0.01 and 0.960 at lr 0.001. Layer norm is unstable at lr 0.01 (loss 2.324 at epoch 20) but trains at lr 0.001 (0.940). Without normalization the network stays at chance at both rates. Normalization does not remove the need to tune the learning rate.

**The train/eval trap.** In train mode, batch norm normalizes each test batch with its own statistics. On the same trained model, test accuracy is 0.969 in eval mode, 0.956 in train mode with random batches of 256, 0.813 with batches of 16, and 0.137 with batches of 16 sorted by label, where a batch of identical digits is normalized into nonsense. A batch of one raises an error. Layer norm gives the same output for an example alone or in a batch, up to $8 \times 10^{-7}$ of float32 round-off.

**Why does batch norm help?** The original paper attributes it to reducing "internal covariate shift", the change in each layer's input distribution during training [Ioffe & Szegedy, 2015]. [Santurkar et al., 2018] argued that this contributes little and that batch norm instead smooths the loss landscape, making gradients more predictable and larger steps safe. The practical benefits are well documented; the mechanism is still discussed.

## 5. Regularization

Three tools cover most practice.

- **Weight decay** shrinks weights toward zero every step. For SGD it equals an L2 penalty; for Adam it does not, and AdamW applies the decay separately (Loshchilov and Hutter, "Decoupled Weight Decay Regularization", ICLR 2019, arXiv:1711.05101; L0-05 derives this).
- **Dropout** [Srivastava et al., 2014] zeroes each unit with probability $p$ during training, so no unit can rely on specific others. PyTorch scales survivors by $1/(1-p)$ in training and computes the identity in eval mode [PyTorch, Dropout], as script 6's first lines show.
- **Early stopping** keeps the checkpoint with the best validation loss.

Script 6 trains a 64-256-256-10 MLP (85,002 parameters) on 300 digits. With clean labels nothing matters much: every setting scores about 0.952 on test. Then 30% of training and validation labels are replaced at random, with test labels kept clean. The unregularized network memorizes the noise (training loss 0.004 on labels that are 30% wrong) and its final test accuracy falls to 0.769. [Zhang et al., 2017] showed the extreme case: standard image networks fit completely random labels, and explicit regularization did not prevent it. Weight decay 1.0 gives 0.778, dropout 0.808, both 0.822. **Early stopping helps most**: the best-validation checkpoint (epochs 4 to 10) scores 0.869 to 0.884 in every setting. Networks tend to fit the consistent signal before the noise, and early stopping catches the moment between, here using a validation set that is itself noisy.

Two cautions. With 3 seeds on one small dataset, differences under about 0.01 are noise (section 8). And on clean data, early stopping on validation loss chose epochs 13 to 20 for the unregularized network, yet training to epoch 80 cost nothing in test accuracy (0.952 against 0.951): validation loss rises as the network grows overconfident while accuracy holds. Choose the stopping criterion that matches what you report.

## 6. Learning rates and schedules

The learning rate is often the most consequential hyperparameter, because curvature sets its scale (L0-05). **The range test** [Smith, 2017] estimates reasonable bounds by increasing the rate during a short run. Script 7 raises it from $10^{-4}$ to 10 over 120 steps: the loss barely moves below $10^{-2}$, falls steeply between 0.03 and 0.3, and explodes by 1.0. In full 15-epoch runs, 0.1 was the best constant rate and 0.3 failed (validation accuracy 0.151), so the safe long-run rate was about a tenth of the blow-up point. Use the test for the order of magnitude.

**Schedules** change the rate during training: **step decay**; **cosine decay** to zero, popularized by Loshchilov and Hutter's SGDR (ICLR 2017, arXiv:1608.03983); and **linear warmup** over the first steps, which Goyal et al. combined with a linear batch-size scaling rule to train ImageNet models with minibatches of 8,192 ("Accurate, Large Minibatch SGD", 2017, arXiv:1706.02677). The Google tuning playbook prefers linear or cosine decay and calls the best schedule family an open problem [Godbole et al., 2023].

At a good base rate (0.03 or 0.1), script 7's four schedules end within 0.005 validation accuracy of each other, inside seed noise; decaying from 0.03 even leaves a higher training loss (0.04 against 0.013). At 0.3, decay rescues part of the damage (cosine 0.718 against constant 0.151) and warmup adds nothing. Schedules matter most for long runs, large batches and large models; on a 15-epoch digits run, the base rate dominates.

## 7. Debugging a network that will not train

Many failed runs come from bugs that raise no error. [Karpathy, 2019] and the CS231n notes cited in L2-01 recommend the same discipline: look at the data, start simple, verify each stage. A sequence that catches most problems:

1. **Check the initial loss**: near $\ln K$ for $K$ classes. Script 8 measures 2.325 against 2.303.
2. **Overfit one batch.** A correct pipeline drives the loss on 32 examples to nearly zero [Karpathy, 2019]. Script 8 plants four bugs, each with its own signature:
   - *Softmax before `CrossEntropyLoss`*: the loss plateaus at exactly $\ln(1 + 9/e) = 1.4612$, because log-softmax applied to probabilities in $[0, 1]$ cannot do better.
   - *No `zero_grad()`*: the effective step grows each iteration. The loss falls faster than the correct run at first (0.0015 at step 10), then explodes to $10^{17}$.
   - *Learning rate 30 times too high*: explodes within 10 steps, then NaN.
   - *Labels shuffled relative to inputs*: **passes**, because 9,610 parameters memorize 32 arbitrary labels. Held-out data exposes it: validation accuracy 0.111 against 0.972.
3. **Check shapes where broadcasting hides errors.** `mse_loss` on shapes (4, 1) and (4,) broadcasts to 4 by 4 and returns 2.510 instead of 0.010, with only a warning.
4. **Localize NaNs** with `torch.autograd.detect_anomaly()`. In script 8 the sqrt at 0 has an infinite derivative and the next backward step multiplies it by 0; anomaly mode names `PowBackward0`, the step after the sqrt. It points near the cause, not always at it.

The one-batch test checks model and optimizer code, not data, which is why Karpathy's first step is looking at the data.

## 8. Reproducibility

Seeding makes a run repeatable on the same machine and software. `torch.manual_seed` seeds the generator used for initialization, dropout and (by default) shuffling. The PyTorch documentation is explicit that completely reproducible results are not guaranteed across PyTorch releases, individual commits or platforms, or between CPU and GPU, and it describes `torch.use_deterministic_algorithms` and how to seed `DataLoader` workers [PyTorch, Reproducibility].

Script 9 tests four claims on this machine. Two runs with the same seed give bitwise-identical weights. One extra `torch.rand(1)` after initialization changes every later dropout mask, and the final weights differ. Passing the `DataLoader` its own seeded generator keeps the shuffle order fixed when other code draws random numbers; relying on the global generator does not. And changing the thread count from 4 to 1 changes the weights by $1.8 \times 10^{-7}$ after 5 epochs, because parallel reductions add numbers in a different order. That is tiny here, but in long runs such differences can grow.

Bitwise reproducibility is for debugging. For conclusions you need **seed variation**: the same configuration trained with 10 seeds gives validation accuracy from 0.956 to 0.972 (standard deviation 0.005). A difference of 0.01 between two single runs is within that range. Report several seeds and the spread, as L1-01 recommends.

## Hands-on

Run from `code/L2-02-training-deep-networks/`, for example `../../.venv/bin/python training_loop.py`. All scripts use PyTorch 2.15.0 (CPU) and scikit-learn's bundled digits; `tensors_autograd.py` imports `mlp_numpy.py` from L2-01. Each finished in under 40 seconds on a shared Linux machine that was busy with other jobs, so timings printed below are rough. With 4 threads set in `common.py`, my numbers should reproduce on the same software; other platforms or versions may differ in later digits, and occasionally in a seed-sensitive result.

### Script 1: tensors and autograd

```python
"""PyTorch tensors and autograd, checked against the hand-written NumPy MLP from L2-01."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "L2-01-neural-networks-and-backprop"))
import mlp_numpy  # noqa: E402

print("torch", torch.__version__)

# 1. Tensors: NumPy-like, float32 by default, and from_numpy shares memory.
a = np.arange(6.0).reshape(2, 3)
t = torch.from_numpy(a)                 # float64, because a is float64; no copy
t[0, 0] = 100.0
print("a[0, 0] after editing the tensor:", a[0, 0], "| default dtype:", torch.ones(1).dtype)
print("broadcast (2,3) + (3,) ->", tuple((t + torch.ones(3, dtype=t.dtype)).shape))

# 2. Autograd: leaves with requires_grad record a graph; backward() fills .grad.
w = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
loss = (w ** 2).sum()                   # d/dw = 2w
loss.backward()
print("\ngrad after one backward:", w.grad.tolist())
(w ** 2).sum().backward()               # a second backward ADDS to .grad
print("grad after a second backward (accumulated):", w.grad.tolist())
w.grad = None                           # what optimizer.zero_grad() does by default
with torch.no_grad():                   # no graph: for evaluation and manual updates
    print("inside no_grad, result requires grad?", (w * 2).requires_grad)
print("detach() cuts the graph:", w.detach().requires_grad)

loss = (w ** 3).sum()
loss.backward()
try:
    loss.backward()                     # the graph's saved tensors were freed
except RuntimeError as err:
    print("second backward on the same graph ->", str(err).split(".")[0])

# 3. Same weights, same batch: torch autograd vs the L2-01 manual backward pass.
rng = np.random.default_rng(0)
params = mlp_numpy.init_params([64, 32, 32, 10], rng)
X, y = rng.normal(size=(16, 64)), rng.integers(0, 10, size=16)
loss_np, grads_np = mlp_numpy.loss_and_grads(params, X, y, "tanh")

tparams = [(torch.tensor(W, requires_grad=True), torch.tensor(b, requires_grad=True)) for W, b in params]
h = torch.tensor(X)
for l, (W, b) in enumerate(tparams):
    h = h @ W + b
    if l < len(tparams) - 1:
        h = torch.tanh(h)
loss_t = torch.nn.functional.cross_entropy(h, torch.tensor(y))
loss_t.backward()
diff = max(np.abs(W.grad.numpy() - dW).max() for (W, _), (dW, _) in zip(tparams, grads_np))
print(f"\nloss numpy {loss_np:.12f} vs torch {loss_t.item():.12f}; max |grad diff| {diff:.1e}")
```

```text
torch 2.15.0+cpu
a[0, 0] after editing the tensor: 100.0 | default dtype: torch.float32
broadcast (2,3) + (3,) -> (2, 3)

grad after one backward: [2.0, -4.0, 6.0]
grad after a second backward (accumulated): [4.0, -8.0, 12.0]
inside no_grad, result requires grad? False
detach() cuts the graph: False
second backward on the same graph -> Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed)

loss numpy 2.684423467916 vs torch 2.684423467916; max |grad diff| 5.6e-17
```

### Script 2: shared helpers

```python
"""Shared helpers for the L2-02 scripts: digits as tensors, an MLP builder,
and a compact version of the training loop written out in full in training_loop.py."""
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(4)  # tiny models: more threads mostly add overhead


def digits(n_train=None):
    """Same test split as L1-02 and L2-01; 20% of the rest is held out for validation."""
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=0, stratify=y_tr)
    if n_train is not None:
        X_tr, y_tr = X_tr[:n_train], y_tr[:n_train]
    scaler = StandardScaler().fit(X_tr)          # fit on training rows only
    as_x = lambda a: torch.tensor(scaler.transform(a), dtype=torch.float32)
    as_y = lambda a: torch.tensor(a, dtype=torch.long)
    return {"train": (as_x(X_tr), as_y(y_tr)), "val": (as_x(X_va), as_y(y_va)), "test": (as_x(X_te), as_y(y_te))}


def mlp(sizes, act=nn.ReLU, norm=None, dropout=0.0):
    """Linear -> [norm] -> act -> [dropout] for each hidden layer, then a linear output."""
    layers = []
    for m, k in zip(sizes[:-2], sizes[1:-1]):
        layers.append(nn.Linear(m, k))
        if norm is not None:
            layers.append(norm(k))
        layers.append(act())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()                                   # dropout off, batch norm uses running stats
    logits = model(X)
    return nn.functional.cross_entropy(logits, y).item(), (logits.argmax(1) == y).float().mean().item()


def fit(model, data, epochs, opt, scheduler=None, batch_size=64, seed=0):
    """Train; return per-epoch (train loss, val loss, val acc). scheduler steps once per batch."""
    X, y = data["train"]
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    history = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
            total += loss.item() * len(xb)
        history.append((total / len(X), *evaluate(model, *data["val"])))
    return history
```

This file only defines helpers. Note `evaluate` switches to eval mode and `fit` back to train mode every epoch; forgetting either is a common bug (section 4).

### Script 3: the full training loop

```python
"""The canonical PyTorch training loop, written out in full: Dataset, DataLoader,
nn.Module, optimizer, validation, keeping the best checkpoint, and one final test."""
import copy
import tempfile
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import digits

torch.manual_seed(0)


class ArrayDataset(Dataset):
    """A Dataset only needs __len__ and __getitem__; DataLoader does batching and shuffling."""
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, n_classes):
        super().__init__()                 # registers submodules and their parameters
        self.hidden = nn.Linear(d_in, d_hidden)
        self.out = nn.Linear(d_hidden, n_classes)

    def forward(self, x):
        return self.out(torch.relu(self.hidden(x)))   # logits: no softmax here


data = digits()
train_loader = DataLoader(ArrayDataset(*data["train"]), batch_size=64, shuffle=True,
                          generator=torch.Generator().manual_seed(0))
model = MLP(64, 128, 10)
print(model)
print("parameters:", sum(p.numel() for p in model.parameters()))
loss_fn = nn.CrossEntropyLoss()           # expects raw logits and integer labels
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)


@torch.no_grad()
def evaluate(X, y):
    model.eval()
    logits = model(X)
    return loss_fn(logits, y).item(), (logits.argmax(1) == y).float().mean().item()


best_val, best_state, best_epoch, patience = float("inf"), None, 0, 10
t0 = time.perf_counter()
for epoch in range(1, 201):
    model.train()
    for xb, yb in train_loader:
        opt.zero_grad()                    # 1. clear gradients from the previous step
        loss = loss_fn(model(xb), yb)      # 2. forward pass and loss
        loss.backward()                    # 3. backward pass fills p.grad
        opt.step()                         # 4. update parameters
    val_loss, val_acc = evaluate(*data["val"])
    if val_loss < best_val:                # early stopping on validation loss
        best_val, best_epoch = val_loss, epoch
        best_state = copy.deepcopy(model.state_dict())
    if epoch % 10 == 0 or epoch == 1:
        print(f"epoch {epoch:3d}  last batch loss {loss.item():.4f}  val loss {val_loss:.4f}  val acc {val_acc:.3f}")
    if epoch - best_epoch >= patience:
        print(f"stopping at epoch {epoch}: no val improvement for {patience} epochs")
        break
print(f"training time {time.perf_counter() - t0:.1f}s")

# Restore the best weights, save and reload them, then touch the test set once.
model.load_state_dict(best_state)
path = Path(tempfile.mkdtemp()) / "best.pt"
torch.save(model.state_dict(), path)
reloaded = MLP(64, 128, 10)
reloaded.load_state_dict(torch.load(path))
model = reloaded
test_loss, test_acc = evaluate(*data["test"])
print(f"best epoch {best_epoch} (val loss {best_val:.4f}); test loss {test_loss:.4f}, test acc {test_acc:.3f}")
```

```text
MLP(
  (hidden): Linear(in_features=64, out_features=128, bias=True)
  (out): Linear(in_features=128, out_features=10, bias=True)
)
parameters: 9610
epoch   1  last batch loss 1.8872  val loss 1.7730  val acc 0.710
epoch  10  last batch loss 0.1330  val loss 0.1904  val acc 0.937
epoch  20  last batch loss 0.0598  val loss 0.1071  val acc 0.972
epoch  30  last batch loss 0.0268  val loss 0.0864  val acc 0.972
epoch  40  last batch loss 0.0163  val loss 0.0796  val acc 0.972
epoch  50  last batch loss 0.0074  val loss 0.0762  val acc 0.972
epoch  60  last batch loss 0.0075  val loss 0.0748  val acc 0.976
epoch  70  last batch loss 0.0057  val loss 0.0745  val acc 0.976
epoch  80  last batch loss 0.0019  val loss 0.0746  val acc 0.976
stopping at epoch 85: no val improvement for 10 epochs
training time 2.0s
best epoch 75 (val loss 0.0737); test loss 0.0792, test acc 0.972
```

"Last batch loss" is noisy because the final batch of each epoch is small (1,005 is not a multiple of 64). Validation loss bottoms out at epoch 75 while validation accuracy holds at 0.976.

### Script 4: initialization

```python
"""Why the weight scale at initialization matters: signal size through 50 layers,
and what that does to training a 20-layer MLP."""
import math

import torch
from torch import nn

from common import digits, fit, mlp


def init_(model, scheme):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if scheme == "N(0, 0.01^2)":
                nn.init.normal_(m.weight, std=0.01)
            elif scheme == "Xavier":
                nn.init.xavier_normal_(m.weight)            # var = 2 / (fan_in + fan_out)
            elif scheme == "He":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")  # var = 2 / fan_in
            # "PyTorch default": leave nn.Linear's own reset_parameters() init alone
            if scheme != "PyTorch default":
                nn.init.zeros_(m.bias)
    return model


SCHEMES = ["N(0, 0.01^2)", "PyTorch default", "Xavier", "He"]

# 1. Push a random batch through 50 layers of width 256 and watch the activation scale.
torch.manual_seed(0)
x = torch.randn(512, 256)
print("per-unit std across the batch after layer 1 / 10 / 50 (input std 1.0)")
for act in [nn.Tanh, nn.ReLU]:
    for scheme in SCHEMES:
        net = init_(mlp([256] * 52, act=act), scheme)
        h, stds = x, []
        with torch.no_grad():
            for layer in list(net)[:-1]:
                h = layer(h)
                if isinstance(layer, act):
                    stds.append(h.std(dim=0).mean().item())  # spread ACROSS inputs
        print(f"  {act.__name__:<5} {scheme:<16} " + "  ".join(f"{stds[i]:.1e}" for i in [0, 9, 49]))

w = nn.Linear(256, 256).weight
print(f"\nnn.Linear default weight std {w.std().item():.4f}; 1/sqrt(3*256) = {1 / math.sqrt(3 * 256):.4f}; "
      f"He std sqrt(2/256) = {math.sqrt(2 / 256):.4f}")

# 2. Train a 20-hidden-layer ReLU MLP with SGD + momentum under each scheme.
data = digits()
print("\n20 hidden layers of width 128, SGD(lr=0.01, momentum=0.9), 20 epochs")
for scheme in SCHEMES:
    torch.manual_seed(0)
    model = init_(mlp([64] + [128] * 20 + [10]), scheme)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    hist = fit(model, data, epochs=20, opt=opt)
    print(f"  {scheme:<16} train loss epoch 1 {hist[0][0]:.3f} -> epoch 20 {hist[-1][0]:.3f}   "
          f"val acc {hist[-1][2]:.3f}")
```

```text
per-unit std across the batch after layer 1 / 10 / 50 (input std 1.0)
  Tanh  N(0, 0.01^2)     1.6e-01  1.1e-08  1.3e-40
  Tanh  PyTorch default  4.6e-01  2.9e-03  5.6e-09
  Tanh  Xavier           6.3e-01  2.3e-01  8.6e-02
  Tanh  He               7.2e-01  5.5e-01  5.6e-01
  ReLU  N(0, 0.01^2)     9.3e-02  1.0e-10  0.0e+00
  ReLU  PyTorch default  3.4e-01  5.0e-05  1.0e-09
  ReLU  Xavier           5.8e-01  8.0e-03  2.4e-09
  ReLU  He               8.3e-01  2.4e-01  1.5e-01

nn.Linear default weight std 0.0362; 1/sqrt(3*256) = 0.0361; He std sqrt(2/256) = 0.0884

20 hidden layers of width 128, SGD(lr=0.01, momentum=0.9), 20 epochs
  N(0, 0.01^2)     train loss epoch 1 2.303 -> epoch 20 2.303   val acc 0.099
  PyTorch default  train loss epoch 1 2.304 -> epoch 20 2.303   val acc 0.099
  Xavier           train loss epoch 1 2.303 -> epoch 20 1.795   val acc 0.321
  He               train loss epoch 1 2.358 -> epoch 20 0.017   val acc 0.964
```

### Script 5: normalization

```python
"""Batch norm and layer norm: rescuing a deep net that will not train,
and the train/eval-mode trap that comes with batch norm."""
import copy

import torch
from torch import nn

from common import digits, evaluate, fit, mlp

data = digits()

# 1. The 20-layer ReLU net with PyTorch's default init, which did not train in initialization.py.
print("20 hidden layers of width 128, default init, SGD(momentum=0.9), 20 epochs")
models = {}
for name, norm in [("no normalization", None), ("BatchNorm1d", nn.BatchNorm1d), ("LayerNorm", nn.LayerNorm)]:
    for lr in [0.01, 0.001]:
        torch.manual_seed(0)
        model = mlp([64] + [128] * 20 + [10], norm=norm)
        hist = fit(model, data, epochs=20, opt=torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9))
        models[name, lr] = model
        print(f"  {name:<17} lr {lr:<6} train loss epoch 1 {hist[0][0]:.3f} -> epoch 20 {hist[-1][0]:.3f}"
              f"   val acc {hist[-1][2]:.3f}")

# 2. Batch norm behaves differently in train and eval mode.
bn_model = models["BatchNorm1d", 0.01]
bn = bn_model[1]
print(f"\nfirst BatchNorm1d layer: running_mean[:3] {[round(v, 3) for v in bn.running_mean[:3].tolist()]}, "
      f"momentum {bn.momentum}")
X_te, y_te = data["test"]


def accuracy_in_train_mode(model, X, y, batch_size, sort_by_label):
    """Predict in train mode (the bug): each batch is normalized with its own statistics."""
    model = copy.deepcopy(model)         # train-mode forward passes would also update running stats
    model.train()
    order = torch.argsort(y, stable=True) if sort_by_label else torch.randperm(len(y), generator=torch.Generator().manual_seed(0))
    correct = 0
    with torch.no_grad():
        for i in range(0, len(y), batch_size):
            idx = order[i:i + batch_size]
            correct += (model(X[idx]).argmax(1) == y[idx]).sum().item()
    return correct / len(y)


print(f"test acc, eval mode (running statistics):          {evaluate(bn_model, X_te, y_te)[1]:.3f}")
for bs in [256, 16]:
    print(f"test acc, train mode, random batches of {bs:<4}:      {accuracy_in_train_mode(bn_model, X_te, y_te, bs, False):.3f}")
print(f"test acc, train mode, batches of 16 sorted by label: {accuracy_in_train_mode(bn_model, X_te, y_te, 16, True):.3f}")
try:
    bn_model.train()
    bn_model(X_te[:1])
except ValueError as err:
    print("train mode, batch of 1 ->", err)

ln_model = models["LayerNorm", 0.001]
ln_model.eval()
with torch.no_grad():
    alone = ln_model(X_te[:1])
    in_batch = ln_model(X_te[:64])[:1]
print(f"LayerNorm: one example alone vs inside a batch, max output diff {(alone - in_batch).abs().max().item():.1e}")
```

```text
20 hidden layers of width 128, default init, SGD(momentum=0.9), 20 epochs
  no normalization  lr 0.01   train loss epoch 1 2.304 -> epoch 20 2.303   val acc 0.099
  no normalization  lr 0.001  train loss epoch 1 2.304 -> epoch 20 2.303   val acc 0.095
  BatchNorm1d       lr 0.01   train loss epoch 1 2.263 -> epoch 20 0.123   val acc 0.976
  BatchNorm1d       lr 0.001  train loss epoch 1 2.374 -> epoch 20 0.302   val acc 0.960
  LayerNorm         lr 0.01   train loss epoch 1 2.249 -> epoch 20 2.324   val acc 0.099
  LayerNorm         lr 0.001  train loss epoch 1 2.285 -> epoch 20 0.044   val acc 0.940

first BatchNorm1d layer: running_mean[:3] [-0.014, 0.076, 0.033], momentum 0.1
test acc, eval mode (running statistics):          0.969
test acc, train mode, random batches of 256 :      0.956
test acc, train mode, random batches of 16  :      0.813
test acc, train mode, batches of 16 sorted by label: 0.137
train mode, batch of 1 -> Expected more than 1 value per channel when training, got input size torch.Size([1, 128])
LayerNorm: one example alone vs inside a batch, max output diff 8.3e-07
```

### Script 6: regularization

```python
"""Dropout, weight decay and early stopping with a large MLP on 300 training digits,
with clean labels and with 30% of training and validation labels replaced at random.
The test labels are always clean, so test accuracy measures what we actually want."""
import copy

import torch
from torch import nn

from common import digits, evaluate, fit, mlp

# Dropout itself: in train mode zero a fraction p and scale survivors by 1/(1-p).
drop = nn.Dropout(p=0.5)
torch.manual_seed(0)
print("train mode:", drop(torch.ones(10)).tolist())
drop.eval()
print("eval mode: ", drop(torch.ones(10)).tolist())


def corrupt(y, rate, seed):
    g = torch.Generator().manual_seed(seed)
    y = y.clone()
    flip = torch.rand(len(y), generator=g) < rate
    y[flip] = torch.randint(0, 10, (int(flip.sum()),), generator=g)   # may redraw the true label
    return y


def run(data, dropout, wd, seed, epochs=80):
    torch.manual_seed(seed)
    model = mlp([64, 256, 256, 10], dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        train_loss, val_loss, _ = fit(model, data, epochs=1, opt=opt, batch_size=32, seed=1000 * seed + epoch)[0]
        if val_loss < best_loss:                       # early stopping: keep the best-val checkpoint
            best_loss, best_state, best_epoch = val_loss, copy.deepcopy(model.state_dict()), epoch
    final_acc = evaluate(model, *data["test"])[1]
    model.load_state_dict(best_state)
    return train_loss, final_acc, evaluate(model, *data["test"])[1], best_epoch


print("\nMLP 64-256-256-10, AdamW(lr=1e-3), 80 epochs, batch 32, 3 seeds (mean test accuracy)")
print(f"{'labels':<10} {'setting':<21} {'final train loss':>16} {'final test acc':>15} "
      f"{'early-stop test acc':>20} {'stop epochs':>13}")
for rate, settings in [(0.0, ["none", "dropout 0.5 + wd 1.0"]),
                       (0.3, ["none", "weight decay 1.0", "dropout 0.5", "dropout 0.5 + wd 1.0"])]:
    data = digits(n_train=300)
    for split, seed in [("train", 1), ("val", 2)]:
        X, y = data[split]
        data[split] = (X, corrupt(y, rate, seed))
    for name in settings:
        dropout = 0.5 if "dropout" in name else 0.0
        wd = 1.0 if "wd" in name or "weight decay" in name else 0.0
        rows = [run(data, dropout, wd, seed) for seed in range(3)]
        avg = [sum(r[i] for r in rows) / 3 for i in range(3)]
        label = "clean" if rate == 0 else f"{rate:.0%} noisy"
        print(f"{label:<10} {name:<21} {avg[0]:>16.3f} {avg[1]:>15.3f} {avg[2]:>20.3f} "
              f"{str([r[3] for r in rows]):>13}")
```

```text
train mode: [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0]
eval mode:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

MLP 64-256-256-10, AdamW(lr=1e-3), 80 epochs, batch 32, 3 seeds (mean test accuracy)
labels     setting               final train loss  final test acc  early-stop test acc   stop epochs
clean      none                             0.000           0.952                0.951  [19, 13, 20]
clean      dropout 0.5 + wd 1.0             0.021           0.953                0.951  [53, 41, 64]
30% noisy  none                             0.004           0.769                0.872     [5, 4, 4]
30% noisy  weight decay 1.0                 0.031           0.778                0.869     [5, 4, 4]
30% noisy  dropout 0.5                      0.198           0.808                0.880    [10, 7, 9]
30% noisy  dropout 0.5 + wd 1.0             0.322           0.822                0.884   [10, 10, 9]
```

### Script 7: learning-rate schedules and the range test

```python
"""Learning-rate schedules on a fixed budget, and a learning-rate range test."""
import math

import torch
from torch.optim.lr_scheduler import LambdaLR

from common import digits, fit, mlp

data = digits()
EPOCHS, BATCH = 15, 64
STEPS = EPOCHS * math.ceil(len(data["train"][1]) / BATCH)   # the scheduler steps once per batch
WARMUP = STEPS // 10


def cosine(step):
    return 0.5 * (1 + math.cos(math.pi * min(step, STEPS) / STEPS))


SCHEDULES = {                         # multiplier on the base learning rate at each step
    "constant": lambda s: 1.0,
    "step (x0.1 at 50%, 75%)": lambda s: 0.1 ** ((s >= STEPS // 2) + (s >= 3 * STEPS // 4)),
    "cosine": cosine,
    "warmup 10% + cosine": lambda s: s / WARMUP if s < WARMUP else cosine((s - WARMUP) * STEPS / (STEPS - WARMUP)),
}

print(f"MLP 64-128-128-10, SGD(momentum=0.9), {EPOCHS} epochs = {STEPS} steps, mean of 3 seeds")
print(f"{'schedule':<25} {'base lr':>7} {'final train loss':>17} {'val loss':>9} {'val acc':>8}")
for base_lr in [0.03, 0.1, 0.3]:
    for name, f in SCHEDULES.items():
        results = []
        for seed in range(3):
            torch.manual_seed(seed)
            model = mlp([64, 128, 128, 10])
            opt = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
            results.append(fit(model, data, EPOCHS, opt, scheduler=LambdaLR(opt, f), batch_size=BATCH, seed=seed)[-1])
        avg = [sum(r[i] for r in results) / 3 for i in range(3)]
        print(f"{name:<25} {base_lr:>7} {avg[0]:>17.4f} {avg[1]:>9.4f} {avg[2]:>8.3f}")

# Range test: raise the learning rate exponentially every step and watch the loss.
torch.manual_seed(0)
model = mlp([64, 128, 128, 10])
opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
n_steps = 120
sched = LambdaLR(opt, lambda s: 10 ** (5 * s / n_steps))      # 1e-4 -> 10 over 120 steps
X, y = data["train"]
g = torch.Generator().manual_seed(0)
smooth = None
print("\nrange test (loss smoothed with an exponential moving average, factor 0.9)")
for step in range(n_steps):
    idx = torch.randint(0, len(y), (BATCH,), generator=g)
    lr = opt.param_groups[0]["lr"]
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(X[idx]), y[idx])
    loss.backward()
    opt.step()
    sched.step()
    smooth = loss.item() if smooth is None else 0.9 * smooth + 0.1 * loss.item()
    if step % 12 == 0 or not math.isfinite(smooth) or smooth > 10:
        print(f"  lr {lr:9.2e}  loss {smooth:.3f}")
        if not math.isfinite(smooth) or smooth > 10:
            print("  diverged: stop")
            break
```

```text
MLP 64-128-128-10, SGD(momentum=0.9), 15 epochs = 240 steps, mean of 3 seeds
schedule                  base lr  final train loss  val loss  val acc
constant                     0.03            0.0129    0.0876    0.970
step (x0.1 at 50%, 75%)      0.03            0.0395    0.1025    0.968
cosine                       0.03            0.0458    0.1056    0.967
warmup 10% + cosine          0.03            0.0423    0.1015    0.968
constant                      0.1            0.0009    0.1504    0.971
step (x0.1 at 50%, 75%)       0.1            0.0028    0.1652    0.968
cosine                        0.1            0.0023    0.1421    0.970
warmup 10% + cosine           0.1            0.0018    0.1519    0.972
constant                      0.3            2.1814    2.4472    0.151
step (x0.1 at 50%, 75%)       0.3            1.5903    1.9431    0.380
cosine                        0.3            0.5924    1.4738    0.718
warmup 10% + cosine           0.3            0.7960    1.5559    0.684

range test (loss smoothed with an exponential moving average, factor 0.9)
  lr  1.00e-04  loss 2.291
  lr  3.16e-04  loss 2.299
  lr  1.00e-03  loss 2.302
  lr  3.16e-03  loss 2.290
  lr  1.00e-02  loss 2.245
  lr  3.16e-02  loss 2.086
  lr  1.00e-01  loss 1.340
  lr  3.16e-01  loss 0.797
  lr  1.00e+00  loss 16.408
  diverged: stop
```

### Script 8: debugging

```python
"""Debugging a network that will not train: the initial-loss check, overfitting one
batch, and the symptoms of five planted bugs."""
import math
import warnings

import torch
from torch import nn

from common import digits, evaluate, fit, mlp

data = digits()
X, y = data["train"]
xb, yb = X[:32], y[:32]                 # the single batch we try to memorize


def overfit_one_batch(bug=None, steps=200):
    torch.manual_seed(0)
    model = mlp([64, 128, 10])
    opt = torch.optim.SGD(model.parameters(), lr=3.0 if bug == "lr = 3.0" else 0.1, momentum=0.9)
    targets = yb[torch.randperm(32)] if bug == "labels shuffled" else yb
    curve = []
    for step in range(steps):
        if bug != "no zero_grad":
            opt.zero_grad()
        logits = model(xb)
        if bug == "softmax before CrossEntropyLoss":
            logits = torch.softmax(logits, dim=1)    # the loss applies log-softmax itself
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        opt.step()
        curve.append(loss.item())
    return curve


# 1. At initialization, a 10-class classifier should be unsure: loss about ln(10).
torch.manual_seed(0)
model = mlp([64, 128, 10])
print(f"initial loss {evaluate(model, X, y)[0]:.3f}, ln(10) = {math.log(10):.3f}")

# 2. A correct pipeline drives the loss on one batch to ~0. Bugs show up as distinct curves.
print("\noverfitting one batch of 32, SGD(lr=0.1, momentum=0.9); loss at steps 0, 10, 50, 100, 199:")
for bug in [None, "softmax before CrossEntropyLoss", "no zero_grad", "lr = 3.0", "labels shuffled"]:
    c = overfit_one_batch(bug)
    print(f"  {str(bug or 'correct'):<32} " + "  ".join(f"{c[i]:9.3g}" for i in [0, 10, 50, 100, 199]))
print(f"  floor for double softmax: ln(1 + 9/e) = {math.log(1 + 9 / math.e):.4f}")

# 3. The shuffled-label bug passed the single-batch test (a big net memorizes anything),
#    so check it on held-out data, where it is obvious.
torch.manual_seed(0)
perm = torch.randperm(len(y))
for name, labels in [("correct labels", y), ("labels shuffled", y[perm])]:
    torch.manual_seed(0)
    model = mlp([64, 128, 10])
    hist = fit(model, {**data, "train": (X, labels)}, epochs=20, opt=torch.optim.Adam(model.parameters(), lr=1e-3))
    print(f"full training, {name:<16} train loss {hist[-1][0]:.3f}  val acc {hist[-1][2]:.3f}")

# 4. A silent broadcasting bug in regression: predictions (n, 1) vs targets (n,).
target = torch.tensor([1.0, 2.0, 3.0, 4.0])
pred = (target + 0.1).unsqueeze(1)          # nearly perfect predictions, shape (4, 1)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    wrong = nn.functional.mse_loss(pred, target)
print(f"\nMSE with shapes (4, 1) vs (4,): {wrong.item():.3f}; correct: "
      f"{nn.functional.mse_loss(pred.squeeze(1), target).item():.3f}")
print("  warning:", str(caught[0].message)[:95], "...")

# 5. Find where a NaN gradient comes from: sqrt has an infinite derivative at 0,
#    and the next backward step multiplies that inf by 0.
w = torch.tensor([1.0, 1.0], requires_grad=True)
dist = torch.sqrt(((w - torch.tensor([1.0, 1.0])) ** 2).sum())
dist.backward()
print(f"\ngradient of ||w - w0|| at w = w0: {w.grad.tolist()}")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")      # anomaly mode also prints a long traceback as a warning
    with torch.autograd.detect_anomaly():
        try:
            torch.sqrt(((w - torch.tensor([1.0, 1.0])) ** 2).sum()).backward()
        except RuntimeError as err:
            print("detect_anomaly ->", str(err)[:70])
```

```text
initial loss 2.325, ln(10) = 2.303

overfitting one batch of 32, SGD(lr=0.1, momentum=0.9); loss at steps 0, 10, 50, 100, 199:
  correct                                2.4      0.119   6.36e-05   3.39e-05   2.49e-05
  softmax before CrossEntropyLoss       2.31       2.21       1.48       1.46       1.46
  no zero_grad                           2.4    0.00148        558   9.55e+07   7.15e+17
  lr = 3.0                               2.4   5.98e+03   3.73e+19        nan        nan
  labels shuffled                       2.26      0.776     0.0029    0.00137    0.00082
  floor for double softmax: ln(1 + 9/e) = 1.4612
full training, correct labels   train loss 0.062  val acc 0.972
full training, labels shuffled  train loss 1.910  val acc 0.111

MSE with shapes (4, 1) vs (4,): 2.510; correct: 0.010
  warning: Using a target size (torch.Size([4])) that is different to the input size (torch.Size([4, 1])). ...

gradient of ||w - w0|| at w = w0: [nan, nan]
detect_anomaly -> Function 'PowBackward0' returned nan values in its 0th output.
```

### Script 9: reproducibility

```python
"""Reproducibility: same seed, same numbers; hidden consumers of the global RNG;
thread count; and how much results move between seeds anyway."""
import statistics

import torch

from common import digits, evaluate, fit, mlp

data = digits()


def train_once(seed, epochs=5, threads=4, extra_random_call=False):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = mlp([64, 128, 10], dropout=0.2)        # dropout draws random numbers every step
    if extra_random_call:
        torch.rand(1)                              # e.g. a debugging line someone added
    hist = fit(model, data, epochs, torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9), seed=seed)
    torch.set_num_threads(4)
    return [h[0] for h in hist], torch.cat([p.detach().flatten() for p in model.parameters()])


losses_a, w_a = train_once(0)
losses_b, w_b = train_once(0)
losses_c, w_c = train_once(1)
print("same seed twice, bitwise-identical weights:", torch.equal(w_a, w_b))
print("seed 0 vs seed 1, max weight difference:   ", f"{(w_a - w_c).abs().max().item():.3f}")
_, w_d = train_once(0, extra_random_call=True)
print("one extra torch.rand(1) after init, identical weights:", torch.equal(w_a, w_d))
_, w_e = train_once(0, threads=1)
print("4 threads vs 1 thread, identical weights:", torch.equal(w_a, w_e),
      f"(max diff {(w_a - w_e).abs().max().item():.1e})")

# DataLoader shuffling: global RNG vs a dedicated generator.
ds = torch.utils.data.TensorDataset(torch.arange(10))
for label, make_gen in [("global RNG", lambda: None), ("own generator", lambda: torch.Generator().manual_seed(0))]:
    orders = []
    for extra in [False, True]:
        torch.manual_seed(0)
        if extra:
            torch.rand(1)
        loader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=True, generator=make_gen())
        orders.append(next(iter(loader))[0].tolist())
    print(f"shuffle order with {label:<13}: {orders[0]} vs after an extra draw {orders[1]}")

# Seed-to-seed spread: the noise floor for comparing two training setups.
accs = []
for seed in range(10):
    torch.manual_seed(seed)
    model = mlp([64, 128, 10])
    fit(model, data, 20, torch.optim.AdamW(model.parameters(), lr=1e-3), seed=seed)
    accs.append(evaluate(model, *data["val"])[1])
print(f"\nval accuracy over 10 seeds: mean {statistics.mean(accs):.3f}, std {statistics.stdev(accs):.3f}, "
      f"min {min(accs):.3f}, max {max(accs):.3f}")
```

```text
same seed twice, bitwise-identical weights: True
seed 0 vs seed 1, max weight difference:    0.757
one extra torch.rand(1) after init, identical weights: False
4 threads vs 1 thread, identical weights: False (max diff 1.8e-07)
shuffle order with global RNG   : [6, 7, 1, 4, 2, 0, 9, 8, 3, 5] vs after an extra draw [1, 2, 3, 5, 7, 6, 0, 9, 4, 8]
shuffle order with own generator: [3, 7, 5, 2, 0, 8, 1, 6, 9, 4] vs after an extra draw [3, 7, 5, 2, 0, 8, 1, 6, 9, 4]

val accuracy over 10 seeds: mean 0.967, std 0.005, min 0.956, max 0.972
```

## Common mistakes

- **Forgetting `model.eval()`** before validation or inference, or forgetting `model.train()` afterward. With dropout or batch norm, predictions change; script 5 shows accuracy falling from 0.969 to 0.137 in the worst case.
- **Forgetting `zero_grad()`**, or calling it in the wrong place (after `backward()`). Gradients accumulate by design.
- **A softmax at the end of the model** when using `CrossEntropyLoss`, which expects logits. The loss floor gives it away.
- **Trusting the default initialization in a deep plain network.** Check that activations still vary across inputs at the last layer.
- **Batch norm with tiny or unshuffled batches**, where batch statistics are noisy or biased.
- **Tuning on the test set**, including choosing the early-stopping epoch by test accuracy. Use a validation set.
- **Comparing single runs.** A 0.01 accuracy difference was within seed noise in every experiment here.
- **Calling `scheduler.step()` at the wrong frequency.** Decide whether your schedule counts epochs or batches, and be consistent. `common.fit` steps per batch.

## Exercises

1. [warm-up] Compute the gradient of $\sum_i w_i^3$ by hand at $w = (1, -2, 3)$ and confirm with autograd. What happens if you skip clearing `w.grad` first?
2. [warm-up] What weight standard deviation keeps a ReLU layer with fan-in 512 stable? Check with `nn.init.kaiming_normal_`.
3. [warm-up] Count the parameters of script 6's MLP with a `BatchNorm1d` after each hidden linear layer.
4. [core] Add `num_workers=2` and a `seed_worker` function to script 3's `DataLoader`, following the PyTorch reproducibility notes, and confirm two runs still match bitwise.
5. [core] Add $\sigma = 1.5\sqrt{2/n}$ to script 4's schemes. Predict, then measure, what happens by layer 50.
6. [core] Repeat script 5 with Adam at lr $10^{-3}$. Does layer norm still need a lower learning rate than batch norm?
7. [core] Vary the label-noise rate in script 6 (0, 10%, 30%, 50%) and track the gap between early-stopped and final test accuracy.
8. [stretch] Implement batch norm as your own `nn.Module`, with running statistics, and match `nn.BatchNorm1d` in both modes to $10^{-6}$.
9. [stretch] Run the range test with batch sizes 16, 64 and 256. Does the blow-up point move the way the linear scaling rule suggests?

### Hints and answers

1. $3w^2 = (3, 12, 27)$; without clearing, it is added to the old `.grad`.
2. $\sqrt{2/512} = 0.0625$.
3. $85{,}002$ plus $\gamma$ and $\beta$ for two layers of 256: $86{,}026$. Running statistics are buffers, not parameters.
4. Seed NumPy and `random` in `seed_worker` from `torch.initial_seed()` and keep the seeded `generator`.
5. The standard deviation grows about 1.5-fold per layer, so activations reach very large values by layer 50 (exploding rather than vanishing).
6. Run it rather than assume; Adam rescales each parameter's step, so the answer can differ from SGD's.
7. Expect the gap to grow with noise, since there is more to memorize once the signal is fit.
8. Normalize with the biased batch variance and store the unbiased one in the running estimate [PyTorch, BatchNorm1d].
9. Measure; the linear scaling rule is an empirical guideline from large-batch ImageNet training, not a law.

## Check your understanding

**1. What happens without `optimizer.zero_grad()`?** Gradients accumulate across `backward()` calls by design, so each step uses the sum of all previous gradients. The effective step grows and training diverges (script 8).

**2. Why does He use variance $2/n$ and Xavier about $1/n$?** Both keep the signal's variance constant across layers. Tanh near zero passes the full variance, so $\sigma^2 = 1/n$; ReLU passes half the second moment of a symmetric input, so $\sigma^2 = 2/n$ compensates.

**3. Why must a batch norm model be in eval mode at test time?** In train mode it normalizes with the current batch's statistics, so a prediction depends on the other examples in the batch, and a single example fails. Eval mode uses running averages from training. Layer norm has no such dependence.

**4. On noisy labels, why did early stopping beat dropout and weight decay?** The network fits the consistent structure first and memorizes noise later; early stopping keeps the checkpoint in between. Dropout and weight decay slowed memorization (final training losses 0.03 to 0.32 instead of 0.004) but did not stop it.

**5. A model passes the one-batch test but validation accuracy is at chance. What do you check?** The data pipeline: inputs and labels still aligned after shuffling or splitting, matching preprocessing, eval mode. The one-batch test only shows the model can memorize.

## Further reading

- **Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola, "Dive into Deep Learning"** (2023, Cambridge University Press; free online, with PyTorch code). <https://d2l.ai/>
  *Read for:* chapter 6 (Builders' Guide) on modules, parameters and initialization, and section 12.11 on learning-rate scheduling. *Level:* intro.
- **Andrej Karpathy, "A Recipe for Training Neural Networks"** (2019, blog post). <https://karpathy.github.io/2019/04/25/recipe/>
  *Read for:* a step-by-step debugging and tuning process, including the initial-loss check and overfitting one batch. *Level:* intro.
- **PyTorch documentation: "Autograd mechanics", "Reproducibility", `nn.Linear`, `nn.BatchNorm1d`, `nn.Dropout`, `nn.CrossEntropyLoss`** (version 2.15, checked October 2026). <https://docs.pytorch.org/docs/stable/notes/autograd.html>, <https://docs.pytorch.org/docs/stable/notes/randomness.html>, <https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html>, <https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html>, <https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html>, <https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>
  *Read for:* the exact contracts this module relies on: default initialization, running statistics, dropout scaling, logits in the loss, and what is and is not reproducible. *Level:* intro to intermediate.
- **Varun Godbole, George E. Dahl, Justin Gilmer, Christopher J. Shallue, Zachary Nado, "Deep Learning Tuning Playbook"** (2023, GitHub, version 1). <https://github.com/google-research/tuning_playbook>
  *Read for:* a systematic process for tuning, including learning-rate decay choices and how to run comparisons. *Level:* intermediate.
- **Xavier Glorot, Yoshua Bengio, "Understanding the difficulty of training deep feedforward neural networks"** (2010, AISTATS, PMLR 9, 249 to 256). <https://proceedings.mlr.press/v9/glorot10a.html>
  *Read for:* how activations and gradients behave across layers at initialization, and the Xavier scheme. *Level:* intermediate.
- **Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"** (2015, arXiv:1502.01852). <https://arxiv.org/abs/1502.01852>
  *Read for:* the derivation of the $2/n$ initialization for rectifier networks. *Level:* intermediate.
- **Sergey Ioffe, Christian Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"** (2015, ICML, PMLR 37, 448 to 456; arXiv:1502.03167). <https://arxiv.org/abs/1502.03167>
  *Read for:* the original method, its train and inference procedures, and the internal-covariate-shift motivation. *Level:* intermediate.
- **Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton, "Layer Normalization"** (2016, arXiv:1607.06450). <https://arxiv.org/abs/1607.06450>
  *Read for:* per-example normalization and why it suits recurrent networks. *Level:* intermediate.
- **Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, Aleksander Madry, "How Does Batch Normalization Help Optimization?"** (2018, NeurIPS; arXiv:1805.11604). <https://arxiv.org/abs/1805.11604>
  *Read for:* the argument that batch norm works by smoothing the loss landscape rather than by reducing covariate shift. *Level:* advanced.
- **Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"** (2014, JMLR 15(56), 1929 to 1958). <https://jmlr.org/papers/v15/srivastava14a.html>
  *Read for:* the method, its interpretation as averaging many thinned networks, and extensive experiments. *Level:* intermediate.
- **Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals, "Understanding deep learning requires rethinking generalization"** (2017, ICLR; arXiv:1611.03530). <https://arxiv.org/abs/1611.03530>
  *Read for:* the random-label experiments behind section 5, and why they unsettle classical explanations of generalization. *Level:* intermediate.
- **Leslie N. Smith, "Cyclical Learning Rates for Training Neural Networks"** (2017, WACV; arXiv:1506.01186). <https://arxiv.org/abs/1506.01186>
  *Read for:* the learning-rate range test and cyclical schedules. *Level:* intermediate.
