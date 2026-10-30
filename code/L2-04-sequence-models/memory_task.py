"""A long-range memory task: read a sequence, then output its FIRST token.
Everything after the first token is a distraction. Chance accuracy is 1/8."""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)  # fixed thread count, so runs on a shared machine are comparable
V, HID = 8, 64            # vocabulary size, hidden size


def batch(n, T, gen):
    x = torch.randint(0, V, (n, T), generator=gen)
    return x, x[:, 0].clone()            # target = the first token


class Reader(nn.Module):
    def __init__(self, cell, forget_bias=None):
        super().__init__()
        self.emb = nn.Embedding(V, 16)
        self.rnn = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[cell](16, HID, batch_first=True)
        self.out = nn.Linear(HID, V)
        if forget_bias is not None:   # PyTorch orders LSTM gate biases (input, forget, cell, output)
            with torch.no_grad():
                self.rnn.bias_ih_l0[HID:2 * HID] = forget_bias
                self.rnn.bias_hh_l0[HID:2 * HID] = 0.0

    def forward(self, x):
        h_all, _ = self.rnn(self.emb(x))
        return self.out(h_all[:, -1])    # classify from the last hidden state only


def run(cell, T, forget_bias=None, steps=800, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = Reader(cell, forget_bias)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    max_norm = 0.0
    for _ in range(steps):
        x, y = batch(128, T, gen)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        # Clip the global gradient norm at 1 (Pascanu et al., 2013); record the largest raw norm.
        max_norm = max(max_norm, nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        opt.step()
    x, y = batch(2000, T, torch.Generator().manual_seed(999))
    with torch.no_grad():
        acc = (model(x).argmax(1) == y).float().mean().item()
    return acc, max_norm


configs = [("RNN", "RNN", None), ("GRU", "GRU", None), ("LSTM", "LSTM", None),
           ("LSTM, forget bias 3", "LSTM", 3.0)]
print("test accuracy (largest raw gradient norm seen in training); chance = 0.125")
print(f"{'length T':>8s} " + " ".join(f"{name:>19s}" for name, _, _ in configs))
for T in [5, 20, 50]:
    t0 = time.time()
    cells = []
    for _, cell, fb in configs:
        acc, gmax = run(cell, T, fb)
        cells.append(f"{acc:8.3f} ({gmax:6.2f})")
    print(f"{T:8d} " + " ".join(f"{c:>19s}" for c in cells) + f"   {time.time() - t0:.0f}s")
