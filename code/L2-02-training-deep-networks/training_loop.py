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
