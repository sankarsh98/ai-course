"""Train the from-scratch MLP on scikit-learn's 8x8 digits and compare it with
softmax regression on the same split as L1-02."""
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import mlp_numpy as mlp

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
scaler = StandardScaler().fit(X_tr)                     # training statistics only
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
print(f"train {X_tr.shape}, test {X_te.shape}")

baseline = LogisticRegression(C=1.0, max_iter=5000).fit(X_tr, y_tr)
print(f"softmax regression (L1-02 baseline): test accuracy {baseline.score(X_te, y_te):.3f}\n")

# Sanity check before training: with random weights the loss should be near ln(10).
params = mlp.init_params([64, 128, 10], np.random.default_rng(0))
print(f"initial loss {mlp.loss_and_grads(params, X_tr, y_tr)[0]:.3f} (ln 10 = {np.log(10):.3f})")

# One run in detail: watch train and test loss separate.
mlp.train(params, X_tr, y_tr, np.random.default_rng(0), lr=0.05, epochs=60, log_every=10,
          X_test=X_te, y_test=y_te)

# Several architectures, 5 seeds each. Settings were fixed before looking at test results.
print(f"\n{'hidden layers':<16} {'act':<5} {'train acc':>9} {'test acc (5 seeds)':>22} {'s/run':>6}")
for hidden, act in [([32], "relu"), ([128], "relu"), ([128], "tanh"), ([128, 128], "relu")]:
    train_acc, test_acc = [], []
    t0 = time.perf_counter()
    for seed in range(5):
        rng = np.random.default_rng(seed)
        p = mlp.train(mlp.init_params([64, *hidden, 10], rng), X_tr, y_tr, rng, act=act, lr=0.05, epochs=60)
        train_acc.append(np.mean(mlp.predict(p, X_tr, act) == y_tr))
        test_acc.append(np.mean(mlp.predict(p, X_te, act) == y_te))
    secs = (time.perf_counter() - t0) / 5
    print(f"{str(hidden):<16} {act:<5} {np.mean(train_acc):>9.3f} "
          f"{np.mean(test_acc):>14.3f} +/- {np.std(test_acc):.3f} {secs:>6.2f}")
