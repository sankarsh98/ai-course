"""A multilayer perceptron in NumPy with a hand-written backward pass.

Shapes: a batch X is (n, d). Layer l has weights W (fan_in, fan_out) and bias
b (fan_out,), so Z = H @ W + b is (n, fan_out). The last layer has no
activation: its outputs are logits that go into softmax cross-entropy.
"""
import numpy as np

# Each activation: (function, derivative). The derivative receives both the
# pre-activation Z and the output H, because some are cheaper from H.
ACTIVATIONS = {
    "relu": (lambda Z: np.maximum(Z, 0.0), lambda Z, H: (Z > 0).astype(Z.dtype)),
    "tanh": (np.tanh, lambda Z, H: 1.0 - H ** 2),
    "sigmoid": (lambda Z: 1.0 / (1.0 + np.exp(-Z)), lambda Z, H: H * (1.0 - H)),
}


def init_params(sizes, rng, dtype=np.float64):
    """Gaussian weights with variance 2 / fan_in (He scaling; L2-02 explains why)."""
    return [[rng.normal(0.0, np.sqrt(2.0 / m), size=(m, k)).astype(dtype), np.zeros(k, dtype)]
            for m, k in zip(sizes[:-1], sizes[1:])]


def log_softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)  # shift for stability (L0-01)
    return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))


def forward(params, X, act="relu"):
    """Return logits and a cache of (layer input, pre-activation, output) per layer."""
    f, _ = ACTIVATIONS[act]
    cache, H = [], X
    for l, (W, b) in enumerate(params):
        Z = H @ W + b
        H_out = Z if l == len(params) - 1 else f(Z)
        cache.append((H, Z, H_out))
        H = H_out
    return H, cache


def loss_and_grads(params, X, y, act="relu"):
    """Mean cross-entropy over the batch and its gradient for every W and b."""
    logits, cache = forward(params, X, act)
    n = X.shape[0]
    logp = log_softmax(logits)
    loss = -logp[np.arange(n), y].mean()

    dZ = np.exp(logp)                 # softmax probabilities P
    dZ[np.arange(n), y] -= 1.0        # P - Y
    dZ /= n                           # because the loss is a mean over n rows
    _, dact = ACTIVATIONS[act]
    grads = [None] * len(params)
    for l in reversed(range(len(params))):
        H_in, _, _ = cache[l]
        W, _ = params[l]
        grads[l] = [H_in.T @ dZ, dZ.sum(axis=0)]   # (fan_in, n) @ (n, fan_out)
        if l > 0:
            dH = dZ @ W.T                           # gradient wrt this layer's input
            _, Z_prev, H_prev = cache[l - 1]
            dZ = dH * dact(Z_prev, H_prev)          # through the elementwise nonlinearity
    return loss, grads


def predict(params, X, act="relu"):
    return forward(params, X, act)[0].argmax(axis=1)


def train(params, X, y, rng, act="relu", lr=0.1, epochs=50, batch_size=32, log_every=0,
          X_test=None, y_test=None):
    """Plain minibatch SGD (optimizers are L0-05's topic). Updates params in place."""
    n = X.shape[0]
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)                  # reshuffle every epoch
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            _, grads = loss_and_grads(params, X[idx], y[idx], act)
            for (W, b), (dW, db) in zip(params, grads):
                W -= lr * dW
                b -= lr * db
        if log_every and epoch % log_every == 0:
            train_loss = loss_and_grads(params, X, y, act)[0]
            msg = f"epoch {epoch:3d}  train loss {train_loss:.4f}  train acc {np.mean(predict(params, X, act) == y):.3f}"
            if X_test is not None:
                test_loss = loss_and_grads(params, X_test, y_test, act)[0]
                msg += f"  test loss {test_loss:.4f}  test acc {np.mean(predict(params, X_test, act) == y_test):.3f}"
            print(msg)
    return params
