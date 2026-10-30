"""The perceptron rule: it learns AND and OR, never settles on XOR,
and one hidden layer of two ReLU units fixes that."""
import numpy as np


def perceptron(X, y, max_epochs=100):
    """Labels y in {-1, +1}. On a mistake, move the boundary toward the example."""
    w, b = np.zeros(X.shape[1]), 0.0
    for epoch in range(1, max_epochs + 1):
        mistakes = 0
        for x_i, y_i in zip(X, y):
            if y_i * (x_i @ w + b) <= 0:      # wrong side of (or on) the boundary
                w, b = w + y_i * x_i, b + y_i
                mistakes += 1
        if mistakes == 0:
            return w, b, epoch, True
    return w, b, max_epochs, False


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
for name, target in [("AND", [0, 0, 0, 1]), ("OR", [0, 1, 1, 1]), ("XOR", [0, 1, 1, 0])]:
    y = 2 * np.array(target) - 1
    w, b, epochs, converged = perceptron(X, y)
    pred = (X @ w + b > 0).astype(int)
    print(f"{name}: converged={converged} after {epochs} epochs, w={w}, b={b:+.0f}, "
          f"predictions={pred.tolist()} targets={target}")

# Stacking linear layers does not help: two linear maps compose to one linear map.
rng = np.random.default_rng(0)
W1, W2 = rng.normal(size=(2, 5)), rng.normal(size=(5, 1))
print("\nlinear-linear network equals one linear layer:", np.allclose((X @ W1) @ W2, X @ (W1 @ W2)))

# One hidden layer with a nonlinearity does. Weights chosen by hand:
#   h1 = relu(x1 + x2), h2 = relu(x1 + x2 - 1), output = h1 - 2*h2
W1 = np.array([[1.0, 1.0], [1.0, 1.0]])
b1 = np.array([0.0, -1.0])
w2 = np.array([1.0, -2.0])
H = np.maximum(X @ W1 + b1, 0.0)
print("hidden activations per input:", H.tolist())
print("XOR network output:", (H @ w2).tolist())
