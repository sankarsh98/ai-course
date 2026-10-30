"""Vanishing and exploding gradients in a vanilla RNN, measured directly.
dh_T/dh_0 is a product of T Jacobians J_t = diag(phi'(a_t)) W. We compute its norm."""
import numpy as np

rng = np.random.default_rng(0)
H = 64                                        # hidden size


def random_orthogonal(n):
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    return q * np.sign(np.diag(r))            # all singular values exactly 1


def jacobian_norms(W, U, T_list, tanh=True, x_scale=1.0, seed=1):
    """Run h_t = phi(W h_{t-1} + U x_t) and return the spectral norm of dh_T/dh_0 for each T."""
    r = np.random.default_rng(seed)
    h = np.zeros(H)
    J = np.eye(H)
    out = {}
    for t in range(1, max(T_list) + 1):
        a = W @ h + U @ (x_scale * r.normal(size=H))
        h = np.tanh(a) if tanh else a
        d = 1 - h ** 2 if tanh else np.ones(H)    # phi'(a_t); for tanh it is 1 - tanh(a)^2
        J = (d[:, None] * W) @ J                  # chain rule: J_t times everything before
        if t in T_list:
            out[t] = np.linalg.norm(J, 2)
    return out, h


# 1. The scalar picture: h_t = w h_{t-1}, so dh_T/dh_0 = w^T.
for w in [0.9, 1.0, 1.1]:
    print(f"scalar w={w}: w^10={w ** 10:.3g}, w^50={w ** 50:.3g}, w^100={w ** 100:.3g}")

# 2. Linear RNN, W = rho * orthogonal: every direction scales by exactly rho per step.
T_list = [1, 10, 25, 50, 100]
Q = random_orthogonal(H)
U = rng.normal(size=(H, H)) / np.sqrt(H)
print(f"\n||dh_T/dh_0|| (spectral norm), linear RNN")
print("rho   " + "".join(f"{'T=' + str(T):>11s}" for T in T_list))
for rho in [0.9, 1.0, 1.1]:
    norms, _ = jacobian_norms(rho * Q, U, T_list, tanh=False)
    print(f"{rho:<5} " + "".join(f"{norms[T]:11.3g}" for T in T_list))

# 3. tanh RNN with a Gaussian W of scale g/sqrt(H) (spectral radius about g).
print(f"\n||dh_T/dh_0||, tanh RNN, W ~ N(0, g^2/H)")
print("g     " + "".join(f"{'T=' + str(T):>11s}" for T in T_list) + "   mean |h_T|  saturated")
for g in [0.5, 1.0, 1.5, 3.0]:
    W = g * rng.normal(size=(H, H)) / np.sqrt(H)
    norms, h = jacobian_norms(W, U, T_list, tanh=True, x_scale=0.5)
    print(f"{g:<5} " + "".join(f"{norms[T]:11.3g}" for T in T_list)
          + f"   {np.abs(h).mean():10.2f}  {np.mean(np.abs(h) > 0.95):9.0%}")

# 4. Why clipping helps: rescale the gradient if its norm exceeds a threshold.
grad = rng.normal(size=1000) * 50
c = 1.0
clipped = grad * min(1.0, c / np.linalg.norm(grad))
cos = grad @ clipped / (np.linalg.norm(grad) * np.linalg.norm(clipped))
print(f"\nclip by norm: ||g|| {np.linalg.norm(grad):.1f} -> {np.linalg.norm(clipped):.1f}, "
      f"direction unchanged (cosine {cos:.3f})")
