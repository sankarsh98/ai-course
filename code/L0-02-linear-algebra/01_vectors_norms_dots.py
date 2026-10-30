"""Vectors, dot products, norms, and what happens to angles in high dimensions."""
import numpy as np

rng = np.random.default_rng(0)

# 1. A small worked example.
u = np.array([3.0, 4.0])
v = np.array([4.0, 3.0])
w = np.array([-4.0, 3.0])
print("u.v =", u @ v, "| u.w =", u @ w, "(orthogonal)")
print("||u||_2 =", np.linalg.norm(u), "| ||u||_1 =", np.linalg.norm(u, 1),
      "| ||u||_inf =", np.linalg.norm(u, np.inf))
cos_uv = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
print(f"cos(u, v) = {cos_uv:.4f} -> angle {np.degrees(np.arccos(cos_uv)):.2f} degrees")

# 2. Projection of u onto the direction of v: the "shadow" of u along v.
proj = (u @ v) / (v @ v) * v
resid = u - proj
print("projection of u onto v:", proj, "| residual:", resid,
      "| residual . v =", round(resid @ v, 12))

# 3. Cauchy-Schwarz |a.b| <= ||a|| ||b||, checked on random pairs.
a = rng.normal(size=(10_000, 50))
b = rng.normal(size=(10_000, 50))
lhs = np.abs((a * b).sum(1))
rhs = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
print("\nCauchy-Schwarz holds for all 10,000 pairs:", bool(np.all(lhs <= rhs + 1e-12)))

# 4. High dimensions: random directions are nearly orthogonal, and
#    distances concentrate (nearest and farthest neighbor look alike).
print(f"\n{'dim':>6}{'mean |cos|':>12}{'std cos':>10}{'max/min dist':>14}")
for d in (2, 10, 100, 1000, 10000):
    X = rng.normal(size=(500, d))
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    C = Xn @ Xn.T
    off = C[~np.eye(500, dtype=bool)]
    q = rng.normal(size=d)
    dist = np.linalg.norm(X - q, axis=1)
    print(f"{d:>6}{np.abs(off).mean():>12.4f}{off.std():>10.4f}"
          f"{dist.max() / dist.min():>14.3f}")
print("theory: std of cos between random directions ~ 1/sqrt(d); "
      f"1/sqrt(1000) = {1/np.sqrt(1000):.4f}")
