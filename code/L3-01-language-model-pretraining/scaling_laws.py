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
