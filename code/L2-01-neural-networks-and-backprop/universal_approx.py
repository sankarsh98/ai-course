"""What one hidden layer can represent, how many units it takes, what depth buys,
and whether gradient-based training actually finds the weights that exist."""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

relu = lambda z: np.maximum(z, 0.0)
x = np.linspace(0.0, 1.0, 2 ** 14 + 1)  # dyadic grid: every kink below lands on a grid point


def interpolating_net(f, n_units):
    """One hidden layer of n_units ReLUs that linearly interpolates f at n_units + 1 knots.
    Unit k is relu(x - t_k); its output weight is the change of slope at knot t_k."""
    t = np.linspace(0.0, 1.0, n_units + 1)
    slopes = np.diff(f(t)) / np.diff(t)
    out_w = np.diff(slopes, prepend=0.0)          # first unit sets the initial slope
    return lambda x: f(t[0]) + relu(x[:, None] - t[:-1][None, :]) @ out_w


def count_pieces(y):
    """Number of linear pieces of a piecewise-linear function sampled on the grid x."""
    slope = np.round(np.diff(y) / np.diff(x), 6)
    return 1 + int(np.sum(slope[1:] != slope[:-1]))


# 1. Width needed for a given accuracy, for a smooth and a wiggly target.
targets = {"sin(2 pi x)": lambda x: np.sin(2 * np.pi * x),
           "sin(2 pi 20 x)": lambda x: np.sin(2 * np.pi * 20 * x)}
print(f"{'units':>6} | " + " | ".join(f"max error on {k:<14}" for k in targets))
for n in [4, 16, 64, 256, 1024]:
    errs = [np.abs(interpolating_net(f, n)(x) - f(x)).max() for f in targets.values()]
    print(f"{n:>6} | " + " | ".join(f"{e:>27.2e}" for e in errs))

# 2. Depth: the tent map g(x) = 2 relu(x) - 4 relu(x - 1/2) uses 2 ReLUs.
#    Composing it k times uses 2k units in k layers and makes 2**k linear pieces.
#    One hidden layer with m ReLUs on a line has at most m + 1 pieces.
tent = lambda z: 2 * relu(z) - 4 * relu(z - 0.5)
y = x.copy()
print("\nlayers  units  pieces  (a 1-hidden-layer net needs >= pieces - 1 units)")
for k in range(1, 7):
    y = tent(y)
    print(f"{k:>6} {2 * k:>6} {count_pieces(y):>7}")

# 3. Existence is not learnability. Train on the k = 3 sawtooth (8 pieces). Three layers
#    of 2 ReLUs can represent it exactly; does Adam find those weights from a random start?
x_tr = np.linspace(0, 1, 500)[:, None]
saw = tent(tent(tent(x_tr[:, 0])))
print("\ntraining on the 8-piece sawtooth, full-batch Adam, 2000 steps, 3 seeds")
print(f"  (predicting the mean everywhere gives MSE {saw.var():.4f})")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
for layers in [(2, 2, 2), (16,), (256,), (64, 64)]:
    mses = []
    for seed in range(3):
        net = MLPRegressor(hidden_layer_sizes=layers, activation="relu", learning_rate_init=1e-2,
                           batch_size=len(x_tr), max_iter=2000, n_iter_no_change=2000, tol=0,
                           random_state=seed)
        mses.append(np.mean((net.fit(x_tr, saw).predict(x_tr) - saw) ** 2))
    n_par = sum(a * b + b for a, b in zip((1,) + layers, layers + (1,)))
    print(f"  hidden {str(layers):<10} {n_par:>5} params  training MSE " + "  ".join(f"{m:.4f}" for m in mses))
