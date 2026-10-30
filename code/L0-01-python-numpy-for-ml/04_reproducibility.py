"""Seeds and random number generators: what makes a run reproducible."""
import numpy as np

# 1. Same seed -> same stream. Different seed -> different stream.
a = np.random.default_rng(42).normal(size=3)
b = np.random.default_rng(42).normal(size=3)
c = np.random.default_rng(43).normal(size=3)
print("seed 42:", np.round(a, 4))
print("seed 42:", np.round(b, 4), "| identical:", np.array_equal(a, b))
print("seed 43:", np.round(c, 4))

# 2. A stream depends on everything drawn before. Inserting one extra draw
#    shifts every later number, which is why "I only added a print" can
#    change your results if that code also touches the RNG.
rng = np.random.default_rng(42)
init_w = rng.normal(size=3)
rng2 = np.random.default_rng(42)
_ = rng2.random()                  # one extra draw, e.g. a new augmentation
init_w2 = rng2.normal(size=3)
print("\nweights, clean run:       ", np.round(init_w, 4))
print("weights, one extra draw:  ", np.round(init_w2, 4))

# 3. Give each component its own generator so changes stay local.
#    spawn() derives statistically independent child streams from one seed.
root = np.random.default_rng(42)
rng_init, rng_data, rng_dropout = root.spawn(3)
w = rng_init.normal(size=3)
_ = rng_data.permutation(10)       # extra draws here...
_ = rng_data.random(1000)
w_again = np.random.default_rng(42).spawn(3)[0].normal(size=3)
print("\ninit stream unaffected by data stream:", np.array_equal(w, w_again))


# 4. Pass generators into functions instead of using hidden global state.
def train_test_split(n, test_frac, rng):
    idx = rng.permutation(n)
    k = int(n * test_frac)
    return idx[k:], idx[:k]


tr, te = train_test_split(10, 0.3, np.random.default_rng(7))
print("\nsplit with seed 7: train", tr, "test", te)

# 5. Legacy global API still exists and still works, but it is shared state:
#    any library that calls np.random.* moves the same stream.
np.random.seed(0)
first = np.random.rand()
np.random.seed(0)
np.random.rand()                   # pretend a library you imported did this
second = np.random.rand()
print(f"\nlegacy global: {first:.4f} vs {second:.4f} (changed by someone else's draw)")
