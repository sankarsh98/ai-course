"""Check the reverse-mode engine against central-difference numerical gradients."""
import random

from engine import Value


def numerical_grad(fn, xs, h=1e-6):
    """Central differences on plain floats: 2 evaluations of fn per input."""
    f = lambda *a: fn(*[Value(x) for x in a]).data
    return [(f(*xs[:i], x + h, *xs[i + 1:]) - f(*xs[:i], x - h, *xs[i + 1:])) / (2 * h)
            for i, x in enumerate(xs)]


def engine_grad(fn, xs):
    vs = [Value(x) for x in xs]
    nodes = fn(*vs).backward()
    return [v.grad for v in vs], len(nodes)


def rel_err(a, b, floor=1e-4):
    # The floor matters: when the true gradient is exactly 0, central differences
    # return round-off noise near 1e-10, which a pure ratio would call a 100% error.
    return abs(a - b) / max(floor, abs(a) + abs(b))


def worst_error(fn, xs):
    g_auto, n_nodes = engine_grad(fn, xs)
    return max(rel_err(a, b) for a, b in zip(g_auto, numerical_grad(fn, xs))), n_nodes


def check(name, fn, xs, tol=1e-6):
    err, n_nodes = worst_error(fn, xs)
    print(f"[{'PASS' if err < tol else 'FAIL'}] {name:<26} nodes={n_nodes:<3} worst rel err={err:.1e}")
    return err < tol


def tiny_net(*p):
    # 2 inputs -> 3 tanh units -> 1 output, squared error against a target.
    x, target = (0.5, -1.2), 0.3
    hidden = [(p[2 * j] * x[0] + p[2 * j + 1] * x[1]).tanh() for j in range(3)]
    return (sum(p[6 + j] * hidden[j] for j in range(3)) - target) ** 2


UNARY = [lambda v: v.tanh(), lambda v: v.sin(), lambda v: (v * 0.3).exp(), lambda v: v ** 2]
BINARY = [lambda a, b: a + b, lambda a, b: a * b, lambda a, b: a - b]


def random_expression(seed, n_inputs=3, n_ops=12):
    """Draw a random 'program' once, so every call builds the same graph."""
    rng = random.Random(seed)
    plan = []
    for k in range(n_ops):
        size = n_inputs + k  # values available at step k
        if rng.random() < 0.5:
            plan.append((UNARY[rng.randrange(4)], rng.randrange(size), None))
        else:
            plan.append((BINARY[rng.randrange(3)], rng.randrange(size), rng.randrange(size)))

    def fn(*inputs):
        pool = list(inputs)
        for op, i, j in plan:
            pool.append(op(pool[i]) if j is None else op(pool[i], pool[j]))
        return pool[-1] + pool[-2]
    return fn


g, _ = engine_grad(lambda a, b: a.log() + a * b - b.sin(), [2.0, 5.0])
print(f"worked example: dy/dx1={g[0]:.6f}, dy/dx2={g[1]:.6f} (hand: 5.500000, 1.716338)")

random.seed(0)
results = [
    check("ln(a) + a*b - sin(b)", lambda a, b: a.log() + a * b - b.sin(), [2.0, 5.0]),
    # Forks: x is used more than once, so contributions from all paths must add.
    check("x*x + x (fork)", lambda x: x * x + x, [3.0]),
    check("(x + x) * x (fork)", lambda x: (x + x) * x, [3.0]),
    check("(1 + exp(-a)) ** -1 / b", lambda a, b: (1 + (-a).exp()) ** -1 / b, [0.7, 2.5]),
    check("tiny tanh net, 9 params", tiny_net, [random.uniform(-1, 1) for _ in range(9)]),
]

passed = sum(worst_error(random_expression(s), [random.uniform(-1.5, 1.5) for _ in range(3)])[0] < 1e-5
             for s in range(200))
print(f"[{'PASS' if passed == 200 else 'FAIL'}] random expressions          {passed}/200 within 1e-5")
results.append(passed == 200)

# A known disagreement: relu has a kink at 0, so it has no derivative there.
g, _ = engine_grad(lambda x: x.relu(), [0.0])
print(f"[NOTE] relu at 0: engine grad={g[0]}, numerical grad={numerical_grad(lambda x: x.relu(), [0.0])[0]:.3f}")
print(f"\n{sum(results)}/{len(results)} checks passed")
