"""Dynamic programming on a small gridworld MDP: policy evaluation, value
iteration and policy iteration, all from the model P[s, a, s'] and R[s, a]."""
import numpy as np

LAYOUT = ["......",
          ".####.",
          "S....G",
          "PPPPPP"]            # S start, G goal, P pit, # wall
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # up, right, down, left
ARROWS = "^>v<"
STEP_R, PIT_R = -1.0, -50.0


def build_mdp(slip):
    """With prob 1-slip move as intended; with prob slip/2 each, move to one side."""
    rows, cols = len(LAYOUT), len(LAYOUT[0])
    cells = [(r, c) for r in range(rows) for c in range(cols) if LAYOUT[r][c] != "#"]
    idx = {rc: i for i, rc in enumerate(cells)}
    n, m = len(cells), len(ACTIONS)
    P, R = np.zeros((n, m, n)), np.zeros((n, m))
    terminal = np.array([LAYOUT[r][c] in "GP" for r, c in cells])
    for s, (r, c) in enumerate(cells):
        if terminal[s]:
            P[s, :, s] = 1.0              # absorbing, reward 0
            continue
        for a in range(m):
            for da, p in [(0, 1 - slip), (1, slip / 2), (-1, slip / 2)]:
                dr, dc = ACTIONS[(a + da) % m]
                r2, c2 = r + dr, c + dc
                if (r2, c2) not in idx:   # wall or edge: stay put
                    r2, c2 = r, c
                s2 = idx[(r2, c2)]
                P[s, a, s2] += p
                R[s, a] += p * (PIT_R if LAYOUT[r2][c2] == "P" else STEP_R)
    return P, R, cells, idx


def evaluate(P, R, pi, gamma, tol=1e-10):
    """Iterative policy evaluation: V <- R_pi + gamma P_pi V. pi is (n, m) probs."""
    R_pi, P_pi = (pi * R).sum(1), np.einsum("sa,sat->st", pi, P)
    V, sweeps = np.zeros(len(R)), 0
    while True:
        V_new = R_pi + gamma * P_pi @ V
        sweeps += 1
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, sweeps, R_pi, P_pi
        V = V_new


def value_iteration(P, R, gamma, tol=1e-10):
    V, it = np.zeros(len(R)), 0
    while True:
        Q = R + gamma * P @ V            # (n, m): one-step lookahead
        V_new, it = Q.max(1), it + 1
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, Q.argmax(1), it
        V = V_new


def policy_iteration(P, R, gamma):
    n, m = R.shape
    greedy, rounds = np.zeros(n, dtype=int), 0
    while True:
        pi = np.eye(m)[greedy]
        V, _, _, _ = evaluate(P, R, pi, gamma)
        new = (R + gamma * P @ V).argmax(1)
        rounds += 1
        if np.array_equal(new, greedy):
            return V, greedy, rounds
        greedy = new


def show(values, cells, fmt):
    rows, cols = len(LAYOUT), len(LAYOUT[0])
    grid = [["  ##  " for _ in range(cols)] for _ in range(rows)]
    for s, (r, c) in enumerate(cells):
        grid[r][c] = fmt(s, r, c)
    print("\n".join("".join(row) for row in grid))


gamma = 0.99
P, R, cells, idx = build_mdp(slip=0.0)
n, m = R.shape
start = idx[(2, 0)]

print("1. Uniform random policy, slip=0, gamma=0.99")
uniform = np.full((n, m), 1 / m)
V_it, sweeps, R_pi, P_pi = evaluate(P, R, uniform, gamma)
V_exact = np.linalg.solve(np.eye(n) - gamma * P_pi, R_pi)
print(f"   iterative evaluation: {sweeps} sweeps, max |V_it - V_exact| = "
      f"{np.max(np.abs(V_it - V_exact)):.1e}")
print(f"   V_random(start) = {V_exact[start]:.2f}")

for slip in [0.0, 0.2]:
    P, R, cells, idx = build_mdp(slip)
    V, greedy, it = value_iteration(P, R, gamma)
    V2, greedy2, rounds = policy_iteration(P, R, gamma)
    print(f"\n2. slip={slip}: value iteration {it} sweeps, policy iteration "
          f"{rounds} improvement rounds, same policy: {np.array_equal(greedy, greedy2)}")
    print(f"   V*(start) = {V[start]:.2f}")
    show(V, cells, lambda s, r, c: f"{V[s]:6.1f}" if LAYOUT[r][c] not in "GP"
         else f"{LAYOUT[r][c]:>6}")
    show(greedy, cells, lambda s, r, c: f"{ARROWS[greedy[s]]:>6}"
         if LAYOUT[r][c] not in "GP" else f"{LAYOUT[r][c]:>6}")

print("\n3. Bellman residual of V* (should be ~0):",
      f"{np.max(np.abs(V - (R + gamma * P @ V).max(1))):.1e}")
print("4. Discount sets the horizon: steps until gamma^k < 0.01")
for g in [0.5, 0.9, 0.99]:
    print(f"   gamma={g}: {int(np.ceil(np.log(0.01) / np.log(g)))} steps")
