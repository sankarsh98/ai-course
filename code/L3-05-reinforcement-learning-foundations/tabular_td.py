"""Model-free tabular RL: Monte Carlo vs TD(0) prediction on a random walk, then
SARSA vs Q-learning control on a cliff-walking grid (after Sutton and Barto, ch. 6)."""
import numpy as np

# ---- 1. Prediction: 5-state random walk A..E, start in C, +1 for exiting right.
TRUE_V = np.arange(1, 6) / 6           # V(A..E) = 1/6 .. 5/6 under the random policy


def random_walk_episode(rng):
    s, states = 2, [2]
    while True:
        s += rng.choice([-1, 1])
        if s < 0 or s > 4:
            return states, float(s > 4)  # reward only on the final transition
        states.append(s)


def run_prediction(method, alpha, episodes, rng):
    V = np.full(5, 0.5)
    errs = []
    for _ in range(episodes):
        states, G = random_walk_episode(rng)
        if method == "MC":                # every-visit constant-alpha MC, gamma = 1
            for s in states:
                V[s] += alpha * (G - V[s])
        else:                             # TD(0): bootstrap from the next state's estimate
            for t, s in enumerate(states):
                target = V[states[t + 1]] if t + 1 < len(states) else G
                V[s] += alpha * (target - V[s])
        errs.append(np.sqrt(np.mean((V - TRUE_V) ** 2)))
    return np.array(errs)


rng = np.random.default_rng(0)
print("1. RMS error of V after N episodes (mean over 100 runs)")
print("   method alpha    N=10    N=50   N=100")
for method, alphas in [("MC", [0.01, 0.03]), ("TD", [0.05, 0.1])]:
    for a in alphas:
        e = np.mean([run_prediction(method, a, 100, rng) for _ in range(100)], 0)
        print(f"   {method:6s} {a:5.2f}  {e[9]:.3f}   {e[49]:.3f}   {e[99]:.3f}")

# ---- 2. Control: 4x12 cliff. Start bottom-left, goal bottom-right, the cells
# between them are a cliff: reward -100 and back to start. Every other step -1.
ROWS, COLS, START, GOAL = 4, 12, (3, 0), (3, 11)
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def cliff_step(s, a):
    r, c = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
    r, c = min(max(r, 0), ROWS - 1), min(max(c, 0), COLS - 1)
    if r == 3 and 0 < c < 11:
        return START, -100.0, False
    return (r, c), -1.0, (r, c) == GOAL


def eps_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(4))
    q = Q[s]
    return int(rng.choice(np.flatnonzero(q == q.max())))   # break ties randomly


def run_control(method, episodes=500, alpha=0.5, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((ROWS, COLS, 4))
    returns = []
    for _ in range(episodes):
        s, total = START, 0.0
        a = eps_greedy(Q, s, eps, rng)
        for _ in range(1000):
            s2, r, done = cliff_step(s, a)
            total += r
            a2 = eps_greedy(Q, s2, eps, rng)
            if method == "SARSA":      # on-policy: the action we will actually take
                target = r + (0 if done else Q[s2][a2])
            else:                      # Q-learning: off-policy, the greedy action
                target = r + (0 if done else Q[s2].max())
            Q[s][a] += alpha * (target - Q[s][a])
            s, a = s2, a2
            if done:
                break
        returns.append(total)
    return Q, np.array(returns)


def greedy_path(Q):
    s, path = START, [START]
    for _ in range(50):
        s, _, done = cliff_step(s, int(Q[s].argmax()))
        path.append(s)
        if done:
            break
    return path


print("\n2. Cliff walking, eps=0.1, alpha=0.5 (20 seeds x 500 episodes)")
for method in ["SARSA", "Q-learning"]:
    runs = [run_control(method, seed=k) for k in range(20)]
    online = np.mean([r[-100:].mean() for _, r in runs])
    path = greedy_path(runs[0][0])
    rows_used = sorted({r for r, _ in path})
    print(f"   {method:10s} return while learning (last 100 eps): {online:7.1f}; "
          f"greedy path length {len(path) - 1}, rows used {rows_used}")
