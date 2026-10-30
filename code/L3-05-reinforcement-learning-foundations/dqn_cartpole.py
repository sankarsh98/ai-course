"""DQN on the NumPy CartPole: an MLP Q-network, experience replay, a target
network and epsilon-greedy exploration. Compares target networks synced every
500 or 100 steps with no target network (bootstrapping from the online net)."""
import time
import numpy as np
import torch
import torch.nn as nn
from cartpole import VecCartPole, run_policy

torch.set_num_threads(1)       # tiny networks: threading overhead costs more than it saves
STEPS, BUFFER, BATCH, GAMMA, LR = 20_000, 20_000, 64, 0.99, 5e-4
EPS_START, EPS_END, EPS_DECAY_STEPS, LEARN_START = 1.0, 0.05, 5_000, 1_000


def make_q():
    return nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                         nn.Linear(128, 2))


def greedy(q):
    return lambda obs: q(torch.as_tensor(obs, dtype=torch.float32)).argmax(1).numpy()


def train(target_sync, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = VecCartPole(1, seed)
    q, q_targ = make_q(), make_q()
    q_targ.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=LR, foreach=True)
    # replay buffer as preallocated arrays: (s, a, r, s', terminated)
    S, S2 = np.zeros((BUFFER, 4), np.float32), np.zeros((BUFFER, 4), np.float32)
    A, R, D = np.zeros(BUFFER, np.int64), np.zeros(BUFFER, np.float32), np.zeros(BUFFER, np.float32)
    obs, evals = env.reset(), []
    for t in range(STEPS):
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * t / EPS_DECAY_STEPS)
        if rng.random() < eps:
            a = int(rng.integers(2))
        else:
            with torch.no_grad():
                a = int(q(torch.as_tensor(obs, dtype=torch.float32)).argmax())
        obs2, r, term, trunc, final = env.step(np.array([a]))
        i = t % BUFFER
        # store the true next state; only true termination stops bootstrapping,
        # a time-limit truncation is not a terminal state of the MDP
        S[i], A[i], R[i], S2[i], D[i] = obs[0], a, r[0], final[0], float(term[0])
        obs = obs2
        if t >= LEARN_START:
            j = rng.integers(0, min(t + 1, BUFFER), size=BATCH)
            s, s2 = torch.from_numpy(S[j]), torch.from_numpy(S2[j])
            with torch.no_grad():
                boot = (q_targ if target_sync else q)(s2).max(1).values
                y = torch.from_numpy(R[j]) + GAMMA * (1 - torch.from_numpy(D[j])) * boot
            q_sa = q(s).gather(1, torch.from_numpy(A[j])[:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(q_sa, y)   # Huber loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()
            if target_sync and t % target_sync == 0:
                q_targ.load_state_dict(q.state_dict())
        if (t + 1) % 4_000 == 0:
            with torch.no_grad():
                evals.append(run_policy(greedy(q), episodes=20, seed=1000 + t))
    with torch.no_grad():
        q0 = q(torch.zeros(1, 4)).max().item()   # Q at the upright start state
    return evals, q0


t0 = time.time()
for name, sync in [("sync every 500", 500), ("sync every 100", 100), ("no target net", 0)]:
    evals, q0 = train(sync, seed=0)
    print(f"{name:16s} greedy return at 4k,8k,..,20k steps {[round(e) for e in evals]}, "
          f"max Q(start) {q0:.4g}")
print(f"({time.time() - t0:.0f} s) Upper bound on any Q here: 1/(1-gamma) = {1 / (1 - GAMMA):.0f}")
