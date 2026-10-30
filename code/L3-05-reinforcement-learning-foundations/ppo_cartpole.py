"""Actor-critic with GAE and the PPO clipped objective on the NumPy CartPole.
Three settings share all code: one epoch per batch (plain advantage actor-critic),
PPO (10 epochs, clip 0.2), and 10 epochs without the clip."""
import time
import numpy as np
import torch
import torch.nn as nn
from cartpole import VecCartPole, run_policy

torch.set_num_threads(1)
N_ENVS, T, ITERS, GAMMA, LAM = 8, 128, 50, 0.99, 0.95     # 8*128*50 = 51,200 env steps
LR, MINIBATCH, ENT_COEF, VF_COEF = 1e-3, 256, 0.01, 0.5


def mlp(out, scale):
    net = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                        nn.Linear(64, out))
    for layer in net[::2]:                    # orthogonal init, small policy output layer
        nn.init.orthogonal_(layer.weight, np.sqrt(2))
        nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(net[-1].weight, scale)
    return net


def gae(rew, val, next_val, done):
    """Generalized advantage estimation over a (T, N) rollout, backwards in time.
    delta_t = r_t + gamma V(s_{t+1}) (1 - done_t) - V(s_t);  A_t = delta_t + gamma lam (1 - done_t) A_{t+1}"""
    adv, last = np.zeros_like(rew), np.zeros(rew.shape[1])
    for t in reversed(range(len(rew))):
        nv = next_val if t == len(rew) - 1 else val[t + 1]
        delta = rew[t] + GAMMA * nv * (1 - done[t]) - val[t]
        last = delta + GAMMA * LAM * (1 - done[t]) * last
        adv[t] = last
    return adv, adv + val                     # advantages and value targets


def train(epochs, clip, seed):
    torch.manual_seed(seed)
    env = VecCartPole(N_ENVS, seed)
    actor, critic = mlp(2, 0.01), mlp(1, 1.0)
    opt = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=LR, eps=1e-5,
                           foreach=True)
    obs, ep_ret, finished, kls, clipfracs = env.reset(), np.zeros(N_ENVS), [], [], []
    for it in range(ITERS):
        O, A, LP, R, V, D = (np.zeros((T, N_ENVS, 4)), *[np.zeros((T, N_ENVS)) for _ in range(5)])
        for t in range(T):                    # ---- collect a rollout with the current policy
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32)
                dist = torch.distributions.Categorical(logits=actor(o))
                a = dist.sample()
                O[t], A[t], LP[t], V[t] = obs, a.numpy(), dist.log_prob(a).numpy(), critic(o)[:, 0].numpy()
            obs, r, term, trunc, final = env.step(a.numpy())
            if trunc.any():                   # time limit is not failure: bootstrap V(final state)
                with torch.no_grad():
                    r = r + GAMMA * trunc * critic(torch.as_tensor(final, dtype=torch.float32))[:, 0].numpy()
            R[t], D[t] = r, term | trunc
            ep_ret += 1
            finished += list(ep_ret[term | trunc])
            ep_ret[term | trunc] = 0
        with torch.no_grad():
            next_v = critic(torch.as_tensor(obs, dtype=torch.float32))[:, 0].numpy()
        adv, ret = gae(R, V, next_v, D)
        b = [torch.as_tensor(x.reshape(T * N_ENVS, *x.shape[2:]), dtype=torch.float32)
             for x in (O, A, LP, adv, ret)]
        bO, bA, bLP, bAdv, bRet = b
        for _ in range(epochs):               # ---- several passes over the same batch
            perm = torch.randperm(T * N_ENVS)
            for i in range(0, T * N_ENVS, MINIBATCH):
                j = perm[i:i + MINIBATCH]
                dist = torch.distributions.Categorical(logits=actor(bO[j]))
                logratio = dist.log_prob(bA[j]) - bLP[j]
                ratio = logratio.exp()        # pi_new(a|s) / pi_old(a|s)
                a_n = (bAdv[j] - bAdv[j].mean()) / (bAdv[j].std() + 1e-8)
                pg_loss = -torch.min(ratio * a_n, ratio.clamp(1 - clip, 1 + clip) * a_n).mean()
                v_loss = ((critic(bO[j])[:, 0] - bRet[j]) ** 2).mean()
                loss = pg_loss + VF_COEF * v_loss - ENT_COEF * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([*actor.parameters(), *critic.parameters()], 0.5)
                opt.step()
                with torch.no_grad():         # KL(old||new) estimate and fraction clipped
                    kls.append(((ratio - 1) - logratio).mean().item())
                    clipfracs.append(((ratio - 1).abs() > 0.2).float().mean().item())
    with torch.no_grad():
        greedy = run_policy(lambda o: actor(torch.as_tensor(o, dtype=torch.float32)).argmax(1).numpy())
    return np.mean(finished[-20:]), greedy, np.mean(kls), np.max(kls), np.mean(clipfracs)


t0 = time.time()
print("setting                 seed  train return  greedy eval  mean KL   max KL  |r-1|>0.2")
for name, epochs, clip in [("1 epoch, no clip", 1, 1e9), ("PPO 10 epochs clip 0.2", 10, 0.2),
                           ("10 epochs, no clip", 10, 1e9)]:
    for seed in range(3):
        tr, ev, kl, klmax, cf = train(epochs, clip, seed)
        print(f"{name:23s} {seed:4d} {tr:12.1f} {ev:12.1f} {kl:9.4f} {klmax:8.3f} {cf:9.3f}")
print(f"({time.time() - t0:.0f} s total)")
