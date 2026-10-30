# Reinforcement Learning Foundations

> **Level 3 · Module 5** · Prerequisites: L0-04, L0-05, L2-01, L2-02 · Time: 8 to 10 hours · Code: `code/L3-05-reinforcement-learning-foundations/`

## What you will be able to do

- Formulate a decision problem as a Markov decision process and write its Bellman equations.
- Solve small MDPs with policy evaluation, value iteration and policy iteration, and explain why they converge.
- Implement Monte Carlo, TD(0), SARSA and Q-learning, and predict how on-policy and off-policy learning behave.
- Derive the policy gradient, show why baselines cut variance without adding bias, and compute GAE advantages.
- Implement DQN and PPO from scratch and read their diagnostics (Q-value scale, KL, clip fraction).
- Judge RL results: seed variance, reward misspecification, and evaluations that hide instability.

## 1. The reinforcement learning problem

In supervised learning (L1-01) every input comes with the right answer. In **reinforcement learning (RL)** an **agent** acts in an **environment** and receives a scalar **reward**. Nobody says which action was right, and the reward may arrive many steps after the action that earned it. Learning from evaluation rather than instruction, with delayed credit, is what makes RL its own subject.

The standard model is the **Markov decision process (MDP)**: states $\mathcal S$, actions $\mathcal A$, transition probabilities $P(s' \mid s, a)$, expected reward $R(s, a)$ and a **discount factor** $\gamma \in [0, 1)$. At step $t$ the agent sees $S_t$, takes $A_t$, receives $R_{t+1}$ and moves to $S_{t+1}$. "Markov" means the next state and reward depend only on the current state and action. A **policy** $\pi(a \mid s)$ chooses actions, and the goal is to maximize the expected **return**

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}.$$

The discount keeps sums finite and sets a horizon of roughly $1/(1-\gamma)$ steps. Script 1 prints how long it takes $\gamma^k$ to fall below 0.01: 7 steps for $\gamma = 0.5$, 44 for 0.9, 459 for 0.99. Choosing $\gamma$ is part of defining the problem.

In CartPole (scripts 4 to 6) the state is the cart's position and velocity and the pole's angle and angular velocity, the actions are "push left" and "push right", and the reward is +1 per step the pole stays up. For a language model in post-training (L3-02), the state is the prompt plus the tokens so far, the action is the next token, the policy is the model, and the reward often arrives once, when the finished answer is checked. The same algorithms cover both.

## 2. Value functions and the Bellman equations

The **state-value** and **action-value** functions of a policy are

$$V^\pi(s) = \mathbb E_\pi[G_t \mid S_t = s], \qquad Q^\pi(s, a) = \mathbb E_\pi[G_t \mid S_t = s, A_t = a].$$

Since $G_t = R_{t+1} + \gamma G_{t+1}$, a state's value is the expected immediate reward plus the discounted value of where you land, the **Bellman expectation equation**:

$$V^\pi(s) = \sum_a \pi(a \mid s)\Big[R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^\pi(s')\Big].$$

In vector form, with $R_\pi$ the expected one-step rewards and $P_\pi$ the state-to-state transition matrix under $\pi$, this is linear: $V^\pi = R_\pi + \gamma P_\pi V^\pi$, so $V^\pi = (I - \gamma P_\pi)^{-1} R_\pi$. The optimal values satisfy the **Bellman optimality equation**, with a max instead of the policy average:

$$V^*(s) = \max_a \Big[R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^*(s')\Big], \qquad Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q^*(s', a').$$

With $Q^*$ in hand, acting optimally needs no lookahead: take $\arg\max_a Q^*(s, a)$. A finite MDP always has a deterministic optimal policy [Sutton & Barto, 2018, ch. 3].

## 3. Dynamic programming: planning with a known model

With $P$ and $R$ known, define the **Bellman operator** $(\mathcal T V)(s) = \max_a [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s')]$. It is a **contraction**: $\|\mathcal T V - \mathcal T U\|_\infty \le \gamma \|V - U\|_\infty$. So it has one fixed point, $V^*$, and repeated application from any start converges to it, the error shrinking by at least $\gamma$ per sweep. Three algorithms follow:

- **Policy evaluation** iterates $V \leftarrow R_\pi + \gamma P_\pi V$.
- **Value iteration** iterates $\mathcal T$, then acts greedily.
- **Policy iteration** alternates full evaluation with greedy improvement. The **policy improvement theorem** guarantees each new policy is at least as good everywhere, so it terminates, usually in a few rounds.

Script 1 builds a 4-by-6 gridworld: walls, a row of pits (reward $-50$, episode ends), a goal, $-1$ per step, $\gamma = 0.99$. With probability `slip` a move goes to one side of the intended direction. Evaluating the random policy takes 648 sweeps and matches the linear solve to $2.6 \times 10^{-9}$; its start value is $-52.36$. With `slip=0`, the optimal path runs along the pits in five steps ($V^* = -4.90$). With `slip=0.2`, each step there risks a 10% slip into a pit, and the optimal policy detours through the top row ($V^* = -11.59$): the model changed the answer, not just the numbers. Value iteration now needs 45 sweeps instead of 8, because with slips the values approach $V^*$ geometrically instead of settling exactly; policy iteration needs 7 rounds. DP needs the full model and touches every state, so the rest of the module learns from samples.

## 4. Monte Carlo and temporal-difference learning

**Monte Carlo (MC)** estimation averages observed returns: $V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]$. The target is unbiased (L0-04) but noisy, since it sums every random reward to the end, and it needs complete episodes. **Temporal-difference (TD)** learning uses one real reward plus the current estimate of the next state:

$$V(S_t) \leftarrow V(S_t) + \alpha\,\delta_t, \qquad \delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t).$$

Updating an estimate from an estimate is **bootstrapping**. The **TD error** $\delta_t$ is biased while $V$ is wrong, but it depends on a single random step, so it has much lower variance, and TD can learn online and in tasks that never end.

Script 2 reproduces the five-state random walk from Sutton and Barto's chapter 6 (true values $1/6, \dots, 5/6$). After 100 episodes, averaged over 100 runs, MC reaches an RMS error of about 0.095 and TD(0) with $\alpha = 0.05$ reaches 0.037. TD with $\alpha = 0.1$ is faster early (0.128 after 10 episodes) but has a higher noise floor (0.057), the step-size trade-off of constant-rate SGD from L0-05. Between the two extremes lie **$n$-step returns** ($n$ real rewards, then bootstrap) and the **$\lambda$-return**, a weighted average of them: $\lambda = 0$ is TD(0) and $\lambda = 1$ is MC. GAE in section 8 is the same idea.

## 5. Control: SARSA, Q-learning and exploration

To improve the policy without a model, learn action values and act mostly greedily. Two TD control methods differ in one term:

$$\text{SARSA:}\ Q(s,a) \mathrel{+}= \alpha\,[r + \gamma Q(s', a') - Q(s,a)], \qquad \text{Q-learning:}\ Q(s,a) \mathrel{+}= \alpha\,[r + \gamma \max_{b} Q(s', b) - Q(s,a)].$$

SARSA uses the action $a'$ it will actually take, so it learns the value of its own exploring policy (**on-policy**). Q-learning uses the greedy action regardless of behavior, so it learns $Q^*$ (**off-policy**); tabular Q-learning converges to $Q^*$ if all state-action pairs keep being tried and step sizes decay suitably [Watkins & Dayan, 1992]. Keeping them tried is the **exploration** problem. The simplest answer is **$\epsilon$-greedy** (random action with probability $\epsilon$); others are optimistic initial values, uncertainty bonuses, and entropy bonuses on stochastic policies. Exploration with rare rewards in large spaces remains unsolved in general.

On script 2's cliff-walking grid (also Sutton and Barto, chapter 6), the shortest path runs along a cliff that costs $-100$. With $\epsilon = 0.1$, Q-learning learns the optimal edge path (13 steps) but keeps falling off while exploring, averaging $-50.5$ per episode. SARSA learns a safer path along the top (17 steps) and averages $-26.6$. Q-learning finds the better policy; SARSA does better while learning.

## 6. Function approximation and DQN

Tables do not scale to continuous or huge state spaces, so approximate $Q_\theta(s, a)$ with a neural network (L2-01, L2-02) and regress it toward TD targets. Naively this is unstable. Sutton and Barto call the combination of **function approximation, bootstrapping and off-policy learning** the **deadly triad** [Sutton & Barto, 2018]: fitting one target moves the targets of other states, and estimates can diverge.

The **deep Q-network (DQN)** learned 49 Atari games from screen pixels with the same architecture and hyperparameters for every game [Mnih et al., 2015]. Two ingredients made it work. **Experience replay** stores transitions $(s, a, r, s', d)$ and trains on random minibatches, which breaks the correlation between consecutive steps and reuses data. A **target network** $Q_{\theta^-}$, a copy synced every $C$ steps, computes the labels:

$$L(\theta) = \mathbb E\Big[\ell\big(r + \gamma (1 - d) \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a)\big)\Big],$$

with $d = 1$ at terminal states and $\ell$ the squared or Huber loss. Between syncs this is ordinary regression with fixed labels.

Script 5 trains DQN on CartPole for 20,000 steps three ways. A sanity bound: with reward 1 per step and $\gamma = 0.99$, no Q-value can exceed $1/(1-\gamma) = 100$.

- **No target network:** Q-values explode to about $7.9 \times 10^6$ and the policy falls in 9 steps. The deadly triad, live.
- **Sync every 500 steps:** stable, but $Q(\text{start}) = 34.0$ and returns stall at 100 to 200. Each sync period acts roughly like one sweep of value iteration, and 38 sweeps can propagate at most $\sum_{k<38} 0.99^k \approx 31.7$.
- **Sync every 100 steps:** $Q(\text{start}) = 91.7$, and the greedy policy balances for 500 steps from 16,000 steps on.

While preparing the module (not in the script), syncing every 50 steps gave $Q(\text{start})$ of 107 to 135 over 6 runs, above the bound, so certainly wrong. One known cause is **overestimation bias**: a max over noisy estimates is biased upward, which Double DQN reduces by choosing the next action with the online network and evaluating it with the target network [van Hasselt et al., 2016]. In my test Double DQN did not fix this case (103 to 120 over 3 seeds), so a target refreshed every 50 steps is probably the bigger problem. These are few-seed results on a toy, and DQN is sensitive to its settings (exercise 6).

## 7. Policy gradients

**Policy gradient** methods parameterize the policy directly, for discrete actions as a network producing softmax logits, and ascend $J(\theta) = \mathbb E_{\tau \sim \pi_\theta}[R(\tau)]$, where $\tau$ is a trajectory and $R(\tau)$ its return. They need no $\max_a$, so they handle huge or continuous action spaces, and they change the policy smoothly. L3-02 builds on this section.

**Derivation.** A trajectory has probability $p_\theta(\tau) = \rho(s_0) \prod_t \pi_\theta(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)$. With the log-derivative trick $\nabla_\theta p_\theta = p_\theta \nabla_\theta \log p_\theta$,

$$\nabla_\theta J = \sum_\tau R(\tau)\, \nabla_\theta p_\theta(\tau) = \mathbb E_{\tau}\big[R(\tau)\, \nabla_\theta \log p_\theta(\tau)\big] = \mathbb E_{\tau}\Big[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, R(\tau)\Big],$$

because the initial-state and transition terms of $\log p_\theta$ do not depend on $\theta$. You can estimate the gradient from samples without a model of the environment. This is the **score-function** or **REINFORCE** estimator [Williams, 1992]. In code, minimize the surrogate $-\sum_t \log \pi_\theta(a_t|s_t)\, \hat R_t$ with $\hat R_t$ held constant. Only its gradient means anything; its value does not track progress.

**Variance reduction without bias.** (1) *Reward-to-go:* an action cannot affect earlier rewards, whose terms have zero mean, so use $G_t$, the return from step $t$, instead of $R(\tau)$. (2) *Baselines:* for any $b(s)$ that does not depend on the action,

$$\mathbb E_{a \sim \pi}\big[\nabla_\theta \log \pi_\theta(a|s)\, b(s)\big] = b(s) \sum_a \nabla_\theta \pi_\theta(a|s) = b(s)\, \nabla_\theta \sum_a \pi_\theta(a|s) = b(s)\, \nabla_\theta 1 = 0.$$

The usual choice $b(s) = V^\pi(s)$ turns $G_t - b(s_t)$ into an estimate of the **advantage** $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$, how much better the action was than the policy's average.

Script 3 checks this on a three-armed bandit with a softmax policy, where the exact gradient is $\partial J / \partial \theta_k = \pi_k (r_k - J)$ (one state, so the trajectory is one action). With 400,000 samples each:

| rewards | baseline | mean / exact | total variance |
|---|---|---|---|
| means 1, 2, 3 | none | 0.997 | 3.26 |
| means 1, 2, 3 | true $J$ | 1.006 | 0.86 |
| shifted by +100 | none | 1.075 | 6393 |
| shifted by +100 | true $J$ | 1.002 | 0.87 |

All are unbiased (the 1.075 is noise from the huge variance), but adding 100 to every reward multiplies the no-baseline variance by about 2,000, and the baseline removes it. In 200 steps of gradient ascent on the shifted rewards, the no-baseline runs end with a mean probability of 0.35 on the best arm (one seed at 0.000); with a baseline every seed exceeds 0.96.

**Group baselines.** If you can sample $G$ actions (or, for a language model, $G$ answers) for the same state, their mean reward is a baseline that needs no value network. If the mean includes the sample itself, the estimator is scaled by $(G-1)/G$: script 3 measures 0.871 against the predicted $7/8 = 0.875$ for $G = 8$. The leave-one-out mean is unbiased (0.996). The shrinkage acts like a smaller learning rate, so it is mostly harmless, but you will meet it again in GRPO, the group-baseline variant of PPO introduced in the DeepSeekMath paper (Shao et al., 2024, arXiv:2402.03300) and covered in L3-02.

## 8. Actor-critic and generalized advantage estimation

An **actor-critic** learns a policy (actor) and a value function $V_\phi$ (critic). The one-step advantage estimate $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ has low variance but is biased when $V_\phi$ is wrong; $G_t - V_\phi(s_t)$ is unbiased but noisy. **Generalized advantage estimation (GAE)** interpolates [Schulman et al., 2016]:

$$\hat A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l\, \delta_{t+l}, \qquad \text{computed backward as}\quad \hat A_t = \delta_t + \gamma \lambda\, (1 - d_t)\, \hat A_{t+1}.$$

$\lambda = 0$ gives the TD error, $\lambda = 1$ the MC advantage, and 0.95 is a common default. The critic regresses on $\hat A_t + V_\phi(s_t)$.

A detail that matters: CartPole-v1 cuts episodes at 500 steps. That is **truncation**, not termination; the pole is still up. Treating it as terminal teaches the critic that late states are worth nothing, which is wrong because time is not in the observation. The scripts keep the two apart, as Gymnasium's API does, and bootstrap from $V(s_{\text{final}})$ on truncation.

## 9. PPO: taking big steps safely

Plain policy gradient uses each batch once, because the formula assumes the data came from the current policy. Reusing the batch for big or repeated steps is risky: a small change in parameters can change action probabilities a lot, and a collapsed policy collects bad data. **Trust region policy optimization (TRPO)** bounds the KL divergence (L0-04) between old and new policy with a second-order constrained step [Schulman et al., 2015]. **Proximal policy optimization (PPO)** gets most of the benefit with first-order SGD and a clipped objective [Schulman et al., 2017].

**The ratio.** For data collected by $\pi_{\text{old}}$, let

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}.$$

This is an importance weight (L0-04): $\hat{\mathbb E}[r_t(\theta) \hat A_t]$ scores the new policy with old samples. At $\theta = \theta_{\text{old}}$, $r_t = 1$ and its gradient is exactly the policy gradient $\hat A_t \nabla \log \pi_\theta$. Far from $\theta_{\text{old}}$ the estimate degrades.

**The clipped surrogate.**

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb E}_t\Big[\min\big(r_t(\theta)\, \hat A_t,\ \operatorname{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)\, \hat A_t\big)\Big].$$

Script 3, part 3 tabulates it for $\epsilon = 0.2$. For a good action ($\hat A > 0$) the objective rewards raising its probability until $r = 1.2$, after which the gradient is zero. For a bad action it rewards lowering it until $r = 0.8$. The min makes it pessimistic: clipping never removes a penalty, so if an update pushed a bad action's ratio to 3, the full gradient is there to undo it. Each sample stops pulling once the policy has moved far enough its way.

**The algorithm.** Run $N$ environments for $T$ steps with $\pi_{\text{old}}$, compute GAE, then for $K$ epochs take Adam steps on shuffled minibatches of

$$\mathcal L(\theta, \phi) = -L^{\text{CLIP}}(\theta) + c_1\, \hat{\mathbb E}_t\big[(V_\phi(s_t) - \hat V_t^{\text{targ}})^2\big] - c_2\, \hat{\mathbb E}_t\big[\mathcal H(\pi_\theta(\cdot \mid s_t))\big],$$

with value-loss weight $c_1$ (0.5 here) and entropy bonus $c_2$ (0.01 here) against premature determinism. In the PPO paper's continuous-control comparison, $\epsilon = 0.2$ scored best of the clip values tried (average normalized score 0.82) and no clipping or penalty scored $-0.39$; an adaptive KL-penalty variant scored slightly below clipping [Schulman et al., 2017].

**Diagnostics.** Log the **approximate KL** $\hat{\mathbb E}[(r - 1) - \log r]$, an unbiased, non-negative estimator of $\mathrm{KL}(\pi_{\text{old}} \| \pi_\theta)$ (John Schulman, "Approximating KL Divergence", 2020, http://joschu.net/blog/kl-approx.html), and the **clip fraction**, the share of samples with $|r - 1| > \epsilon$. Script 6 runs three settings for 51,200 steps and 3 seeds:

| setting | training return (last 20 episodes) | mean KL | max KL | clip fraction |
|---|---|---|---|---|
| 1 epoch, no clip (plain actor-critic) | 212 to 243 | 0.0001 | 0.001 | 0.000 |
| PPO: 10 epochs, clip 0.2 | 500 on all seeds | 0.003 to 0.004 | 0.018 to 0.061 | 0.03 to 0.04 |
| 10 epochs, no clip | 126, 292, 477 | 0.11 to 0.13 | 2.3 to 16.7 | 0.18 to 0.25 |

Ten epochs on the same data is what makes PPO fast. Without the clip, the same reuse produces single updates with KL up to 16.7 (the policy changes almost completely) and erratic returns. Yet all three unclipped runs end with a greedy policy that balances for 500 steps: CartPole is forgiving, and one final evaluation can hide an unstable run. Read the KL and the learning curve, not only the final score.

**Implementation details matter.** Huang et al. list 37 details of reference PPO implementations, among them per-minibatch advantage normalization, orthogonal initialization with a small policy output layer, Adam $\epsilon = 10^{-5}$, gradient-norm clipping at 0.5, and value-loss clipping, which they note other studies found unhelpful [Huang et al., 2022]. Script 6 uses the first four. Check these first when a reimplementation underperforms.

**Bridge to L3-02.** For a language model an episode is one response, an action is a token, and the dynamics are deterministic. RLHF with PPO adds a learned reward model and a KL penalty toward a reference model; GRPO-style methods drop the critic for a group baseline. As of October 2026, a survey of RL for LLM post-training (Wang et al., arXiv:2407.16216, revised May 2026) organizes the field around PPO- and GRPO-based methods. The ratio, clip, advantage and KL above are the machinery.

## 10. What makes RL hard

- **Credit assignment:** one reward after 500 steps must be split among 500 decisions.
- **Exploration:** the data depend on the policy, which depends on the data.
- **Moving targets:** bootstrapped targets shift as estimates change, hence target networks and trust regions.
- **Sample cost:** even CartPole took tens of thousands of steps. Irpan's essay collects examples of how many more realistic tasks need [Irpan, 2018].
- **Reward misspecification:** the agent optimizes the reward you wrote, such as the boat-racing agent in Irpan's essay that circles collecting power-ups instead of finishing. With learned rewards this becomes reward hacking (L3-02, L4-05).
- **Seed and implementation variance:** Henderson et al. showed that seeds, hyperparameters and codebases change reported deep RL results substantially, so claimed improvements need significance tests and careful reporting [Henderson et al., 2018]. Every single-seed result here is an anecdote.

## Hands-on

The scripts in `code/L3-05-reinforcement-learning-foundations/` need only NumPy and PyTorch (CPU). Gymnasium is not installed in the course environment, and I did not add it: `cartpole.py` reimplements CartPole-v1's dynamics, constants and episode rules from Gymnasium's source (`gymnasium/envs/classic_control/cartpole.py`, which credits the problem to Barto, Sutton and Anderson, "Neuronlike adaptive elements that can solve difficult learning control problems", 1983). To check it, I installed Gymnasium 1.4.0 into a temporary directory outside the shared environment and stepped both with the same actions for 50 episodes (23,210 steps). Every termination and truncation flag matched, and states agreed within $2.3 \times 10^{-7}$ (Gymnasium uses float32).

Outputs are from one run on a 24-core Linux machine, PyTorch 2.15 CPU, fixed seeds, one PyTorch thread. The DQN and PPO scripts took 50 s and 42 s with other jobs running; under heavier load they took longer. Other platforms may give different digits but the same patterns.

### Script 1: dynamic programming on a gridworld

```python
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
```

```
1. Uniform random policy, slip=0, gamma=0.99
   iterative evaluation: 648 sweeps, max |V_it - V_exact| = 2.6e-09
   V_random(start) = -52.36

2. slip=0.0: value iteration 8 sweeps, policy iteration 8 improvement rounds, same policy: True
   V*(start) = -4.90
  -6.8  -5.9  -4.9  -3.9  -3.0  -2.0
  -5.9  ##    ##    ##    ##    -1.0
  -4.9  -3.9  -3.0  -2.0  -1.0     G
     P     P     P     P     P     P
     >     >     >     >     >     v
     v  ##    ##    ##    ##       v
     >     >     >     >     >     G
     P     P     P     P     P     P

2. slip=0.2: value iteration 45 sweeps, policy iteration 7 improvement rounds, same policy: True
   V*(start) = -11.59
  -8.7  -7.4  -6.2  -5.0  -3.8  -2.6
  -9.8  ##    ##    ##    ##    -1.2
 -11.6 -16.7 -17.4 -12.3  -6.5     G
     P     P     P     P     P     P
     >     >     >     >     >     v
     ^  ##    ##    ##    ##       v
     ^     <     >     >     >     G
     P     P     P     P     P     P

3. Bellman residual of V* (should be ~0): 2.4e-11
4. Discount sets the horizon: steps until gamma^k < 0.01
   gamma=0.5: 7 steps
   gamma=0.9: 44 steps
   gamma=0.99: 459 steps
```

With `slip=0.2`, the cell next to the start points back left, accepting a 10% chance of slipping into a pit, and the agent then climbs the left column, where slips only push it sideways.

### Script 2: Monte Carlo, TD, SARSA and Q-learning

```python
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
```

```
1. RMS error of V after N episodes (mean over 100 runs)
   method alpha    N=10    N=50   N=100
   MC      0.01  0.211   0.144   0.097
   MC      0.03  0.188   0.104   0.095
   TD      0.05  0.175   0.054   0.037
   TD      0.10  0.128   0.051   0.057

2. Cliff walking, eps=0.1, alpha=0.5 (20 seeds x 500 episodes)
   SARSA      return while learning (last 100 eps):   -26.6; greedy path length 17, rows used [0, 1, 2, 3]
   Q-learning return while learning (last 100 eps):   -50.5; greedy path length 13, rows used [2, 3]
```

### Script 3: the policy-gradient estimator, baselines and the PPO clip

```python
"""The score-function (REINFORCE) gradient on a 3-armed bandit with a softmax
policy, checked against the exact gradient; baselines; and the PPO clip."""
import numpy as np

rng = np.random.default_rng(0)
theta = np.array([0.5, 0.0, -0.5])        # policy logits
mean_r = np.array([1.0, 2.0, 3.0])        # expected reward of each arm
NOISE = 1.0                               # reward = mean_r[a] + N(0, NOISE^2)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


pi = softmax(theta)
J = pi @ mean_r
exact = pi * (mean_r - J)                 # dJ/dtheta_k = pi_k (r_k - J) for softmax
print(f"pi = {np.round(pi, 3)}, J = {J:.3f}, exact grad = {np.round(exact, 4)}")


def grad_samples(n, offset, baseline, G=8):
    """n single-sample gradient estimates (R - b) * grad log pi(a).
    Samples come in groups of G so that group baselines can be formed."""
    a = rng.choice(3, size=(n // G, G), p=pi)
    R = mean_r[a] + offset + NOISE * rng.normal(size=a.shape)
    if baseline == "none":
        b = 0.0
    elif baseline == "true J":
        b = J + offset
    elif baseline == "group mean":       # includes the sample itself (GRPO-style)
        b = R.mean(1, keepdims=True)
    elif baseline == "leave-one-out":    # mean of the other G-1 samples
        b = (R.sum(1, keepdims=True) - R) / (G - 1)
    score = np.eye(3)[a] - pi             # grad log softmax = onehot(a) - pi
    return ((R - b)[..., None] * score).reshape(-1, 3)


print("\n1. Single-sample estimators, 400k samples each")
print("   offset baseline        mean estimate                 bias/exact  total var")
for offset in [0.0, 100.0]:
    for baseline in ["none", "true J", "group mean", "leave-one-out"]:
        g = grad_samples(400_000, offset, baseline)
        m = g.mean(0)
        ratio = (m @ exact) / (exact @ exact)   # 1.0 means unbiased
        print(f"   {offset:6.0f} {baseline:14s} {np.array2string(m, precision=4):28s}"
              f"  {ratio:6.3f}   {g.var(0).sum():10.2f}")

print("\n2. Gradient-ascent steps with batch 64, lr 0.1, offset 100: P(best arm) after 200 steps")
for baseline in ["none", "leave-one-out"]:
    finals = []
    for seed in range(20):
        th = np.zeros(3)
        loc = np.random.default_rng(seed)
        for _ in range(200):
            p = softmax(th)
            a = loc.choice(3, size=(8, 8), p=p)
            R = mean_r[a] + 100 + NOISE * loc.normal(size=a.shape)
            b = 0.0 if baseline == "none" else (R.sum(1, keepdims=True) - R) / 7
            th += 0.1 * ((R - b)[..., None] * (np.eye(3)[a] - p)).reshape(-1, 3).mean(0)
        finals.append(softmax(th)[2])
    print(f"   {baseline:14s} mean {np.mean(finals):.3f}, worst seed {np.min(finals):.3f}")

print("\n3. PPO clipped objective L = min(r A, clip(r, 0.8, 1.2) A) and dL/dr")
print("   ratio r   A=+1: L, dL/dr     A=-1: L, dL/dr")
for r in [0.5, 0.8, 1.0, 1.1, 1.2, 1.5, 3.0]:
    row = []
    for A in [1.0, -1.0]:
        unclipped, clipped = r * A, np.clip(r, 0.8, 1.2) * A
        L = min(unclipped, clipped)
        dL = A if unclipped <= clipped else 0.0   # gradient flows only through r*A
        row.append(f"{L:6.2f}, {dL:4.1f}")
    print(f"   {r:5.2f}     {row[0]}       {row[1]}")
```

```
pi = [0.506 0.307 0.186], J = 1.680, exact grad = [-0.3443  0.0984  0.246 ]

1. Single-sample estimators, 400k samples each
   offset baseline        mean estimate                 bias/exact  total var
        0 none           [-0.3434  0.0989  0.2445]      0.997         3.26
        0 true J         [-0.3465  0.0997  0.2468]      1.006         0.86
        0 group mean     [-0.2998  0.0856  0.2142]      0.871         0.77
        0 leave-one-out  [-0.3436  0.0999  0.2437]      0.996         1.00
      100 none           [-0.3975  0.2146  0.183 ]      1.075      6393.48
      100 true J         [-0.345   0.0992  0.2459]      1.002         0.87
      100 group mean     [-0.3004  0.0857  0.2147]      0.873         0.77
      100 leave-one-out  [-0.3437  0.0974  0.2463]      0.999         1.00

2. Gradient-ascent steps with batch 64, lr 0.1, offset 100: P(best arm) after 200 steps
   none           mean 0.348, worst seed 0.000
   leave-one-out  mean 0.968, worst seed 0.963

3. PPO clipped objective L = min(r A, clip(r, 0.8, 1.2) A) and dL/dr
   ratio r   A=+1: L, dL/dr     A=-1: L, dL/dr
    0.50       0.50,  1.0        -0.80,  0.0
    0.80       0.80,  1.0        -0.80, -1.0
    1.00       1.00,  1.0        -1.00, -1.0
    1.10       1.10,  1.0        -1.10, -1.0
    1.20       1.20,  1.0        -1.20, -1.0
    1.50       1.20,  0.0        -1.50, -1.0
    3.00       1.20,  0.0        -3.00, -1.0
```

### Script 4: CartPole in NumPy

```python
"""A vectorized CartPole in NumPy, following the dynamics, constants and episode
rules of Gymnasium's CartPole-v1 (Barto, Sutton and Anderson, 1983).
Gymnasium is not needed. Import VecCartPole, or run this file for a sanity check."""
import numpy as np

GRAVITY, M_CART, M_POLE, HALF_LEN, FORCE, TAU = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
TOTAL_M, POLE_ML = M_CART + M_POLE, M_POLE * HALF_LEN
THETA_MAX, X_MAX, MAX_STEPS = 12 * 2 * np.pi / 360, 2.4, 500


class VecCartPole:
    """n independent cart-poles stepped together. Observation: (x, x_dot, theta,
    theta_dot). Actions: 0 push left, 1 push right. Reward +1 per step."""

    def __init__(self, n, seed=0):
        self.n, self.rng = n, np.random.default_rng(seed)
        self.state = np.zeros((n, 4))
        self.t = np.zeros(n, dtype=int)
        self.reset_where(np.ones(n, dtype=bool))

    def reset_where(self, mask):
        self.state[mask] = self.rng.uniform(-0.05, 0.05, size=(mask.sum(), 4))
        self.t[mask] = 0

    def reset(self):
        self.reset_where(np.ones(self.n, dtype=bool))
        return self.state.copy()

    def step(self, action):
        """Returns obs, reward, terminated, truncated, final_obs. Finished envs are
        reset automatically: obs is then the new start state and final_obs holds
        the state the episode actually ended in (needed for correct bootstrapping)."""
        x, x_dot, th, th_dot = self.state.T
        force = np.where(action == 1, FORCE, -FORCE)
        cos, sin = np.cos(th), np.sin(th)
        temp = (force + POLE_ML * th_dot ** 2 * sin) / TOTAL_M
        th_acc = (GRAVITY * sin - cos * temp) / (
            HALF_LEN * (4.0 / 3.0 - M_POLE * cos ** 2 / TOTAL_M))
        x_acc = temp - POLE_ML * th_acc * cos / TOTAL_M
        # explicit Euler, positions first with the old velocities (as in Gymnasium)
        self.state = np.stack([x + TAU * x_dot, x_dot + TAU * x_acc,
                               th + TAU * th_dot, th_dot + TAU * th_acc], axis=1)
        self.t += 1
        terminated = (np.abs(self.state[:, 0]) > X_MAX) | (np.abs(self.state[:, 2]) > THETA_MAX)
        truncated = (self.t >= MAX_STEPS) & ~terminated
        final_obs = self.state.copy()
        self.reset_where(terminated | truncated)
        return self.state.copy(), np.ones(self.n), terminated, truncated, final_obs


def run_policy(policy, episodes=100, seed=123):
    """Average undiscounted return of policy(obs_batch) -> actions over full episodes."""
    env = VecCartPole(episodes, seed)
    obs, alive, ret = env.reset(), np.ones(episodes, dtype=bool), np.zeros(episodes)
    while alive.any():
        obs, r, term, trunc, _ = env.step(policy(obs))
        ret += r * alive
        alive &= ~(term | trunc)
    return ret.mean()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("random policy        :", run_policy(lambda o: rng.integers(0, 2, len(o))))
    print("push toward the lean :", run_policy(lambda o: (o[:, 2] > 0).astype(int)))
    print("lean + angular vel.  :", run_policy(lambda o: (o[:, 2] + 0.5 * o[:, 3] > 0).astype(int)))
```

```
random policy        : 21.58
push toward the lean : 42.02
lean + angular vel.  : 500.0
```

A random policy lasts about 22 steps and "push toward the lean" 42, but a hand-written controller that also uses angular velocity balances for all 500. CartPole tests whether your RL code works, not whether RL is needed.

### Script 5: DQN

```python
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
```

```
sync every 500   greedy return at 4k,8k,..,20k steps [204, 163, 99, 108, 180], max Q(start) 34.02
sync every 100   greedy return at 4k,8k,..,20k steps [206, 169, 244, 500, 500], max Q(start) 91.65
no target net    greedy return at 4k,8k,..,20k steps [9, 10, 9, 9, 9], max Q(start) 7.881e+06
(49 s) Upper bound on any Q here: 1/(1-gamma) = 100
```

### Script 6: actor-critic and PPO

```python
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
```

```
setting                 seed  train return  greedy eval  mean KL   max KL  |r-1|>0.2
1 epoch, no clip           0        222.8        463.8    0.0001    0.001     0.000
1 epoch, no clip           1        212.2        454.8    0.0001    0.001     0.000
1 epoch, no clip           2        242.7        453.7    0.0001    0.001     0.000
PPO 10 epochs clip 0.2     0        500.0        500.0    0.0033    0.031     0.034
PPO 10 epochs clip 0.2     1        500.0        500.0    0.0036    0.018     0.037
PPO 10 epochs clip 0.2     2        500.0        500.0    0.0039    0.061     0.042
10 epochs, no clip         0        125.7        500.0    0.1339   16.697     0.175
10 epochs, no clip         1        291.5        500.0    0.1075    5.077     0.187
10 epochs, no clip         2        477.1        500.0    0.1275    2.290     0.252
(41 s total)
```

## Common mistakes

- **Treating time-limit truncation as termination.** Store `terminated` and `truncated` separately and bootstrap through truncations.
- **Bootstrapping across a reset.** After auto-reset, the next observation is a new episode's start. Use the stored final state and cut GAE at episode boundaries.
- **Missing `(1 - done)`, or gradients flowing into the target.** Compute targets under `torch.no_grad()`. Q-values above $r_{\max}/(1-\gamma)$ flag these bugs and overestimation.
- **Reading the surrogate loss as progress.** Track returns, KL and entropy instead.
- **Sign errors.** Policy gradient ascends; optimizers descend. A flipped sign learns to fail, efficiently.
- **One seed, one evaluation, or mixing greedy and exploring returns.** Script 6 has runs with KL spikes of 16.7 that end with a perfect greedy policy. Report several seeds with their spread (L0-04), and say which policy you evaluated.

## Exercises

1. [warm-up] In script 1, set `gamma = 0.9` with `slip=0.2`. What does the policy do in the pit-side corridor, and why?
2. [warm-up] Show that $\mathbb E_{a \sim \pi}[\nabla_\theta \log \pi_\theta(a)] = 0$ and check it numerically for the softmax in script 3.
3. [core] Prove that the Bellman optimality operator is a $\gamma$-contraction in the max norm, and bound the value-iteration sweeps needed for error $\varepsilon$.
4. [core] On the cliff, decay $\epsilon$ from 0.1 to 0 over the 500 episodes. What happens to SARSA's and Q-learning's paths and online returns?
5. [core] Add $n$-step TD ($n = 1, 2, 4, 8$) to the random walk and compare RMS error after 50 episodes across $\alpha$.
6. [core] Run script 5 with `sync=100` for 5 seeds and report the median and range of greedy returns. How much of section 6 survives?
7. [core] Implement Double DQN (one line) and run it at `sync=50` and `sync=100`. Does $Q(\text{start})$ stay below 100?
8. [core] In script 6, sweep $\lambda \in \{0, 0.5, 0.95, 1\}$ for PPO and relate learning speed to section 8's bias-variance argument.
9. [stretch] Turn script 6 into REINFORCE with a learned baseline (MC returns, one epoch) and compare sample efficiency with PPO.
10. [stretch] In script 6, treat truncation as termination and set `MAX_STEPS = 200`. Compare the critic's start-state value with the true discounted return.

### Hints and answers

1. Running it: the corridor cells all point up, into the wall. That leaves the agent in place 80% of the time and slips it sideways 20%, never into a pit, so it drifts safely; with $\gamma = 0.9$ slowness is cheap. The optimizer exploits every detail of the model you give it. (Value iteration needs 166 sweeps because of the self-loops.)
2. $\sum_a \pi \nabla \log \pi = \sum_a \nabla \pi = \nabla 1 = 0$. For the softmax, $\nabla \log \pi(a) = e_a - \pi$ and $\sum_a \pi_a (e_a - \pi) = 0$.
3. Use $|\max f - \max g| \le \max |f - g|$ and that transition probabilities sum to 1. From $V_0 = 0$, the error after $k$ sweeps is at most $\gamma^k \|V^*\|_\infty$.
4. As $\epsilon \to 0$ both converge to the edge path and the online gap closes, because SARSA's cost of exploring vanishes.
5. Intermediate $n$ (2 to 4) usually wins at its best $\alpha$, as in Sutton and Barto's treatment of n-step methods.
6. A 3-seed sweep while writing gave final returns of 500, 500 and 488, with dips to 115 to 170 earlier. Report the spread across seeds and checkpoints.
7. Use `boot = q_targ(s2).gather(1, q(s2).argmax(1, keepdim=True)).squeeze(1)`. At `sync=50` I measured 103 to 120 over 3 seeds: Double DQN removes one source of upward bias, not all of them.
8. $\lambda = 0$ suffers from an untrained critic's bias, $\lambda = 1$ from variance; intermediate values usually learn fastest.
9. Expect REINFORCE to need several times more steps: one use per batch and high-variance returns.
10. Expect the critic to underestimate the start state, since it is taught that episodes end by step 200 though nothing in the observation says how many steps remain. The policy may still learn, which is why this bug survives.

## Check your understanding

**1. Why does TD usually have lower variance than MC, and what does it pay?** Its target involves one random reward and transition instead of a whole episode. It pays with bias from the current estimate $V(s')$, which with function approximation can compound (the deadly triad).

**2. What does off-policy mean for Q-learning, concretely?** Its target uses $\max_{a'} Q(s', a')$, the value of acting greedily next, whatever the agent actually does, so it learns $Q^*$ from exploratory behavior. SARSA learns the value of the exploring policy itself, which makes it cautious on the cliff.

**3. Why does a baseline not bias the policy gradient?** The score has zero mean, $\mathbb E_{a\sim\pi}[\nabla \log \pi(a|s)] = 0$, so any term $b(s)$ that ignores the action contributes nothing in expectation. It removes reward that every action would have gotten, which is pure noise for the gradient.

**4. What do replay and the target network fix in DQN?** Replay decorrelates samples and reuses data. The target network freezes the labels between syncs so the regression is not chasing itself. Without it, script 5's Q-values grew to about $7.9 \times 10^6$.

**5. What does PPO's clip do, and what does it not do?** It removes the incentive to push a sample's ratio beyond $1 \pm \epsilon$ in the direction its advantage favors, which makes several epochs on one batch safe. It is not a hard KL limit: the policy can drift through many small steps or through generalization to other states, so you still monitor KL.

## Further reading

- **Richard S. Sutton and Andrew G. Barto, "Reinforcement Learning: An Introduction", 2nd edition** (2018, MIT Press). http://incompleteideas.net/book/the-book-2nd.html
  *Read for:* the standard text, including the random-walk and cliff-walking examples used here. Free PDF from the authors. *Level:* intro.
- **OpenAI, "Spinning Up in Deep RL", Part 3: Intro to Policy Optimization** (documentation). https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
  *Read for:* a short derivation of the policy gradient, reward-to-go and baselines, with code. *Level:* intro.
- **Alex Irpan, "Deep Reinforcement Learning Doesn't Work Yet"** (2018, blog). https://www.alexirpan.com/2018/02/14/rl-hard.html
  *Read for:* concrete examples of sample inefficiency, reward hacking, local optima and seed variance. *Level:* intro.
- **Christopher J. C. H. Watkins and Peter Dayan, "Q-learning"** (1992, *Machine Learning* 8, 279 to 292). https://doi.org/10.1007/BF00992698
  *Read for:* the convergence proof for tabular Q-learning. May be paywalled. *Level:* advanced.
- **Ronald J. Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning"** (1992, *Machine Learning* 8, 229 to 256). https://doi.org/10.1007/BF00992696
  *Read for:* the original REINFORCE algorithms and reinforcement baselines. May be paywalled. *Level:* advanced.
- **Volodymyr Mnih et al., "Human-level control through deep reinforcement learning"** (2015, *Nature* 518, 529 to 533). https://doi.org/10.1038/nature14236
  *Read for:* DQN with replay and target networks on 49 Atari games. Paywalled; the earlier workshop paper "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013, arXiv:1312.5602, https://arxiv.org/abs/1312.5602) is free. *Level:* intermediate.
- **Hado van Hasselt, Arthur Guez, David Silver, "Deep Reinforcement Learning with Double Q-learning"** (2016, AAAI; arXiv:1509.06461). https://arxiv.org/abs/1509.06461
  *Read for:* why max-based targets overestimate, and the Double DQN fix. *Level:* intermediate.
- **John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel, "Trust Region Policy Optimization"** (2015, ICML; arXiv:1502.05477). https://arxiv.org/abs/1502.05477
  *Read for:* the KL-constrained update that PPO approximates. *Level:* advanced.
- **John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation"** (2016, ICLR; arXiv:1506.02438). https://arxiv.org/abs/1506.02438
  *Read for:* GAE and the bias-variance role of $\lambda$. *Level:* advanced.
- **John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov, "Proximal Policy Optimization Algorithms"** (2017, arXiv:1707.06347). https://arxiv.org/abs/1707.06347
  *Read for:* the clipped objective, the KL-penalty variant, and Table 1's comparison of clip values. *Level:* intermediate.
- **Shengyi Huang, Rousslan Fernand Julien Dossa, Antonin Raffin, Anssi Kanervisto, Weixun Wang, "The 37 Implementation Details of Proximal Policy Optimization"** (2022, ICLR Blog Track). https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
  *Read for:* what reference PPO implementations actually do, and which details matter. *Level:* intermediate.
- **Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, David Meger, "Deep Reinforcement Learning that Matters"** (2018, AAAI; arXiv:1709.06560). https://arxiv.org/abs/1709.06560
  *Read for:* how seeds, hyperparameters and implementations change deep RL results, and how to report them. *Level:* intermediate.
