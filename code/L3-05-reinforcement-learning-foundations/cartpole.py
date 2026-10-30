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
