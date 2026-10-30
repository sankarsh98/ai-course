"""Three ways an agent benchmark score can mislead: contamination, weak graders,
and peeking at the grader. Agents are scripted stand-ins for models (see shop.py).
"""
import random

import numpy as np

from shop import (AGENTS, Shop, correct_plan, execute, goal_db, grade_state,
                  initial_db, make_sampled_agent, make_tasks, run_trial)

N_TRIALS = 16
public = make_tasks(seed=0, first_id=100)          # published suite
fresh = make_tasks(seed=1, first_id=200)           # same templates, new instances


def score(agent, tasks, grader=grade_state, seed0=0):
    wins = []
    for i, task in enumerate(tasks):
        goal = goal_db(tasks, task)
        for j in range(N_TRIALS):
            env = Shop(initial_db(tasks))
            agent(task, env, random.Random(seed0 + 1000 * i + j))
            wins.append(grader(env.orders, goal, task))
    return np.mean(wins)


def full_grader(final, goal, task):
    return grade_state(final, goal)


# ---- 1. Contamination ---------------------------------------------------------
# The "memorizer" saw the public tasks' solutions during training: for those order
# ids it replays the stored actions. Elsewhere it is a weaker general agent.
memory = {t["oid"]: correct_plan(t["kind"], t["order"], t["oid"]) for t in public}
weak = make_sampled_agent({"refund": 0.25, "cancel": 0.25, "partial": 0.6}, 0.15)


def memorizer(task, env, rng):
    if task["oid"] in memory:
        env.get_order(task["oid"])
        execute(env, memory[task["oid"]])
        env.finish()
    else:
        weak(task, env, rng)


print("1. Contamination: public suite vs fresh instances of the same task types")
for name, agent in [("sampled-A", AGENTS["sampled-A"]), ("memorizer", memorizer)]:
    print(f"  {name:10s} public {score(agent, public, full_grader):.3f}   "
          f"fresh {score(agent, fresh, full_grader):.3f}")


# ---- 2. Weak graders ------------------------------------------------------------
def status_only_grader(final, goal, task):
    """Checks only the target order's status, like a test suite that misses cases."""
    return final[task["oid"]]["status"] == goal[task["oid"]]["status"]


def no_overrefund_grader(final, goal, task):
    """Checks only that no order was refunded more than it should be (a safety check)."""
    return all(final[k]["refunded"] <= goal[k]["refunded"] for k in goal)


def do_nothing(task, env, rng):
    env.finish()                                   # replies without acting


print("\n2. Weak graders: full state vs status-only vs no-over-refund")
for name, agent in [("sampled-A", AGENTS["sampled-A"]), ("rule-bot", AGENTS["rule-bot"]),
                    ("do-nothing", do_nothing)]:
    print(f"  {name:10s} full-state {score(agent, public, full_grader):.3f}   "
          f"status-only {score(agent, public, status_only_grader):.3f}   "
          f"no-over-refund {score(agent, public, no_overrefund_grader):.3f}")
n_noop = sum(correct_plan(t["kind"], t["order"], t["oid"]) == [] for t in public)
print(f"  tasks whose correct outcome is 'change nothing': {n_noop} of {len(public)}")


# ---- 3. Peeking at the grader ---------------------------------------------------
# A harness bug or a leaky setup lets the agent run the hidden grader and retry.
# The reported number is then pass@k, not the pass@1 a user would experience.
def with_oracle_retries(agent, tries):
    def run(task, env, rng):
        goal = goal_db(public, task)
        for _ in range(tries):
            trial = Shop(initial_db(public))
            agent(task, trial, rng)
            env.input_tokens += trial.input_tokens   # we still pay for every try
            env.output_tokens += trial.output_tokens
            if grade_state(trial.orders, goal):
                break
        env.orders = trial.orders
    return run


print("\n3. Letting the agent retry against the hidden grader")
for tries in (1, 3, 5):
    agent = with_oracle_retries(AGENTS["sampled-A"], tries)
    toks = []
    wins = 0
    for i, task in enumerate(public):
        for j in range(N_TRIALS):
            ok, env = run_trial(agent, public, task, seed=1000 * i + j)
            wins += ok
            toks.append(env.input_tokens + env.output_tokens)
    print(f"  up to {tries} tries: reported score {wins / (len(public) * N_TRIALS):.3f}, "
          f"tokens per task {np.mean(toks):6.0f}")
