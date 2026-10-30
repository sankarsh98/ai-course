"""A tiny customer-service environment, a task suite, a grader, and scripted agents.

IMPORTANT: the "agents" below are rule-based or randomized scripts. They stand in
for language-model agents so that evaluation ideas can be run offline, with no
API key. Their error rates are numbers we chose, not measurements of any model.
"""
import copy
import random

TODAY = 100          # the environment's current day number
SYSTEM_TOKENS = 800  # pretend size of system prompt + tool definitions
CALL_TOKENS = 30     # pretend output tokens per tool call
RESULT_TOKENS = 80   # pretend tokens per tool result added to the context


# ---------------------------------------------------------------- environment
class Shop:
    """An order database exposed through three tools, with token accounting."""

    def __init__(self, orders):
        self.orders = copy.deepcopy(orders)
        self.context = SYSTEM_TOKENS + 60      # system prompt + user request
        self.input_tokens = 0                  # summed over all model calls
        self.output_tokens = 0
        self.calls = 0

    def _turn(self):
        # Each tool call is one model call that re-reads the whole context.
        self.input_tokens += self.context
        self.output_tokens += CALL_TOKENS
        self.context += CALL_TOKENS + RESULT_TOKENS
        self.calls += 1

    def get_order(self, oid):
        self._turn()
        if oid not in self.orders:
            return {"error": f"no order {oid}"}
        return copy.deepcopy(self.orders[oid])

    def refund(self, oid, amount):
        self._turn()
        if oid not in self.orders or amount is None:
            return {"error": "bad arguments"}
        self.orders[oid]["refunded"] += amount
        return {"ok": True}

    def set_status(self, oid, status):
        self._turn()
        if oid not in self.orders or status is None:
            return {"error": "bad arguments"}
        self.orders[oid]["status"] = status
        return {"ok": True}

    def finish(self):
        self.input_tokens += self.context      # final answer turn
        self.output_tokens += 50


# ---------------------------------------------------------------- tasks
def make_tasks(n=30, seed=0, first_id=100):
    """Three task kinds. Each task names one order; the rest of the DB is shared."""
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        kind = "refund" if i < 0.4 * n else "cancel" if i < 0.73 * n else "partial"
        items = [{"price": rng.randint(5, 80), "damaged": rng.random() < 0.4}
                 for _ in range(rng.randint(2, 4))]
        if kind == "partial" and not any(it["damaged"] for it in items):
            items[0]["damaged"] = True
        if kind == "refund":
            # Two tasks sit exactly on the 30-day policy boundary.
            age = 30 if i in (3, 8) else rng.choice([5, 12, 20, 29, 31, 45, 60])
            status = "delivered"
        elif kind == "cancel":
            age, status = 0, rng.choice(["processing", "processing", "shipped"])
        else:
            age, status = rng.randint(1, 25), "delivered"
        order = {"status": status, "day": TODAY - age, "items": items, "refunded": 0}
        tasks.append({"kind": kind, "oid": first_id + i, "order": order})
    return tasks


def initial_db(tasks):
    return {t["oid"]: t["order"] for t in tasks}


def total(order, damaged_only=False):
    return sum(it["price"] for it in order["items"] if it["damaged"] or not damaged_only)


def correct_plan(kind, order, oid):
    """The policy, written as the list of write actions a perfect agent makes."""
    if kind == "refund":
        if TODAY - order["day"] <= 30:
            return [("refund", oid, total(order)), ("set_status", oid, "refunded")]
        return [("set_status", oid, "denied")]
    if kind == "cancel":
        if order["status"] == "processing":
            return [("set_status", oid, "cancelled"), ("refund", oid, total(order))]
        return []                                   # shipped: change nothing
    return [("refund", oid, total(order, damaged_only=True)),
            ("set_status", oid, "partially_refunded")]


def goal_db(tasks, task):
    """Ground truth final state: apply the correct plan to a fresh database."""
    env = Shop(initial_db(tasks))
    execute(env, correct_plan(task["kind"], task["order"], task["oid"]))
    return env.orders


def execute(env, plan):
    for tool, oid, arg in plan:
        if tool == "refund":
            env.refund(oid, arg)
        else:
            env.set_status(oid, arg)


def grade_state(final, goal):
    """Outcome grading: the whole database must match, so side effects count."""
    return final == goal


# ---------------------------------------------------------------- agents
def rule_bot(task, env, rng):
    """Deterministic. Has no rule for partial refunds, and an off-by-one at 30 days."""
    oid = task["oid"]
    o = env.get_order(oid)
    if task["kind"] == "cancel":
        plan = correct_plan("cancel", o, oid)
    elif TODAY - o["day"] < 30:                     # bug: policy says <= 30
        plan = [("refund", oid, total(o)), ("set_status", oid, "refunded")]
    else:
        plan = [("set_status", oid, "denied")]
    # Partial-refund requests also land in the branches above: a full refund.
    execute(env, plan)
    env.finish()


def make_sampled_agent(p_reason, p_exec, verify=False):
    """A randomized stand-in for a sampled LLM agent.

    p_reason: per-task probability of a reasoning slip (wrong decision or amount),
              a dict by task kind. Re-reading the order cannot catch these.
    p_exec:   per-call probability of an execution slip. 70% of slips are malformed
              calls the tool rejects; 30% hit the wrong order (a side effect).
    verify:   after each write, re-read the order and retry rejected calls.
    """
    def agent(task, env, rng):
        oid = task["oid"]
        o = env.get_order(oid)
        plan = correct_plan(task["kind"], o, oid)
        if rng.random() < p_reason[task["kind"]]:
            plan = corrupt(plan, task["kind"], o, oid, rng)
        for tool, target, arg in plan:
            call = (tool, target, arg)
            if rng.random() < p_exec:
                call = (tool, target, None) if rng.random() < 0.7 else (tool, target + 1, arg)
            out = env.refund(*call[1:]) if tool == "refund" else env.set_status(*call[1:])
            if verify:
                env.get_order(oid)                  # costs a call; sees the error
                if "error" in out:
                    execute(env, [(tool, target, arg)])
        env.finish()
    return agent


def corrupt(plan, kind, o, oid, rng):
    """Plausible reasoning mistakes."""
    if kind == "partial":                           # forgets an item or refunds all
        first_damaged = next(it["price"] for it in o["items"] if it["damaged"])
        amt = total(o) if rng.random() < 0.5 else total(o, True) - first_damaged
        return [("refund", oid, amt), ("set_status", oid, "partially_refunded")]
    if kind == "refund":                            # misreads the policy
        flipped = TODAY - o["day"] > 30
        return correct_plan("refund", dict(o, day=TODAY if flipped else TODAY - 99), oid)
    if o["status"] == "processing":                 # cancels but forgets the refund
        return [("set_status", oid, "cancelled")]
    return [("set_status", oid, "cancelled"), ("refund", oid, total(o))]  # ignores "shipped"


def run_trial(agent, tasks, task, seed):
    env = Shop(initial_db(tasks))
    agent(task, env, random.Random(seed))
    ok = grade_state(env.orders, goal_db(tasks, task))
    return ok, env


AGENTS = {
    "rule-bot": rule_bot,
    "sampled-A": make_sampled_agent({"refund": 0.06, "cancel": 0.06, "partial": 0.25}, 0.08),
    "sampled-B": make_sampled_agent({"refund": 0.10, "cancel": 0.10, "partial": 0.25}, 0.08),
    "sampled-A+verify": make_sampled_agent({"refund": 0.06, "cancel": 0.06, "partial": 0.25},
                                           0.08, verify=True),
}
