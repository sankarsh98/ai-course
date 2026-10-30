"""A tiny scalar reverse-mode autodiff engine (in the spirit of Karpathy's micrograd).

Each Value records the Values it was computed from and a closure that pushes
its gradient back to them using the local derivative (the chain rule).
"""
import math


class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = parents
        self._op = op
        self._backward = lambda: None  # set by the operation that created this node

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    @staticmethod
    def lift(x):
        return x if isinstance(x, Value) else Value(x)

    # ---- elementary operations: value plus local derivative ----
    def __add__(self, other):
        other = Value.lift(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1. Use += so that a node used twice
            # (a fork in the graph) accumulates gradient from both paths.
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = Value.lift(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad  # reuse the forward result
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def sin(self):
        out = Value(math.sin(self.data), (self,), "sin")

        def _backward():
            self.grad += math.cos(self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            # Convention: derivative 0 at exactly 0 (the kink).
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    # ---- derived operations, built from the ones above ----
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-Value.lift(other))

    def __truediv__(self, other):
        return self * Value.lift(other) ** -1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return Value.lift(other) - self

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return Value.lift(other) / self

    # ---- reverse pass ----
    def backward(self):
        # Topological order: every node appears after all the nodes it depends on.
        order, seen = [], set()

        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._parents:
                    visit(p)
                order.append(v)
        visit(self)

        for v in order:
            v.grad = 0.0  # clear stale gradients from earlier calls
        self.grad = 1.0  # dL/dL = 1
        for v in reversed(order):
            v._backward()
        return order  # handy for counting nodes
