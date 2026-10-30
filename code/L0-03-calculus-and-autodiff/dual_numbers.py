"""Forward-mode autodiff with dual numbers a + b*eps, where eps**2 = 0.

The 'tan' part carries the derivative along a chosen input direction.
"""
import math


class Dual:
    def __init__(self, val, tan=0.0):
        self.val, self.tan = val, tan  # value and tangent

    def __add__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.val + o.val, self.tan + o.tan)

    def __sub__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.val - o.val, self.tan - o.tan)

    def __mul__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        # (a + b eps)(c + d eps) = ac + (ad + bc) eps: the product rule, for free
        return Dual(self.val * o.val, self.val * o.tan + self.tan * o.val)

    __radd__, __rmul__ = __add__, __mul__


def sin(x):
    return Dual(math.sin(x.val), math.cos(x.val) * x.tan)


def log(x):
    return Dual(math.log(x.val), x.tan / x.val)


def f(x1, x2):
    return log(x1) + x1 * x2 - sin(x2)  # the worked example from section 5


x1, x2 = 2.0, 5.0
for seed in [(1.0, 0.0), (0.0, 1.0), (3.0, -1.0)]:
    y = f(Dual(x1, seed[0]), Dual(x2, seed[1]))
    print(f"seed {seed}: value = {y.val:.6f}, tangent = {y.tan:.6f}")

grad = (1 / x1 + x2, x1 - math.cos(x2))
print(f"analytic gradient = ({grad[0]:.6f}, {grad[1]:.6f}); "
      f"gradient . (3, -1) = {3 * grad[0] - grad[1]:.6f}")
