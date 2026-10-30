"""Arrays, shapes, dtypes, memory, and views vs copies."""
import sys

import numpy as np

# 0. A Python float is a full object; a NumPy float64 element is 8 raw bytes.
lst = [float(i) for i in range(1_000_000)]
list_bytes = sys.getsizeof(lst) + sum(sys.getsizeof(v) for v in lst)
print(f"one Python float object: {sys.getsizeof(1.0)} bytes")
print(f"list of 1e6 floats: {list_bytes / 2**20:.1f} MiB | "
      f"np.float64 array: {np.array(lst).nbytes / 2**20:.1f} MiB\n")


def show(name, a):
    print(f"{name:<10} shape={a.shape!s:<10} dtype={a.dtype!s:<8} "
          f"nbytes={a.nbytes:<6} strides={a.strides}")


# 1. Shape and dtype are the two facts you check first about any array.
x = np.arange(12)                      # default integer dtype is int64 on Linux/macOS
m = x.reshape(3, 4)                    # same 12 numbers, viewed as 3 rows of 4
f32 = m.astype(np.float32)             # a real conversion: new memory
show("x", x)
show("m", m)
show("f32", f32)
show("m.T", m.T)                       # transpose only swaps strides, no copy

# 2. Views share memory with the original; copies do not.
row = m[1]                             # basic indexing -> view
row[0] = 100
print("\nafter row[0] = 100, m[1] =", m[1], "| shares memory:", np.shares_memory(row, m))
picked = m[[0, 2]]                     # integer-array (advanced) indexing -> copy
picked[0, 0] = -1
print("after picked[0,0] = -1, m[0,0] =", m[0, 0], "| shares memory:",
      np.shares_memory(picked, m))

# 3. Integer overflow wraps silently for array arithmetic.
small = np.array([100, 120], dtype=np.int8)   # int8 holds -128..127
print("\nint8 [100, 120] + 10 =", small + np.int8(10))
pix = np.array([200, 250], dtype=np.uint8)    # typical image dtype
print("uint8 [200, 250] + 10 =", pix + np.uint8(10))
print("same sum after casting to int16 =", pix.astype(np.int16) + 10)

# 4. Float precision: float32 has about 7 significant decimal digits.
for dt in (np.float16, np.float32, np.float64):
    info = np.finfo(dt)
    print(f"{dt.__name__:<8} eps={info.eps:.3e}  max={info.max:.3e}  "
          f"smallest normal={info.smallest_normal:.3e}")
big = np.float32(16_777_216)                  # 2**24
print("float32: 2**24 + 1 =", big + np.float32(1), "(the +1 is lost)")
