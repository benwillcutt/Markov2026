import numpy as np
import matplotlib.pyplot as plt

a, b = 1, 1

p = np.zeros((5, 5))
for k in range(5):
    if k < 4:
        p[k, k + 1] = a * (4 - k) / 4
    if k > 0:
        p[k, k - 1] = b * k / 4
    p[k, k] = 1 - p[k].sum()

N = 60
q = np.zeros((N + 1, 5))
q[0, 0] = 1
for n in range(N):
    q[n + 1] = q[n] @ p

running_avg = np.cumsum(q[:, 2]) / np.arange(1, N + 2)

np.set_printoptions(precision=6, suppress=True)
print("p =")
print(p)
print("q50 =", q[50])
print("q51 =", q[51])
print("avg of q_k(2), k <= 60:", running_avg[N])

eps = 2.0 ** -(N + 1)
print("formula q60:", np.array([1/8 + eps, 0, 3/4, 0, 1/8 - eps]))
print("formula q51:", np.array([0, 1/2 + 2.0**-51, 0, 1/2 - 2.0**-51, 0]))

n = np.arange(N + 1)
plt.figure(figsize=(8, 4.5))
plt.plot(n, q[:, 2], "o-", ms=3, label="q_n(2)")
plt.plot(n, q[:, 4], "s-", ms=3, label="q_n(4)")
plt.plot(n, running_avg, "k--", label="running average of q_k(2)")
plt.axhline(3 / 8, color="gray", ls=":", lw=0.8)
plt.xlabel("n")
plt.ylabel("probability")
plt.title("a = b = 1, start at state 0")
plt.legend()
plt.tight_layout()
plt.savefig("HW5 problem 1b.png", dpi=150)