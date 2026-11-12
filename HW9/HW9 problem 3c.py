import numpy as np, scipy.linalg as sl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

Q = np.array([[-2,2,0,0],[1,-3,2,0],[0,2,-4,2],[0,0,3,-3]], float)
pi = np.array([3,6,6,4]) / 19
x0 = np.array([1,0,0,0.])
l2 = np.sort(np.linalg.eigvals(Q).real)[::-1][1]

tv = lambda t: 0.5 * np.abs(x0 @ sl.expm(t*Q) - pi).sum()
t = np.linspace(0, 5, 501)
X = np.array([x0 @ sl.expm(s*Q) for s in t])
TV = np.array([tv(s) for s in t])

t01 = brentq(lambda s: tv(s) - 0.01, 0.1, 5)
print(t01, np.log(100) / abs(l2))

C = TV[200] / np.exp(l2 * t[200])
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i in range(4):
    ax[0].plot(t, X[:, i], label=f"x{i}")
ax[0].set_xlabel("t (hours)")
ax[0].set_ylabel("P(X_t = i)")
ax[0].set_title("x_i(t)")
ax[0].legend()
ax[1].semilogy(t, TV, label="TV(t)")
ax[1].semilogy(t, C * np.exp(l2 * t), "--", label="slope λ2")
ax[1].axhline(0.01, color="gray", lw=0.5)
ax[1].set_ylim(1e-6, 1)
ax[1].set_xlabel("t (hours)")
ax[1].set_ylabel("TV(t) (log scale)")
ax[1].legend()
plt.tight_layout()
plt.savefig("HW9/HW9 problem 3c.png", dpi=120)