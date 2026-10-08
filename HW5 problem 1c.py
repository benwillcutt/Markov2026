import numpy as np
from math import comb
import matplotlib.pyplot as plt


def build_p(a, b):
    p = np.zeros((5, 5))
    for k in range(5):
        if k < 4:
            p[k, k + 1] = a * (4 - k) / 4
        if k > 0:
            p[k, k - 1] = b * k / 4
        p[k, k] = 1 - p[k].sum()
    return p


def run_chain(p, steps):
    q = np.zeros((steps + 1, 5))
    q[0, 0] = 1
    for n in range(steps):
        q[n + 1] = q[n] @ p
    return q


p1 = build_p(1, 1)
p2 = build_p(0.3, 0.1)

A = np.vstack([p2.T - np.eye(5), np.ones(5)])
rhs = np.zeros(6)
rhs[-1] = 1
pi = np.linalg.lstsq(A, rhs, rcond=None)[0]

theta = 0.3 / (0.3 + 0.1)
pi_binom = np.array([comb(4, k) * theta**k * (1 - theta)**(4 - k) for k in range(5)])

np.set_printoptions(precision=6, suppress=True)
print("p =")
print(p2)
print("pi (linear solve):", pi)
print("pi (binomial)    :", pi_binom)
print("max difference   :", np.abs(pi - pi_binom).max())

q2 = run_chain(p2, 400)
err = np.abs(q2 - pi).max(axis=1)
n_star = int(np.argmax(err < 1e-6))
print("smallest n with error < 1e-6:", n_star)
print("error at n-1 and n:", err[n_star - 1], err[n_star])

eigs = np.linalg.eigvals(p2)
print("eigenvalues:", np.sort(eigs.real))
print("second largest modulus:", np.sort(np.abs(eigs))[-2])
print("trace of p:", np.trace(p2))
print("err / 0.9^n at n = 134:", err[134] / 0.9**134)

q1 = run_chain(p1, 60)
avg = np.cumsum(q1[:, 2]) / np.arange(1, 62)
n = np.arange(61)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

ax[0].plot(n, q1[:, 2], "o-", ms=3, label="q_n(2)")
ax[0].plot(n, q1[:, 4], "s-", ms=3, label="q_n(4)")
ax[0].plot(n, avg, "k--", label="running average of q_k(2)")
ax[0].set_title("a = b = 1")
ax[0].set_xlabel("n")
ax[0].set_ylabel("probability")
ax[0].legend()

for k in range(5):
    line, = ax[1].plot(n, q2[:61, k], label=f"q_n({k})")
    ax[1].axhline(pi[k], color=line.get_color(), ls=":", lw=0.8)
ax[1].set_title("a = 0.3, b = 0.1")
ax[1].set_xlabel("n")
ax[1].legend()

plt.tight_layout()
plt.savefig("HW5 problem 1c.png", dpi=150)