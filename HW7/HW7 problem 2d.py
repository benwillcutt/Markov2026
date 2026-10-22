import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import ks_2samp

lam = np.array([3, 2, 1.5, 1.2, 1, 0.75])
n = len(lam)

def dp(lam):
    n = len(lam)
    E = np.zeros(1 << n)
    L = np.zeros((1 << n, n))
    for S in range(1, 1 << n):
        idx = [i for i in range(n) if S >> i & 1]
        r = lam[idx].sum()
        if len(idx) == 1:
            E[S] = 1 / r
            L[S, idx[0]] = 1
            continue
        E[S] = 1 / r
        for i in idx:
            T = S ^ (1 << i)
            E[S] += lam[i] / r * E[T]
            L[S] += lam[i] / r * L[T]
    return E, L

E, L = dp(lam)
full = (1 << n) - 1
print("subsets:", 1 << n, "nonempty:", (1 << n) - 1)
print("E[M] recursion:", E[full])
print("P(last):", L[full], L[full].sum())

alt = 0.0
for k in range(1, n + 1):
    for g in combinations(range(n), k):
        alt += (-1) ** (k + 1) / lam[list(g)].sum()
print("E[M] alternating:", alt)

rng = np.random.default_rng(1)
N = 200000
X = rng.exponential(1 / lam, size=(N, n))
M = X.max(axis=1)
last = X.argmax(axis=1)
print("sim E[M]:", M.mean(), "+/-", M.std() / np.sqrt(N))
print("sim P(last):", np.bincount(last, minlength=n) / N)

t = np.linspace(0, M.max(), 600)
F = np.prod(1 - np.exp(-np.outer(t, lam)), axis=1)
dens = np.zeros_like(t)
for i in range(n):
    others = [j for j in range(n) if j != i]
    dens += lam[i] * np.exp(-lam[i] * t) * np.prod(1 - np.exp(-np.outer(t, lam[others])), axis=1)
plt.figure(figsize=(7, 4))
plt.hist(M, bins=80, density=True, alpha=0.5, label="simulation")
plt.plot(t, dens, "k", label="exact density")
plt.xlabel("M (hours)")
plt.ylabel("density")
plt.legend()
plt.tight_layout()
plt.savefig("HW7 problem 2d fig1.png", dpi=130)

mn = X.min(axis=1)
who = X.argmin(axis=1)
print("1/Lambda:", 1 / lam.sum())
for i in range(n):
    s = mn[who == i]
    print(i + 1, "P(I=i) sim", len(s) / N, "exact", lam[i] / lam.sum(), "mean X|I=i", s.mean(),
          "KS vs rest p", ks_2samp(s, mn[who != i]).pvalue)

ks = np.arange(1, 13)
Ek = []
for k in ks:
    Ek.append(dp(np.arange(1, k + 1, dtype=float))[0][(1 << k) - 1])
print(list(zip(ks, np.round(Ek, 5))))
plt.figure(figsize=(6, 4))
plt.plot(ks, Ek, "o-", label="E[M]")
plt.axhline(2 * np.log(2), color="r", ls="--", label="2 ln 2")
plt.xlabel("k")
plt.ylabel("E[M]")
plt.legend()
plt.tight_layout()
plt.savefig("HW7 problem 2d fig2.png", dpi=130)
print("2ln2", 2 * np.log(2))