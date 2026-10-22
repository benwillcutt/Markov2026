import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

lam = 4.0
H = 200000
rng = np.random.default_rng(7)

G = rng.exponential(1 / lam, size=(H, 40))
T = np.cumsum(G, axis=1)
NA = (T <= 1).sum(axis=1)
firstA = T[:, 0]
T3A = T[:, 2]
halfA = (T <= 0.5).sum(axis=1)

NB = rng.poisson(lam, size=H)
maxn = NB.max()
mask = np.arange(maxn)[None, :] < NB[:, None]
U = np.sort(np.where(mask, rng.random((H, maxn)), np.inf), axis=1)
halfB = (U <= 0.5).sum(axis=1)
T3B = U[:, 2]
firstB = U[:, 0]

for name, N in (("gaps", NA), ("poisson+uniform", NB)):
    print(name, "mean", N.mean(), "var", N.var(), "P(N=0)", (N == 0).mean())

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
k = np.arange(0, 15)
for a, N, nm in ((ax[0], NA, "exponential gaps"), (ax[1], NB, "Poisson count + uniforms")):
    a.bar(k, np.bincount(N, minlength=15)[:15] / H, alpha=0.5, label="simulated")
    a.plot(k, stats.poisson.pmf(k, lam), "ko-", ms=4, label="Poisson(4)")
    a.set_title(nm)
    a.set_xlabel("N(1)")
ax[0].legend()
plt.tight_layout()
plt.savefig("HW7 problem 3d fig1.png", dpi=130)

fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
for r in range(10):
    ta = T[r][T[r] <= 1]
    ax[0].vlines(ta, r, r + 0.8)
    ub = U[r][np.isfinite(U[r])]
    ax[1].vlines(ub, r, r + 0.8)
ax[0].set_title("exponential gaps")
ax[1].set_title("Poisson count + uniforms")
for a in ax:
    a.set_xlim(0, 1)
    a.set_xlabel("time (hours)")
    a.set_ylabel("realization")
plt.tight_layout()
plt.savefig("HW7 problem 3d fig2.png", dpi=130)

x = np.linspace(0, 3, 400)
plt.figure(figsize=(7, 4))
plt.hist(T3A, bins=80, range=(0, 3), density=True, alpha=0.5, label="gaps: T3 histogram")
plt.plot(x, stats.gamma.pdf(x, 3, scale=1 / lam), "k", label="Gamma(3,4) density")
plt.xlabel("T3 (hours)")
plt.legend()
plt.tight_layout()
plt.savefig("HW7 problem 3d fig3.png", dpi=130)

bins = np.linspace(0, 1, 41)
cB, _ = np.histogram(T3B[np.isfinite(T3B)], bins=bins)
w = bins[1] - bins[0]
dB = cB / (H * w)
mid = (bins[:-1] + bins[1:]) / 2
print("T3 gaps: mean", T3A.mean(), "exact", 3 / lam, "var", T3A.var(), "exact", 3 / lam**2)
print("T3 via uniforms (T3<=1) max abs density err",
      np.abs(dB - stats.gamma.pdf(mid, 3, scale=1 / lam)).max())

for name, N, h, f1 in (("gaps", NA, halfA, firstA), ("poisson+uniform", NB, halfB, firstB)):
    s = N == 3
    print(name, "hours with N=3:", s.sum(),
          "P(N(.5)=0|N=3)", (h[s] == 0).mean(),
          "E[T1|N=3]", f1[s].mean(),
          "corr first/second half", np.corrcoef(h, N - h)[0, 1])

print("exact P(N(.5)=0|N=3)", 1 / 8, "E[T1|N=3]", 1 / 4)
print("P(N(.25)=0)", np.exp(-1), "P(T3>1)", stats.poisson.cdf(2, 4))

ms = np.unique(np.round(np.geomspace(10, 2000, 25)).astype(int))
ms = np.union1d(ms, [10, 2000])
tv = []
for m in ms:
    p = lam / m
    kk = np.arange(0, m + 1)
    d = np.abs(stats.binom.pmf(kk, m, p) - stats.poisson.pmf(kk, lam)).sum() + stats.poisson.sf(m, lam)
    tv.append(0.5 * d)
tv = np.array(tv)
dt = 1 / ms
slope, icpt = np.polyfit(np.log(dt), np.log(tv), 1)
print("TV slope", slope, "TV(dt=1/10)", tv[0], "TV(dt=1/2000)", tv[-1], "ratio TV/dt", tv[0] / dt[0], tv[-1] / dt[-1])
plt.figure(figsize=(6, 4))
plt.loglog(dt, tv, "o", label="TV distance")
plt.loglog(dt, np.exp(icpt) * dt**slope, "k--", label=f"fit slope {slope:.3f}")
plt.xlabel("dt")
plt.ylabel("TV")
plt.legend()
plt.tight_layout()
plt.savefig("HW7 problem 3d fig4.png", dpi=130)