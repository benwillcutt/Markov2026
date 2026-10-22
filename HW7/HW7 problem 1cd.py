import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "HW7/"
x = np.array([-1.77, 2.47, 1.34, 2.18, 13.05, 1.60, 1.50, -1.97, 0.79, 1.50, 2.00, 2.63, 1.19, -2.31, 0.87])
n = len(x)

def logg(a, b):
    return n * b - np.sum(np.log(np.exp(2 * b) + (x - a) ** 2))

def run(s, T=50000, seed=0):
    rng = np.random.default_rng(seed)
    z = np.zeros(2)
    lg = logg(*z)
    out = np.empty((T, 2))
    acc = 0
    for t in range(T):
        y = z + s * rng.standard_normal(2)
        ly = logg(*y)
        if np.log(rng.random()) < ly - lg:
            z, lg = y, ly
            acc += 1
        out[t] = z
    return out, acc / T

def iat(v):
    v = v - v.mean()
    m = len(v)
    f = np.fft.rfft(v, 2 * m)
    acf = np.fft.irfft(f * np.conj(f))[:m]
    acf /= acf[0]
    tau = 2 * np.cumsum(acf) - 1
    k = np.arange(m)
    ok = np.where(k >= 5 * tau)[0]
    return tau[ok[0]]

res = {}
fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
for i, s in enumerate([0.02, 0.5, 5.0]):
    ch, ar = run(s)
    res[s] = ch
    print("s", s, "accept", ar, "IAT alpha", iat(ch[:, 0]))
    ax[i].plot(ch[:400, 0])
    ax[i].set_ylabel(f"alpha, s={s}")
ax[2].set_xlabel("step")
plt.tight_layout()
plt.savefig(D + "HW7 problem 1c fig1.png", dpi=130)

best = min(res, key=lambda s: iat(res[s][:, 0]))
print("best s", best)
ch = res[best][5000:]
al = ch[:, 0]
be = np.exp(ch[:, 1])

ag = np.linspace(-1.5, 4.5, 1200)
bg = np.linspace(-3.5, 2.0, 1200)
A, B = np.meshgrid(ag, bg, indexing="ij")
L = n * B - np.sum(np.log(np.exp(2 * B)[..., None] + (x - A[..., None]) ** 2), axis=-1)
W = np.exp(L - L.max())
W /= W.sum()

def wq(vals, w, qs):
    o = np.argsort(vals)
    c = np.cumsum(w[o])
    return [vals[o][np.searchsorted(c, q)] for q in qs]

wa = W.sum(axis=1)
wb = W.sum(axis=0)
amean = (ag * wa).sum()
bmean = (np.exp(bg) * wb).sum()
aq = wq(ag, wa, [0.05, 0.95])
bq = wq(np.exp(bg), wb, [0.05, 0.95])
print("grid alpha mean", amean, "90%", aq)
print("grid beta mean", bmean, "90%", bq)
print("chain alpha mean", al.mean(), "90%", np.percentile(al, [5, 95]))
print("chain beta mean", be.mean(), "90%", np.percentile(be, [5, 95]))
print("sample mean", x.mean(), "sample median", np.median(x))

plt.figure(figsize=(6, 5))
plt.scatter(ch[::10, 0], ch[::10, 1], s=3, alpha=0.4)
plt.contour(ag, bg, W.T, levels=np.max(W) * np.array([0.01, 0.05, 0.2, 0.5, 0.8]), colors="k")
plt.xlabel("alpha")
plt.ylabel("log beta")
plt.tight_layout()
plt.savefig(D + "HW7 problem 1c fig2.png", dpi=130)

rng = np.random.default_rng(3)
print("n, std-equivalent scatter (IQR/1.349) for mean, median")
for m in (10, 100, 1000):
    d = 1.5 + 0.8 * np.tan(rng.uniform(-np.pi / 2, np.pi / 2, size=(2000, m)))
    for nm, e in (("mean", d.mean(axis=1)), ("median", np.median(d, axis=1))):
        q = np.percentile(e, [25, 75])
        print(m, nm, (q[1] - q[0]) / 1.349)
print("theory: mean", 2 * 0.8 / 1.349, "median approx", 0.8 * np.pi / 2 / np.sqrt(np.array([10, 100, 1000])))

c = 1.5 + 0.8 * rng.standard_cauchy(10**6)
rm = np.cumsum(c) / np.arange(1, 10**6 + 1)
print("running mean at", [(k, round(rm[k - 1], 3)) for k in (10, 100, 10**3, 10**4, 10**5, 10**6)])