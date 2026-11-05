import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
D, T = 30, 720.0

def lam(t):
    return 12 + 8 * np.cos(np.pi * (t - 16) / 12)

N = rng.poisson(20 * T)
prop = rng.uniform(0, T, N)
t = np.sort(prop[rng.uniform(size=N) < lam(prop) / 20])
n = len(t)
print("proposed", N, "kept", n, "fraction", n / N)

hour = (t % 24).astype(int)
mean_by_hour = np.bincount(hour, minlength=24) / D
tt = np.linspace(0, 24, 400)
plt.bar(np.arange(24) + 0.5, mean_by_hour, width=0.9, label="simulated mean per clock hour")
plt.plot(tt, lam(tt), color="red", label="true rate")
plt.xlabel("hour of day")
plt.ylabel("arrivals per hour")
plt.legend()
plt.savefig("HW8/HW8 problem 3c.png", dpi=150)

day = (t // 24).astype(int)
win = ((t % 24) >= 3) & ((t % 24) < 4)
cnt = np.bincount(day[win], minlength=D)
print("3-4am mean/var", cnt.mean(), cnt.var(ddof=1))

def nll(p):
    a, b, phi = p
    l = a + b * np.cos(np.pi * (t - phi) / 12)
    if np.any(l <= 0):
        return 1e10
    L = a * T + b * 12 / np.pi * (np.sin(np.pi * (T - phi) / 12) - np.sin(-np.pi * phi / 12))
    return -(np.log(l).sum() - L)

res = optimize.minimize(nll, [10, 5, 14], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
print("fit (a, b, phi)", res.x, "max loglik", -res.fun)
print("constant rate", n / T, "loglik", n * np.log(n / T) - n)