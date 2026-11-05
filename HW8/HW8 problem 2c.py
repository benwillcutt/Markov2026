import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
W = 200000

S1 = rng.binomial(rng.poisson(15, W), 0.4)
K1 = rng.binomial(rng.poisson(10, W), 0.4)
S2 = rng.poisson(6, W)
K2 = rng.poisson(4, W)

for name, S, K in ("thin", S1, K1), ("direct", S2, K2):
    N = S + K
    print(name, "star mean/var", S.mean(), S.var(), "sky mean/var", K.mean(), K.var())
    print(name, "corr", np.corrcoef(S, K)[0, 1], "total mean/var", N.mean(), N.var())
    emp = np.bincount(N, minlength=40)[:21] / W
    print(name, "max pmf gap vs Poisson(10)", np.abs(emp - stats.poisson.pmf(np.arange(21), 10)).max())
    m = N == 10
    print(name, "windows with 10:", m.sum(), "P(6 star)", (S[m] == 6).mean())

Ts = rng.exponential(1 / 1.2, W)
Tk = rng.exponential(1 / 0.8, W)
first = np.minimum(Ts, Tk)
star = (Ts < Tk).astype(float)
print("P(star first)", star.mean())
print("P(star first and within 1 s)", ((Ts < Tk) & (first <= 1)).mean())
print("E[M]", np.maximum(Ts, Tk).mean())
print("corr(time, type)", np.corrcoef(first, star)[0, 1])