import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

R = 20000
T = 10000
d0 = 10 # gap

# one lion
D = np.full(R, d0)
caught = np.zeros(R, dtype=bool)
t_caught1 = np.full(R, T + 1)   # if it never gets caught by T, set to T+1

for t in range(1, T + 1):
    lamb_step = rng.choice([-1, 1], R)
    lion_step = rng.choice([-1, 1], R)
    D = D + lion_step - lamb_step
    just_got_it = (~caught) & (D == 0)
    t_caught1[just_got_it] = t
    caught[just_got_it] = True

# two lions
D1 = np.full(R, d0)
D2 = np.full(R, d0)
caught2 = np.zeros(R, dtype=bool)
t_caught2 = np.full(R, T + 1)

for t in range(1, T + 1):
    lamb_step = rng.choice([-1, 1], R)
    D1 = D1 + rng.choice([-1, 1], R) - lamb_step
    D2 = D2 + rng.choice([-1, 1], R) - lamb_step
    got = (~caught2) & ((D1 == 0) | (D2 == 0))
    t_caught2[got] = t
    caught2[got] = True

# build S(t)
ts = np.unique(np.logspace(0, 4, 200).astype(int))
ts = ts[ts >= 1]

S1 = np.array([(t_caught1 > t).mean() for t in ts])
S2 = np.array([(t_caught2 > t).mean() for t in ts])

window = (ts >= 100) & (ts <= 10000)

# guard for log(0)
ok1 = window & (S1 > 0)
ok2 = window & (S2 > 0)
n_dropped1 = window.sum() - ok1.sum()
n_dropped2 = window.sum() - ok2.sum()

slope1 = np.polyfit(np.log10(ts[ok1]), np.log10(S1[ok1]), 1)[0]
slope2 = np.polyfit(np.log10(ts[ok2]), np.log10(S2[ok2]), 1)[0]
beta1 = -slope1
beta2 = -slope2

print("beta1 =", round(beta1, 3), " (dropped", n_dropped1, "zero points)")
print("beta2 =", round(beta2, 3), " (dropped", n_dropped2, "zero points)")

print("t, S2, S1^2")
for t in [100, 1000, 10000]:
    i = np.argmin(np.abs(ts - t))
    print(ts[i], round(S2[i], 4), round(S1[i] ** 2, 4))

S1_erf = erf(d0 / (2 * np.sqrt(ts)))

plt.figure(figsize=(7.5, 6))
plt.loglog(ts, S1, 'o', markersize=3, label='S1(t), simulated')
plt.loglog(ts, S1_erf, '-', linewidth=1, label='erf(d0 / 2*sqrt(t))')
plt.loglog(ts, S2, 's', markersize=3, label='S2(t), simulated')
plt.loglog(ts, S1 ** 2, '-', linewidth=1, label='S1(t)^2')
plt.axvspan(100, 10000, alpha=0.08, color='gray')
plt.xlabel('t')
plt.ylabel('survival probability')
plt.title('beta1 = %.3f, beta2 = %.3f, fit over 10^2 <= t <= 10^4, R=%d T=%d' % (beta1, beta2, R, T))
plt.legend()
plt.tight_layout()
plt.savefig('HW3 problem 4.png', dpi=150)