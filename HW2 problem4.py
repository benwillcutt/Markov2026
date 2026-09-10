import numpy as np
import matplotlib.pyplot as plt

a, lf, ls = 0.9, 1000.0, 10.0
rng = np.random.default_rng(0)
N = 100000

U1, U2 = rng.random(N), rng.random(N)
T = np.where(U1 < a, -np.log(1 - U2) / lf, -np.log(1 - U2) / ls)

print("mean:", T.mean(), "vs theory", a/lf + (1-a)/ls)
print("P(T>50ms):", np.mean(T > 0.05), "vs theory", a*np.exp(-lf*0.05) + (1-a)*np.exp(-ls*0.05))

counts, edges, _ = plt.hist(T, bins=150, range=(0, 0.3), density=True, alpha=0.6, label='Empirical')
t = np.linspace(1e-5, 0.3, 2000)
plt.plot(t, a*lf*np.exp(-lf*t) + (1-a)*ls*np.exp(-ls*t), 'r-', label='f(t)')
plt.yscale('log')
plt.xlabel('t (s)'); plt.ylabel('density'); plt.legend()
plt.title(f'bin width = {edges[1]-edges[0]:.4f} s')
plt.savefig('HW2 problem4.png', dpi=140)