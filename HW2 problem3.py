import numpy as np
import matplotlib.pyplot as plt
import time

def sample(lam, c, N, rng):
    out = []
    trials = 0
    while len(out) < N:
        U1, U2 = rng.random(), rng.random()
        X = -np.log(1 - U1) / lam
        trials += 1
        if U2 < (X * np.exp(-X)) / (c * lam * np.exp(-lam * X)):
            out.append(X)
    return np.array(out), trials

rng = np.random.default_rng(0)
N = 10000
params = [(0.5, 4/np.e), (0.2, 1/(0.16*np.e))]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (lam, c) in zip(axes, params):
    t0 = time.perf_counter()
    samples, trials = sample(lam, c, N, rng)
    t1 = time.perf_counter()

    acc_frac = N / trials
    time_per = (t1 - t0) / N
    counts, edges, _ = ax.hist(samples, bins=60, density=True, alpha=0.6, label='Samples')
    width = edges[1] - edges[0]

    x = np.linspace(0, 15, 500)
    ax.plot(x, x * np.exp(-x), 'r-', linewidth=2, label='f(x)')
    ax.set_title(f'λ={lam}, c={c:.4f}, bin width={width:.4f}\n'
                 f'accept frac={acc_frac:.4f}, 1/c={1/c:.4f}, {time_per*1e6:.3f} μs/sample')
    ax.legend()

plt.tight_layout()
plt.savefig('HW2 problem3.png', dpi=140)