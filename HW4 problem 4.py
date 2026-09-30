import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

P = np.array([
    [0,    0.5, 0.5, 0,   0   ],
    [0.25, 0,   0,   0.5, 0.25],
    [0.75, 0,   0,   0,   0.25],
    [0,    0,   0,   1,   0   ],
    [0,    0,   0,   0,   1   ],
])
names = ["U", "I", "M"]
cum = P.cumsum(axis=1)

def simulate(start):
    state, steps = start, 0
    while state < 3:
        u = rng.random()
        state = np.searchsorted(cum[state], u, side="right")
        steps += 1
    return state, steps

n_runs = 10_000
results = {}
print("start  h_hat   g_hat   tauF_hat  tauA_hat")
for s in range(3):
    runs = [simulate(s) for _ in range(n_runs)]
    folded = np.array([r[0] == 3 for r in runs])
    T = np.array([r[1] for r in runs])
    results[s] = (folded, T)
    print(f"{names[s]:<6} {folded.mean():.4f}  {T.mean():.4f}  "
          f"{T[folded].mean():.4f}    {T[~folded].mean():.4f}")

folded, T = results[1]
n = np.arange(1, 26, 2)
j = (n - 1) // 2
pmf_fold = np.where(n == 1, 4/5, (1/5) * 0.5**j)
pmf_agg = np.where(n == 1, 2/3, (1/3) * 0.5**j)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, label, mask, pmf in [(axes[0], "folded", folded, pmf_fold),
                             (axes[1], "aggregated", ~folded, pmf_agg)]:
    ax.hist(T[mask], bins=np.arange(0, 28) - 0.5, density=True,
            alpha=0.6, label="simulated")
    ax.plot(n, pmf, "ro-", label="exact")
    ax.set_title(f"Start at I, {label}")
    ax.set_xlabel("T")
    ax.set_ylabel("probability")
    ax.legend()

plt.tight_layout()
plt.savefig("HW4 problem 4.png", dpi=120)
plt.show()