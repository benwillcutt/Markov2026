import numpy as np
import matplotlib.pyplot as plt
analytic = 23 * np.pi / 192

def mc_estimate(N, rng):
    X = rng.random(N)
    Y = rng.random(N)
    Z = rng.random(N)
    indicator = (X**2 + Y**2 < Z) & (Z**2 > X * Y)
    p_hat = indicator.mean()
    se = np.sqrt(p_hat * (1 - p_hat) / N)
    return p_hat, se


rng = np.random.default_rng(12345)
Ns = np.unique(np.logspace(2, 7, 60).astype(int))

estimates = np.zeros(len(Ns))
stderrs   = np.zeros(len(Ns))

for i, N in enumerate(Ns):
    estimates[i], stderrs[i] = mc_estimate(N, rng)

print(f"Analytic value:        {analytic:.6f}")
print(f"MC estimate (N={Ns[-1]:,}): {estimates[-1]:.6f}  (SE = {stderrs[-1]:.6f})")
print(f"Difference:             {estimates[-1]-analytic:.6f}  "
      f"({(estimates[-1]-analytic)/stderrs[-1]:.2f} SE)")

plt.figure(figsize=(9, 5.5))
plt.plot(Ns, estimates, 'o-', markersize=3, linewidth=0.8,
         color='#2b6cb0', label='Monte Carlo estimate')
plt.axhline(analytic, color='crimson', linestyle='--', linewidth=1.8,
            label=f'Analytic value = 23π/192 ≈ {analytic:.4f}')
plt.xscale('log')
plt.xlabel('Sample size N (log scale)')
plt.ylabel('Estimated probability')
plt.title(r'Monte Carlo estimate of $P(X^2+Y^2<Z,\ Z^2>XY)$ vs sample size')
plt.legend(loc='upper right')
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('HW1 problem6 figure.png', dpi=150)
plt.show()