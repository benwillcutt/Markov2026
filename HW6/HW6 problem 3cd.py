import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded


def h1_exact(N, K, R0):
    k = np.arange(1, K)
    rho = np.concatenate(([1.0], np.cumprod(N / (R0 * (N - k)))))
    return 1.0 / rho.sum()


def rates(N, R0, g, dt, K):
    n = np.arange(1, K)
    b = R0 * g / N
    return b * n * (N - n) * dt, g * n * dt


def expected_T(N, K, R0, g, dt):
    p, q = rates(N, R0, g, dt, K)
    M = K - 1
    ab = np.zeros((3, M))
    ab[1] = p + q
    ab[0, 1:] = -p[:-1]
    ab[2, :-1] = -q[1:]
    return solve_banded((1, 1), ab, np.ones(M))


def simulate(N, K, R0, g, dt, runs, seed=0):
    rng = np.random.default_rng(seed)
    p, q = rates(N, R0, g, dt, K)
    P = np.concatenate(([0], p, [0]))
    Q = np.concatenate(([0], q, [0]))
    x = np.ones(runs, int)
    steps = np.zeros(runs, int)
    live = np.ones(runs, bool)
    while live.any():
        u = rng.random(runs)
        up = u < P[x]
        dn = (u >= P[x]) & (u < P[x] + Q[x])
        x = x + up - dn
        steps += live
        live = (x > 0) & (x < K)
    return steps.mean() * dt, steps.std() * dt / np.sqrt(runs)


if __name__ == "__main__":
    R = np.linspace(0.5, 3, 251)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(R, np.maximum(0, 1 - 1 / R), "--", color="#52514e", lw=2, label="max(0, 1 − 1/R₀)")
    for N, c in [(100, "#2a78d6"), (1000, "#eb6834")]:
        ax.plot(R, [h1_exact(N, N // 10, r) for r in R], color=c, lw=2, label=f"exact, N = {N}, K = N/10")
    ax.set_xlabel("R₀")
    ax.set_ylabel("h₁ = P₁(reach K before 0)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig("HW6/HW6 problem 3c.png", dpi=140)

    for N, K in [(100, 10), (1000, 100)]:
        print(N, K, [round(h1_exact(N, K, r), 4) for r in (0.5, 0.8, 1.0, 1.5, 3)], [round(max(0, 1 - 1 / r), 4) for r in (0.5, 0.8, 1.0, 1.5, 3)])

    for r in (0.5, 0.8, 1.5, 3):
        print("R0", r, "e1*dt", [round(expected_T(100, 10, r, 1, dt)[0] * dt, 6) for dt in (0.005, 0.001, 0.0001)])
    for r in (1.5, 0.8):
        print("sim", r, simulate(100, 10, r, 1, 0.005, 200000))