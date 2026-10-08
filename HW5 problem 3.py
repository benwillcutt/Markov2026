import numpy as np
import matplotlib.pyplot as plt

links = {1: [2, 3], 2: [3, 4], 3: [1, 5], 4: [1, 3, 5],
         5: [2, 6, 7], 6: [6], 7: [8], 8: [7]}
m = 8

p = np.zeros((m, m))
for i, outs in links.items():
    for j in outs:
        p[i - 1, j - 1] = 1 / len(outs)

np.set_printoptions(precision=6, suppress=True, linewidth=120)
print("p =")
print(p)

reach = np.linalg.matrix_power(np.eye(m) + (p > 0), m) > 0
seen = set()
for i in range(m):
    cls = frozenset(j for j in range(m) if reach[i, j] and reach[j, i])
    if cls in seen:
        continue
    seen.add(cls)
    closed = all(reach[j, k] <= (k in cls) for j in cls for k in range(m))
    print("class", sorted(c + 1 for c in cls), "closed" if closed else "not closed")

rank = np.linalg.matrix_rank(p - np.eye(m))
print("dimension of stationary space:", m - rank)

q0 = np.ones(m) / m
q100 = q0 @ np.linalg.matrix_power(p, 100)
q101 = q100 @ p
print("q100 =", q100)
print("q101 =", q101)

d = 0.85
G = d * p + (1 - d) / m * np.ones((m, m))

q = q0.copy()
iters = 0
while True:
    q_next = q @ G
    iters += 1
    done = np.abs(q_next - q).sum() < 1e-10
    q = q_next
    if done:
        break
pi_power = q

A = np.vstack([G.T - np.eye(m), np.ones(m)])
rhs = np.zeros(m + 1)
rhs[-1] = 1
pi = np.linalg.lstsq(A, rhs, rcond=None)[0]

print("iterations:", iters)
print("pi (power)      =", pi_power)
print("pi (linear)     =", pi)
print("L1 difference   =", np.abs(pi_power - pi).sum())
order = np.argsort(-pi, kind="stable")
print("ranking:", [int(i) + 1 for i in order])
print("pi_1 - pi_5 =", pi[0] - pi[4])
print("bound d^n * 2 at n =", iters, ":", 2 * d**iters)

rng = np.random.default_rng(2024)
R, T = 100, 100_000
cumG = np.cumsum(G, axis=1)
checkpoints = np.unique(np.round(np.logspace(2, 5, 31)).astype(int))
cp_set = set(checkpoints.tolist())

state = np.zeros(R, dtype=int)
counts = np.zeros((R, m))
rows = np.arange(R)
rms = []

for t in range(1, T + 1):
    u = rng.random(R)
    state = (u[:, None] > cumG[state]).sum(axis=1)
    state = np.minimum(state, m - 1)
    counts[rows, state] += 1
    if t in cp_set:
        est = counts / t
        rms.append(np.sqrt(np.mean(np.abs(est - pi).max(axis=1) ** 2)))

rms = np.array(rms)
est = counts / T

slope, intercept = np.polyfit(np.log10(checkpoints), np.log10(rms), 1)
print("fitted slope:", slope)
print("rms of max error at T = 1e5:", rms[-1])

iid = np.sqrt(pi[5] * (1 - pi[5]) / T)
print("sqrt(pi6 (1-pi6)/T):", iid, " ratio:", rms[-1] / iid)

Z = np.linalg.inv(np.eye(m) - G + np.outer(np.ones(m), pi))
sigma2 = np.zeros(m)
for i in range(m):
    f = np.zeros(m)
    f[i] = 1
    fbar = f - pi[i]
    sigma2[i] = 2 * np.sum(pi * fbar * (Z @ fbar)) - np.sum(pi * fbar**2)

per_page_obs = np.sqrt(np.mean((est - pi) ** 2, axis=0))
per_page_iid = np.sqrt(pi * (1 - pi) / T)
per_page_clt = np.sqrt(sigma2 / T)
print("per-page observed rms :", per_page_obs)
print("per-page iid formula  :", per_page_iid)
print("per-page Markov CLT   :", per_page_clt)
print("variance inflation sigma2 / (pi(1-pi)):", sigma2 / (pi * (1 - pi)))

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(1, m + 1)
ax.bar(x - 0.2, pi, 0.4, label="stationary pi")
ax.bar(x + 0.2, est[0], 0.4, label="one surfer, T = 1e5")
ax.set_xlabel("page")
ax.set_ylabel("probability")
ax.set_title("PageRank, d = 0.85")
ax.legend()
plt.tight_layout()
plt.savefig("HW5 problem 3.bars.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(checkpoints, rms, "o-", ms=4, label="rms over surfers of max error")
ax.loglog(checkpoints, 10**intercept * checkpoints**slope, "k--", label=f"fit, slope {slope:.3f}")
ax.loglog(checkpoints, np.sqrt(pi[5] * (1 - pi[5]) / checkpoints), ":", label="sqrt(pi6 (1 - pi6) / T)")
ax.set_xlabel("T")
ax.set_ylabel("error")
ax.legend()
plt.tight_layout()
plt.savefig("HW5 problem 3.error.png", dpi=150)