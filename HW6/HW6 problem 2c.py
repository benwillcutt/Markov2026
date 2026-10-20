import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def h(x):
    return 0.5 + 0.5 * np.sign(x - 0.5) * np.sqrt(np.abs(2 * x - 1))


def load(path):
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    d = d[d["completed"] == 1].copy()
    d["hv"] = h((d["home_pts"] + 1) / (d["home_pts"] + d["away_pts"] + 2))
    return d


def build(games, teams, Pi, w):
    m = len(teams)
    ix = {t: k for k, t in enumerate(teams)}
    S = np.zeros((m, m))
    n = np.zeros(m)
    for _, g in games.iterrows():
        i, j = ix[g["home"]], ix[g["away"]]
        S[i, j] += g["hv"]
        S[j, i] += 1 - g["hv"]
        n[i] += 1
        n[j] += 1
    A = (S + w * Pi) / (n[:, None] + w * (m - 1))
    np.fill_diagonal(A, 0)
    return A


def perron(A, tol=1e-13):
    m = len(A)
    r = np.ones(m) / m
    for _ in range(100000):
        s = A @ r
        s = s / s.sum()
        if np.abs(s - r).sum() < tol:
            r = s
            break
        r = s
    lam = (A @ r).sum() / r.sum()
    ev = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
    return r, lam, ev[1] / ev[0]


def prior_last(prev, teams):
    ix = {t: k for k, t in enumerate(teams)}
    m = len(teams)
    s = np.zeros((m, m))
    c = np.zeros((m, m))
    for _, g in prev.iterrows():
        i, j = ix[g["home"]], ix[g["away"]]
        s[i, j] += g["hv"]
        s[j, i] += 1 - g["hv"]
        c[i, j] += 1
        c[j, i] += 1
    return np.where(c > 0, s / np.maximum(c, 1), 0.5)


def prior_strength(strength):
    s = np.asarray(strength, float)
    return s[:, None] / (s[:, None] + s[None, :])


def strength_from_season(prev, teams):
    ix = {t: k for k, t in enumerate(teams)}
    tot = np.zeros(len(teams))
    n = np.zeros(len(teams))
    for _, g in prev.iterrows():
        i, j = ix[g["home"]], ix[g["away"]]
        tot[i] += g["hv"]
        tot[j] += 1 - g["hv"]
        n[i] += 1
        n[j] += 1
    return tot / np.maximum(n, 1)


def run(path, season=2026, w=1.0, upto_week=None, poll=None):
    d = load(path)
    teams = sorted(set(d["home"]) | set(d["away"]))
    cur = d[d["season"] == season]
    if upto_week is not None:
        cur = cur[cur["week"] <= upto_week]
    prev = d[d["season"] == season - 1]
    m = len(teams)
    flat = np.full((m, m), 0.5)
    last = prior_last(prev, teams)
    own = prior_strength(poll if poll is not None else strength_from_season(prev, teams))
    out = {}
    for name, Pi in [("flat", flat), ("last", last), ("own", own)]:
        A = build(cur, teams, Pi, w)
        r, lam, rho = perron(A)
        out[name] = dict(A=A, r=r, lam=lam, rho=rho)
    return teams, out


def predict(teams, r, games):
    ix = {t: k for k, t in enumerate(teams)}
    res = []
    for _, g in games.iterrows():
        a, b = r[ix[g["home"]]], r[ix[g["away"]]]
        res.append((g["home"], g["away"], a / (a + b)))
    return res


if __name__ == "__main__":
    path = "HW6/big12.csv"
    teams, out = run(path)
    for k, v in out.items():
        print(k, round(v["lam"], 4), round(v["rho"], 4))

d = load("HW6/big12.csv")
teams, out = run("HW6/big12.csv")
raw = pd.read_csv("HW6/big12.csv")
wk6 = raw[(raw.season == 2026) & (raw.week == 6)]

for k, v in out.items():
    print(k, "lam*=%.5f" % v["lam"], "|l2/l*|=%.5f" % v["rho"])

R = pd.DataFrame({k: v["r"] for k, v in out.items()}, index=teams)
print((R * 100).round(2).sort_values("flat", ascending=False))

A = out["flat"]["A"]
print("row sums min/max", A.sum(1).min().round(4), A.sum(1).max().round(4), "lam", out["flat"]["lam"].round(4))

for k, v in out.items():
    p = predict(teams, v["r"], wk6)
    print(k)
    for a, b, q in p:
        print("  %-15s over %-12s %s  r-share %.3f" % (a if q > .5 else b, b if q > .5 else a, "(home)" if q > .5 else "(away)", max(q, 1 - q)))

fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
order = R["flat"].sort_values().index
for a, k in zip(ax, ["flat", "last", "own"]):
    a.barh(order, R.loc[order, k] * 100, color="#4C78A8")
    a.set_title("prior: " + k)
    a.set_xlabel("r (%)")
plt.tight_layout()
plt.savefig("HW6/HW6 problem 2c.png", dpi=130)

w0 = build(d[d.season == 2026], teams, np.full((16, 16), .5), 0.0)
print("w=0 rowsums zero?", (w0.sum(1) == 0).sum(), "games per team min", min(((d[d.season==2026].home==t)|(d[d.season==2026].away==t)).sum() for t in teams))