import numpy as np
from math import gcd, pi
from functools import reduce


def chain(m, t=2):
    P = np.zeros((m, m))
    for i in range(m - 1):
        P[i, i + 1] = 1
    P[m - 1, 0] += 0.5
    P[m - 1, t - 1] += 0.5
    return P


def exponent(P):
    Bp = (P > 0).astype(int)
    B = Bp.copy()
    M = 1
    while not B.all():
        B = ((B @ Bp) > 0).astype(int)
        M += 1
    return M


def stationary(P):
    w, v = np.linalg.eig(P.T)
    k = np.argmin(abs(w - 1))
    x = np.real(v[:, k])
    return x / x.sum()


def return_times(P, s, N):
    Bp = (P > 0).astype(int)
    B = Bp.copy()
    out = []
    for n in range(1, N + 1):
        if B[s, s]:
            out.append(n)
        B = ((B @ Bp) > 0).astype(int)
    return out


def mix_time(P, eps):
    p = stationary(P)
    Q = np.eye(len(P))
    n = 0
    while 0.5 * np.abs(Q - p).sum(axis=1).max() >= eps:
        Q = Q @ P
        n += 1
    return n


P4 = chain(4)
print("pi", stationary(P4) * 7)
print("R(1)", return_times(P4, 0, 20))
print("R(4)", return_times(P4, 3, 20))

print("part b")
for m in range(3, 13):
    P = chain(m, 3)
    R = return_times(P, 0, 4 * m)
    d = reduce(gcd, R)
    w = np.linalg.eigvals(P)
    u = np.sort_complex(np.round(w[abs(abs(w) - 1) < 1e-8], 8))
    print(m, d, u)

print("part c")
for m in range(3, 13):
    P = chain(m)
    M = exponent(P)
    w = np.linalg.eigvals(P)
    w = w[np.argsort(-abs(w))]
    l2 = abs(w[1])
    print(m, M, (m - 1) ** 2 + 1, round(l2, 6), round(1 - l2, 6), round(pi ** 2 / (2 * m ** 3), 6), mix_time(P, 0.01))