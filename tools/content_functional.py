"""
Is the functional f -> int_{C_{n,r}} f(eps) a content-evaluation functional,
f -> sum_{lambda |- n} w_lambda f(contents of lambda minus one 0)?

At r = 2 it is, with the Frobenius weights dim(lambda) chi^lambda(n-cycle)/n!
(the plane-sector theorem).  At r = 3 the exact Chern tables determine the whole
functional (every packet with max M <= 2), and this script tests whether any
weights fit.  See report/ballot.tex, "Where a proof has to go".

    python tools/content_functional.py d r
"""

import json, sys
from fractions import Fraction
from itertools import permutations
from collections import Counter
from chernpp import boxes


def partitions(n, top=None):
    top = n if top is None else top
    if n == 0:
        yield ()
        return
    for k in range(min(n, top), 0, -1):
        for r in partitions(n - k, k):
            yield (k,) + r


def contents(lam):
    return [j - i for i, row in enumerate(lam) for j in range(row)]


def m_eval(nu, xs):
    # monomial symmetric function m_nu at the multiset xs (len(xs) = d), distinct assignments
    d = len(xs)
    nu = list(nu) + [0] * (d - len(nu))
    tot = 0
    for perm in set(permutations(nu)):
        t = 1
        for e, x in zip(perm, xs):
            t *= x**e
        tot += t
    return tot


def solve(A, b):
    M = [[Fraction(x) for x in r] + [Fraction(y)] for r, y in zip(A, b)]
    n = len(A[0])
    piv = []
    r = 0
    for c in range(n):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = 1 / M[r][c]
        M[r] = [v * inv for v in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * bb for a, bb in zip(M[i], M[r])]
        piv.append(c)
        r += 1
    if any(all(v == 0 for v in M[i][:n]) and M[i][n] != 0 for i in range(r, len(M))):
        return None
    x = [Fraction(0)] * n
    for i, c in enumerate(piv):
        x[c] = M[i][n]
    return x, n - len(piv)


def functional_data(d, r, table):
    # I_r(m_nu) for nu = (r-1) - M over packets M with max M <= r-1 (valid reduction to C_{n,r})
    rows = []
    for M, c in table.items():
        if max(M) <= r - 1:
            nu = tuple(sorted((r - 1 - a for a in M), reverse=True))
            rows.append((nu, c))
    return rows


d = int(sys.argv[1])
r = int(sys.argv[2])
table = boxes.chern_table_exact(boxes.level_box(d, 8 if d <= 5 else 6))
data = functional_data(d, r, table)
n = d + 1
lams = list(partitions(n))
pts = []
for lam in lams:
    c = contents(lam)
    c.remove(0)
    pts.append(c)
A = [[m_eval(nu, pt) for pt in pts] for nu, _ in data]
b = [c for _, c in data]
res = solve(A, b)
print(
    f"d={d} r={r}: equations {len(data)}, unknowns {len(lams)}:",
    "INCONSISTENT" if res is None else f"consistent, free={res[1]}",
)
if res:
    for lam, w in zip(lams, res[0]):
        print("  ", lam, w)

from math import factorial


def hook_dim(lam):
    n = sum(lam)
    conj = [sum(1 for r in lam if r > j) for j in range(lam[0])]
    h = 1
    for i, row in enumerate(lam):
        for j in range(row):
            h *= row - j + conj[j] - i - 1
    return factorial(n) // h


def chi_ncycle(lam):
    n = sum(lam)
    if len(lam) > 1 and lam[1] > 1:
        return 0  # not a hook
    return (-1) ** (len(lam) - 1)


w = [Fraction(hook_dim(l) * chi_ncycle(l), factorial(n)) for l in lams]
ok = all(sum(wi * a for wi, a in zip(w, row)) == bb for row, bb in zip(A, b))
print("Frobenius weights satisfy the equations:", ok)
