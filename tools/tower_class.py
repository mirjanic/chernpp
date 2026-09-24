"""
Compare the rank-one nonassociative tower point class P*_d with the stored Q_d.

    P*_d = prod_{k=4..d} prod_{2 <= i <= j, i + j <= k} (z_1 + z_{i-1} + z_j - z_k)

Prints, for d = 4..7, whether Q_d - P*_d is residue-null (packet sums unchanged).
See report/q8_feasibility.tex.

    python tools/tower_class.py
"""

from chernpp.artifacts import load_algebra
from chernpp.optimisation import gauge
from chernpp.polynomial import poly_mul


def linear(d, coefficients):
    out = {}
    for index, c in coefficients:
        e = [0] * d
        e[index - 1] += 1
        out[tuple(e)] = out.get(tuple(e), 0) + c
    return {k: v for k, v in out.items() if v}


def tower_class(d):
    P = {tuple([0] * d): 1}
    for k in range(4, d + 1):
        for i in range(2, k):
            for j in range(i, k):
                if i + j <= k:
                    P = poly_mul(P, linear(d, [(1, 1), (i - 1, 1), (j, 1), (k, -1)]))
    return P


def main():
    for d, truncation in ((4, 12), (5, 14), (6, 12), (7, 10)):
        alg = load_algebra(d)
        g = gauge.setup(d, 4)
        Q = gauge.to_z(alg.multidegree, d, g.degree)
        P = tower_class(d)
        diff = {k: Q.get(k, 0) - P.get(k, 0) for k in set(Q) | set(P)}
        diff = {k: v for k, v in diff.items() if v}
        if not diff:
            print(f"d = {d}: P* = Q exactly")
            continue
        v = gauge.validate_gauge(diff, d, truncation)
        print(f"d = {d}: Q - P* has {len(diff)} terms; {v.packets_changed} of {v.packets} packet sums change")


if __name__ == "__main__":
    main()
