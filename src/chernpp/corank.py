"""
The corank filtration of Thom polynomials.

Write Thom polynomials in the relative Chern classes and read ``c_k`` as the
complete homogeneous function ``h_k``, so that ``s_nu(c) = det(c_{nu_i + j - i})``.
This is the convention in which Pragacz--Weber positivity holds: every Thom
polynomial is a nonnegative combination of Schur polynomials.  Let

    C_r = { nonnegative combinations of products s_{nu^1} ... s_{nu^k},
            every nu^i with at most r rows }.

``C_1`` is the Chern-monomial cone (``s_(k) = c_k``), so Rimányi's conjecture reads
``Tp_{A_d} in C_1``; ``C_infinity`` is the Schur-positive cone.  The filtration

    C_1  <  C_2  <  C_3  <  ...  <  C_infinity

interpolates between the two.  The invariant this module computes is

    rho(Tp_eta) = min { r : Tp_eta in C_r },

and Rimányi's conjecture is exactly ``rho(Tp_{A_d}) = 1``.

*Lower bound (a theorem).*  If every Schur component of ``Tp_eta`` has at least
``r`` rows then ``rho >= r``: every nonzero element of ``C_{r-1}`` has a positive
coefficient on some Schur function with at most ``r - 1`` rows -- the row-sum
partition of each product occurs with LR coefficient 1 and nothing can cancel
it.  A corank-``r`` singularity has every component containing ``(l + r)^r``,
so ``rho >= corank``.  The first instance is a one-line consequence: the sum of
the Chern coefficients equals the coefficient of the one-row Schur function
(every Kostka number ``K_{(n), mu}`` is 1), which is ``(d!)^L`` for ``A_d`` and
0 for every singularity of corank at least two -- no such Thom polynomial can
be Chern-positive.

*The naive upper bound is false.*  ``rho = corank`` fails on the registry:
``Tp_{I_{2,4}}`` is not in ``C_2`` (certificate ``[s_33] + [s_222] - [s_321]``,
value ``4 + 5 - 12 < 0``) and ``Tp_{B_{5,3}}`` at ``l = 1`` needs ``r = 4``.
The survey records ``rho`` for every table (``results/corank_survey.json``).

Everything here is exact.  Membership answers come with certificates that are
re-verified in rational arithmetic before being returned: a nonnegative rational
decomposition when ``Tp`` is in the cone, a Farkas vector when it is not.  The
linear programme is only a search heuristic, as in :mod:`chernpp.certificates`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from itertools import permutations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Partition = Tuple[int, ...]
SchurPoly = Dict[Partition, int]

PUBLISHED = Path(__file__).resolve().parents[2] / "tests" / "data" / "published_thom_polynomials.json"


# --------------------------------------------------------------------------
# symmetric functions: Pieri, Jacobi--Trudi, Kostka
# --------------------------------------------------------------------------


def _horizontal_strips(shape: Partition, m: int) -> Iterable[Partition]:
    """All partitions obtained from ``shape`` by adding a horizontal strip of size ``m``."""
    rows = list(shape) + [0]

    def extend(i: int, remaining: int, acc: List[int]):
        if i == len(rows):
            if remaining == 0:
                yield tuple(x for x in acc if x > 0)
            return
        cap = remaining if i == 0 else min(remaining, rows[i - 1] - rows[i])
        for add in range(cap, -1, -1):
            yield from extend(i + 1, remaining - add, acc + [rows[i] + add])

    yield from extend(0, m, [])


def pieri(expansion: SchurPoly, m: int) -> SchurPoly:
    """``expansion * h_m`` in the Schur basis (Pieri's rule)."""
    if m == 0:
        return dict(expansion)
    out: Dict[Partition, int] = {}
    for shape, c in expansion.items():
        for new in _horizontal_strips(shape, m):
            out[new] = out.get(new, 0) + c
    return {k: v for k, v in out.items() if v}


@lru_cache(maxsize=None)
def h_to_schur(mu: Partition) -> Tuple[Tuple[Partition, int], ...]:
    """``h_mu`` in the Schur basis, i.e. the Kostka numbers ``K_{lambda, mu}``."""
    mu = tuple(sorted((m for m in mu if m > 0), reverse=True))
    if not mu:
        return (((), 1),)
    head = dict(h_to_schur(mu[:-1]))
    return tuple(sorted(pieri(head, mu[-1]).items()))


def chern_to_schur(terms: Dict[Partition, int]) -> SchurPoly:
    """A polynomial in the Chern classes, ``{sorted indices: coefficient}``, in the Schur basis."""
    out: Dict[Partition, int] = {}
    for mu, c in terms.items():
        for lam, k in h_to_schur(tuple(sorted(mu, reverse=True))):
            out[lam] = out.get(lam, 0) + c * k
    return {k: v for k, v in out.items() if v}


def _perm_sign(p: Sequence[int]) -> int:
    sign, seen = 1, [False] * len(p)
    for i in range(len(p)):
        if seen[i]:
            continue
        j, length = i, 0
        while not seen[j]:
            seen[j] = True
            j = p[j]
            length += 1
        sign *= -1 if length % 2 == 0 else 1
    return sign


@lru_cache(maxsize=None)
def jacobi_trudi(nu: Partition) -> Tuple[Tuple[Partition, int], ...]:
    """``s_nu = det(h_{nu_i + j - i})`` as a combination of complete products ``h_mu``."""
    nu = tuple(x for x in nu if x > 0)
    k = len(nu)
    out: Dict[Partition, int] = {}
    for p in permutations(range(k)):
        parts = [nu[i] + p[i] - i for i in range(k)]
        if any(x < 0 for x in parts):
            continue
        mu = tuple(sorted((x for x in parts if x > 0), reverse=True))
        out[mu] = out.get(mu, 0) + _perm_sign(p)
    return tuple(sorted((m, c) for m, c in out.items() if c))


def times_schur(expansion: SchurPoly, nu: Partition) -> SchurPoly:
    """``expansion * s_nu`` in the Schur basis, through Jacobi--Trudi and Pieri."""
    out: Dict[Partition, int] = {}
    for mu, sign in jacobi_trudi(nu):
        cur = dict(expansion)
        for m in mu:
            cur = pieri(cur, m)
        for lam, c in cur.items():
            out[lam] = out.get(lam, 0) + sign * c
    return {k: v for k, v in out.items() if v}


def schur_product(factors: Sequence[Partition]) -> SchurPoly:
    """``s_{nu^1} ... s_{nu^k}`` in the Schur basis."""
    cur: SchurPoly = {(): 1}
    for nu in factors:
        cur = times_schur(cur, nu)
    return cur


def schur_to_chern(expansion: SchurPoly) -> Dict[Partition, int]:
    """The Chern-monomial (``h``) coefficients of a Schur combination."""
    out: Dict[Partition, int] = {}
    for lam, c in expansion.items():
        for mu, k in jacobi_trudi(lam):
            out[mu] = out.get(mu, 0) + c * k
    return {k: v for k, v in out.items() if v}


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Table:
    """One published Thom polynomial."""

    singularity: str
    relative_dimension: int
    codimension: int
    chern: Dict[Partition, int]

    @property
    def family(self) -> str:
        return self.singularity.split("_")[0]


def load_registry(path: Path = PUBLISHED) -> List[Table]:
    data = json.loads(Path(path).read_text())
    tables = []
    for t in data["tables"]:
        chern: Dict[Partition, int] = {}
        for term in t["terms"]:
            key = tuple(sorted(term["chern_indices"], reverse=True))
            chern[key] = chern.get(key, 0) + int(term["coefficient"])
        tables.append(Table(t["singularity"], int(t["relative_dimension"]), int(t["codimension"]), chern))
    return tables


#: Corank read off the local algebra, where the registry's name determines it.
#: A_d = C[x]/(x^{d+1}); I, III and the two named algebras are quotients of
#: C[x, y]; the ``cup`` entry is the open part of Sigma^3.  The B and C families
#: are not described in the registry's machine-readable form, so their corank
#: is inferred from the data (see :func:`support_corank`) and flagged as such.
def algebra_corank(table: Table) -> Optional[int]:
    name = table.singularity
    if name.startswith("A_"):
        return 1 if name != "A_0" else 0
    if name.startswith(("I_", "III_", "(x^2")):
        return 2
    if name.startswith("cup"):
        return 3
    return None


def support_corank(expansion: SchurPoly, relative_dimension: int) -> int:
    """
    The largest ``r`` with every Schur component containing the rectangle ``(l + r)^r``.

    A corank-``r`` singularity lies in ``Sigma^r``, whose Thom polynomials are
    supported on partitions containing that rectangle; so this is an upper
    bound for the corank that the data can confirm.
    """
    r = 0
    while all(len(lam) > r and lam[r] >= relative_dimension + r + 1 for lam in expansion):
        r += 1
    return r


def row_count_floor(expansion: SchurPoly) -> int:
    """Fewest rows among the Schur components -- the obstruction to ``C_{r-1}``."""
    return min(len(lam) for lam in expansion) if expansion else 0


# --------------------------------------------------------------------------
# cone membership with exact certificates
# --------------------------------------------------------------------------


def _vector_decompositions(mu: Partition, r: int) -> Iterable[Tuple[Partition, ...]]:
    """
    Multisets of nonzero partitions with at most ``r`` rows summing, row by row, to ``mu``.

    These index the products whose row-sum partition is ``mu``.  Parts are
    produced in a canonical nonincreasing order so each multiset appears once.
    """
    mu = tuple(mu) + (0,) * (r - len(mu))

    def parts_below(rest: Tuple[int, ...], bound: Optional[Tuple[int, ...]]):
        # nonzero partitions nu (as r-vectors) with nu <= rest componentwise, nu <= bound in lex
        def rec(i: int, prev: int, acc: List[int]):
            if i == r:
                nu = tuple(acc)
                if any(nu) and (bound is None or nu <= bound):
                    yield nu
                return
            for v in range(min(prev, rest[i]), -1, -1):
                yield from rec(i + 1, v, acc + [v])

        yield from rec(0, rest[0] if r else 0, [])

    def rec(rest: Tuple[int, ...], bound, acc):
        if not any(rest):
            yield tuple(tuple(x for x in nu if x > 0) for nu in acc)
            return
        for nu in parts_below(rest, bound):
            left = tuple(a - b for a, b in zip(rest, nu))
            yield from rec(left, nu, acc + [nu])

    yield from rec(mu, None, [])


@dataclass
class Membership:
    """Whether a Thom polynomial lies in ``C_r``, with an exact certificate either way."""

    r: int
    member: bool
    #: Nonnegative rational weights on products of <= r-row Schur functions.
    decomposition: Dict[Tuple[Partition, ...], Fraction] = field(default_factory=dict)
    #: Farkas vector on Schur functions: <y, g> >= 0 for every generator, <y, Tp> < 0.
    farkas: Dict[Partition, Fraction] = field(default_factory=dict)
    generators: int = 0
    note: str = ""


def _generators(target: SchurPoly, r: int, max_generators: int) -> Dict[Tuple[Partition, ...], SchurPoly]:
    """
    Every product of ``<= r``-row Schur functions that can occur in a decomposition.

    With nonnegative weights nothing cancels, so a usable product has Schur
    support inside ``supp(target)``; in particular its row-sum partition, which
    has at most ``r`` rows, lies in the support.  Enumerating through row sums is
    therefore complete, and far smaller than enumerating all products.
    """
    support = set(target)
    out: Dict[Tuple[Partition, ...], SchurPoly] = {}
    for mu in sorted(support):
        if len(mu) > r:
            continue
        for factors in _vector_decompositions(mu, r):
            if factors in out:
                continue
            product = schur_product(factors)
            if set(product) <= support:
                out[factors] = product
                if len(out) > max_generators:
                    raise RuntimeError(f"more than {max_generators} usable generators")
    return out


def _solve_exact(columns: List[SchurPoly], target: SchurPoly) -> Optional[List[Fraction]]:
    """Exact solution of ``sum x_j columns_j = target`` (least-squares-free), or None."""
    rows = sorted(set(target).union(*[set(c) for c in columns]))
    index = {k: i for i, k in enumerate(rows)}
    n = len(columns)
    matrix = [[Fraction(0)] * (n + 1) for _ in rows]
    for j, col in enumerate(columns):
        for k, v in col.items():
            matrix[index[k]][j] = Fraction(v)
    for k, v in target.items():
        matrix[index[k]][n] = Fraction(v)
    pivots, row = [], 0
    for col in range(n):
        piv = next((i for i in range(row, len(rows)) if matrix[i][col] != 0), None)
        if piv is None:
            continue
        matrix[row], matrix[piv] = matrix[piv], matrix[row]
        inv = 1 / matrix[row][col]
        matrix[row] = [x * inv for x in matrix[row]]
        for i in range(len(rows)):
            if i != row and matrix[i][col] != 0:
                f = matrix[i][col]
                matrix[i] = [a - f * b for a, b in zip(matrix[i], matrix[row])]
        pivots.append(col)
        row += 1
    if any(all(x == 0 for x in matrix[i][:n]) and matrix[i][n] != 0 for i in range(row, len(rows))):
        return None
    x = [Fraction(0)] * n
    for i, col in enumerate(pivots):
        x[col] = matrix[i][n]
    return x


def membership(target: SchurPoly, r: int, max_generators: int = 200000) -> Membership:
    """
    Decide ``target in C_r`` exactly.

    A float LP (HiGHS) proposes either a decomposition or a Farkas vector; the
    proposal is then rebuilt and checked in exact rational arithmetic, and only
    a verified certificate is returned.  A proposal that fails verification
    raises rather than being reported either way.
    """
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import csc_matrix

    if not target:
        return Membership(r, True, note="zero polynomial")
    gens = _generators(target, r, max_generators)
    rows = sorted(target)
    index = {k: i for i, k in enumerate(rows)}
    keys = list(gens)
    if keys:
        data, ri, ci = [], [], []
        for j, key in enumerate(keys):
            for lam, v in gens[key].items():
                data.append(float(v))
                ri.append(index[lam])
                ci.append(j)
        A = csc_matrix((data, (ri, ci)), shape=(len(rows), len(keys)))
        b = np.array([float(target[k]) for k in rows])
        scale = max(1.0, float(np.abs(b).max()))
        res = linprog(np.zeros(len(keys)), A_eq=A / scale, b_eq=b / scale, bounds=(0, None), method="highs")
    else:
        res = None

    if res is not None and res.status == 0:
        support = [j for j, v in enumerate(res.x) if v > 1e-9 * max(1.0, float(res.x.max()))]
        weights = _solve_exact([gens[keys[j]] for j in support], target)
        if weights is not None and all(w >= 0 for w in weights):
            dec = {keys[j]: w for j, w in zip(support, weights) if w != 0}
            _verify_decomposition(dec, target)
            return Membership(r, True, decomposition=dec, generators=len(keys))
        # fall through: the float basis did not rationalise; try the full exact route below
        raise ArithmeticError(
            f"C_{r}: LP reported feasible but no exact nonnegative decomposition was recovered"
        )

    # Infeasible (or no generators): a Farkas vector y with y.g >= 0 for all g, y.target < 0.
    farkas = _farkas(target, [gens[k] for k in keys], rows)
    if farkas is None:
        raise ArithmeticError(f"C_{r}: LP infeasible but no exact Farkas certificate was found")
    return Membership(r, False, farkas=farkas, generators=len(keys))


def _verify_decomposition(dec: Dict[Tuple[Partition, ...], Fraction], target: SchurPoly) -> None:
    total: Dict[Partition, Fraction] = {}
    for factors, w in dec.items():
        if w < 0:
            raise ArithmeticError("negative weight in a decomposition")
        if any(len(nu) == 0 for nu in factors):
            raise ArithmeticError("empty factor")
        for lam, v in schur_product(factors).items():
            total[lam] = total.get(lam, 0) + w * v
    total = {k: v for k, v in total.items() if v}
    if total != {k: Fraction(v) for k, v in target.items()}:
        raise ArithmeticError("decomposition does not reproduce the target")


def _farkas(
    target: SchurPoly, columns: List[SchurPoly], rows: List[Partition]
) -> Optional[Dict[Partition, Fraction]]:
    """
    Search for ``y`` with ``<y, g> >= 0`` for every column and ``<y, target> = -1``.

    The coordinates of ``y`` range over *all* partitions of the degree: those
    outside ``supp(target)`` are exactly where excluded generators would put
    mass, so the certificate below is checked only against the listed columns
    and is valid because every other product has support outside ``supp(target)``
    and is therefore unusable.  Rows are restricted to ``supp(target)``; a
    product meeting a partition outside it can never appear, which is the
    completeness argument of :func:`_generators`.
    """
    import numpy as np
    from scipy.optimize import linprog

    index = {k: i for i, k in enumerate(rows)}
    n = len(rows)
    t = np.array([float(target[k]) for k in rows])
    if columns:
        G = np.zeros((len(columns), n))
        for j, col in enumerate(columns):
            for lam, v in col.items():
                G[j, index[lam]] = float(v)
        A_ub = -G
        b_ub = np.zeros(len(columns))
    else:
        A_ub, b_ub = None, None
    res = linprog(
        np.zeros(n),
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=t.reshape(1, -1),
        b_eq=[-1.0],
        bounds=(None, None),
        method="highs",
    )
    if res.status != 0:
        return None
    y = {rows[i]: Fraction(v).limit_denominator(10**6) for i, v in enumerate(res.x) if abs(v) > 1e-12}
    if _is_farkas(y, target, columns):
        return y
    # Rounding did not verify.  Re-solve a bounded version whose optimum is a
    # vertex, and recover that vertex exactly from its tight constraints.
    return _farkas_vertex(target, columns, rows, t, G if columns else np.zeros((0, n)))


def _is_farkas(y: Dict[Partition, Fraction], target: SchurPoly, columns: List[SchurPoly]) -> bool:
    if sum(y.get(k, 0) * v for k, v in target.items()) >= 0:
        return False
    return all(sum(y.get(k, 0) * v for k, v in col.items()) >= 0 for col in columns)


def _farkas_vertex(target, columns, rows, t, G) -> Optional[Dict[Partition, Fraction]]:
    """
    Minimise <y, target> over {G y >= 0, -1 <= y <= 1} with the dual simplex, so the
    optimum is a vertex; read off its tight constraints and solve them exactly.
    The float solution only selects which constraints are tight -- the returned
    vector is an exact rational solution, and it is verified before it is returned.
    """
    import numpy as np
    from scipy.optimize import linprog

    n = len(rows)
    res = linprog(
        t,
        A_ub=-G if len(G) else None,
        b_ub=np.zeros(len(G)) if len(G) else None,
        bounds=[(-1, 1)] * n,
        method="highs-ds",
    )
    if res.status != 0 or res.fun >= -1e-9:
        return None
    x = res.x
    scale = max(1.0, float(np.abs(G).max()) if len(G) else 1.0)
    equations: List[Tuple[List[Fraction], Fraction]] = []
    for j in range(len(G)):
        if abs(float(G[j] @ x)) <= 1e-9 * scale * n:
            equations.append(([Fraction(int(round(v))) for v in G[j]], Fraction(0)))
    for i in range(n):
        if abs(x[i] - 1) <= 1e-9:
            equations.append(([Fraction(int(k == i)) for k in range(n)], Fraction(1)))
        elif abs(x[i] + 1) <= 1e-9:
            equations.append(([Fraction(int(k == i)) for k in range(n)], Fraction(-1)))
    exact = _solve_square(equations, n)
    if exact is None:
        return None
    y = {rows[i]: v for i, v in enumerate(exact) if v != 0}
    return y if _is_farkas(y, target, columns) else None


def _solve_square(equations, n) -> Optional[List[Fraction]]:
    """Unique solution of a consistent system of rank ``n`` (extra rows must agree), else None."""
    matrix = [row[:] + [rhs] for row, rhs in equations]
    pivots, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, len(matrix)) if matrix[i][c] != 0), None)
        if p is None:
            return None  # not a vertex: rank deficient
        matrix[r], matrix[p] = matrix[p], matrix[r]
        inv = 1 / matrix[r][c]
        matrix[r] = [v * inv for v in matrix[r]]
        for i in range(len(matrix)):
            if i != r and matrix[i][c] != 0:
                f = matrix[i][c]
                matrix[i] = [a - f * b for a, b in zip(matrix[i], matrix[r])]
        pivots.append(c)
        r += 1
    if any(matrix[i][n] != 0 for i in range(r, len(matrix))):
        return None
    return [matrix[i][n] for i in range(n)]


# --------------------------------------------------------------------------
# the survey
# --------------------------------------------------------------------------


@dataclass
class Survey:
    singularity: str
    relative_dimension: int
    codimension: int
    terms: int
    chern_negative: int
    schur_negative: int
    one_row: int
    chern_sum: int
    algebra_corank: Optional[int]
    support_corank: int
    row_floor: int
    memberships: Dict[int, bool] = field(default_factory=dict)
    generators: Dict[int, int] = field(default_factory=dict)


def survey_table(table: Table, ranks: Sequence[int] = (1, 2, 3), max_generators: int = 200000) -> Survey:
    schur = chern_to_schur(table.chern)
    n = table.codimension
    s = Survey(
        singularity=table.singularity,
        relative_dimension=table.relative_dimension,
        codimension=n,
        terms=len(table.chern),
        chern_negative=sum(1 for v in table.chern.values() if v < 0),
        schur_negative=sum(1 for v in schur.values() if v < 0),
        one_row=schur.get((n,), 0),
        chern_sum=sum(table.chern.values()),
        algebra_corank=algebra_corank(table),
        support_corank=support_corank(schur, table.relative_dimension),
        row_floor=row_count_floor(schur),
    )
    for r in ranks:
        if any(s.memberships.values()):
            s.memberships[r] = True  # C_r grows with r
            continue
        if r == 1:
            s.memberships[1] = s.chern_negative == 0
            continue
        if r < s.row_floor:
            s.memberships[r] = False  # the sharpness theorem: no Schur term with <= r rows
            continue
        m = membership(schur, r, max_generators=max_generators)
        s.memberships[r] = m.member
        s.generators[r] = m.generators
    return s
