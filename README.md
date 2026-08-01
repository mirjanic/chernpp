# Chern++

**Positive Chern Classes in Thom Polynomials**

Tools for the Thom polynomials of the Morin singularities $A_d$ ($d \le 6$), in the form given by
the Bérczi–Szenes residue formula, and for the positivity conjectures attached to them.

## The setting

Bérczi–Szenes express $\mathrm{Tp}_{A_d}$ as an iterated residue whose only $d$-dependent input is
$\mathcal{Q}_d$, the $T_d$-equivariant multidegree of the Borel orbit closure
$\mathcal{O}_d = \overline{B_d \cdot \epsilon_{\mathrm{ref}}}$ inside
$\widehat{N}_d \subset \mathrm{Hom}(\mathbb{C}^d, \mathrm{Sym}^2\mathbb{C}^d)$. Normalising
$z_d = 1$ and setting $x_j = z_j / z_{j+1}$, positivity is read off the **chamber series**

$$F_d(x) \;=\; \frac{\prod_{m<l}(1 - z_m/z_l)\,\mathcal{Q}_d}{\prod_{m+r\le l}(1 - z_m/z_l - z_r/z_l)}
\;=\; \frac{N_d(x)}{\prod_r\bigl(1 - f_r(x)\bigr)} \;=\; \sum_{\beta\ge 0} A_\beta x^\beta,$$

where every $f_r$ has nonnegative coefficients and zero constant term.

| conjecture | statement | status |
|---|---|---|
| **Rimányi**, weak | the coefficients of $\mathrm{Tp}_{A_d}$ in the relative Chern classes are $\ge 0$ | open; the target |
| **Bérczi–Szenes**, strong | $A_\beta \ge 0$ for every $\beta$ | true for $d=4$, false for $d \ge 5$ |

The strong statement implies the weak one because each Chern coefficient is a *sum* of several
$A_\beta$; the first negative Laurent coefficient, $A_{(1,1,2,1)} = -1$ for $d=5$, cancels inside a
Chern coefficient equal to $10$. By the $\ell$-free reduction (`papers/report.pdf`, Thm. 5) the
weak conjecture "for all relative dimensions $\ell$" collapses to the single $\ell$-free statement
$C(M) \ge 0$ over zero-sum multisets $M$ of $d$ integers.

## Results

Reproducible from a clean checkout:

- The classical Thom polynomials for $d = 4, 5, 6$, including all eleven $A_6$ coefficients.
- **Rimányi's conjecture for $A_6$**, verified at every relative dimension $\ell \le 7$, and for
  $A_5$ at $\ell \le 11$. Past $\ell = 5$ the coefficients outgrow `int64` (they reach
  $1.75\times10^{22}$ at $A_6$, $\ell = 7$), so these use CRT reconstruction over word-sized
  primes, verified against a prime held back from the reconstruction.
- **Agreement with Rimányi's published tables** at relative dimension 1 — all 15, 30 and 58
  coefficients for $A_4$, $A_5$, $A_6$. Those are computed by the restriction-equation method,
  mathematics independent of the residue formula implemented here, so this constrains
  $\mathcal{Q}_d$ externally rather than self-consistently.
- The **unpaired tail** ($i>j \Rightarrow A_{i,j,\dots}\ge 0$) and the **paired inequality**
  ($A_{i,j,\dots} + A_{j-i,j,\dots} \ge 0$) — the reduction that would settle $A_5$ — both hold
  for $A_6$ in the tested range.
- **Prefix positivity fails at $d=6$.** $F_5/(1-a)$ is coefficientwise nonnegative, but
  $F_6/(1-a)$ is not: $A_6$ has negative coefficients with $i = 0$, which the prefix sum over
  $r \le i$ cannot touch. The stepping stone proposed in §11.4 of the handoff note is unavailable
  at $d=6$.
- An explicit, machine-found and **exactly verified denominator certificate for $A_4$**, of order
  exactly 4 — an independent computational proof of strong Laurent positivity at $d=4$.
- **Rigorous lower bounds on certificate order.** Nothing of order $\le 7$ certifies $A_5$, even
  for the prefix series.
- The **unpaired tail collapses onto one series** $J_d(1/2,\dots)$ in $d-2$ variables. Ours
  reproduces the published closed form at $d=5$ coefficient for coefficient, and the multiplicative
  certificate search reconstructs Proposition 3 mechanically, with leftover exactly $1$.

## Denominator certificates

If $N = \sum_{|S|\le k} P_S \prod_{r\in S}(1-f_r)$ with every $P_S$ coefficientwise nonnegative,
then $F = \sum_S P_S / \prod_{r \notin S}(1-f_r)$ is a sum of products of nonnegative series, so
$F \ge 0$. This is an **order-$k$ certificate**.

The search (`chernpp/certificates.py`) is a feasibility LP in which $P_\emptyset$ is *not* a free
unknown but the remainder $N - \sum_{S\ne\emptyset} P_S\prod_{r\in S}(1-f_r)$. Whatever the
floating-point LP proposes, that remainder is recomputed in exact rational arithmetic and the
result accepted only if it verifies — **a returned certificate is a proof**, independent of the
solver.

Failure is provable too. A constraint on a monomial of total degree $T$ involves only the $P_S$
coefficients of degree $\le T$, so truncating both at $T$ is an exact *projection* of the feasible
set: if the truncated LP is infeasible, no certificate of that order exists at **any** degree.

## Layout

```
src/multidegree/     SageMath only.  Computes Q_d, writes chernpp/data/a{d}_algebra.npz
  morin.py             the A_d model: ambient weights, reference point, chamber assembly
  backends/            selectable multidegree algorithms (see below)
  build.py             command-line entry point and artifact export
src/chernpp/         Pure Python/JAX.  Reads the artifacts; never re-derives them
  polynomial.py        exact sparse polynomial arithmetic on exponent dictionaries
  artifacts.py         the chamber algebra and its invariants
  chamber.py           chamber series, negatives, tails, tau-pairing, C(M)
  chern.py             XLA fixed-point expansion + Chern-coefficient extraction
  certificates.py      additive denominator certificates and order obstructions
  lemma1.py            multiplicative certificates: Lemma 1, matching, absorption
  crt.py               exact Chern coefficients past the int64 ceiling
  tables.py            text tables and statistics for the mined objects
  families.py          closed-form domination along infinite families
  lorentzian.py        log-concavity / M-convexity tests
  experiments.py       command-line runner
  data/                the mined algebras, tracked
src/examples.ipynb   annotated tour, from the published results to the new ones
tests/               seven tiers, in dependency order
papers/              project reports plus Bérczi–Szenes, Annals 175 (2012)
```

## Running it

Two environments, deliberately separate. The Sage stage is needed only to regenerate
`src/chernpp/data/*.npz`, which are tracked — collaborators without SageMath can skip it.

```bash
pip install -e .
```

(or `uv pip install -e .`). The artifacts ship inside the package, so after installing, everything
below works from any directory.

```bash
python -m unittest discover -s tests
```

```bash
python -m chernpp.experiments --dim 6
```

Formatting is enforced at commit time by black. Once per clone:

```bash
pre-commit install
```

To regenerate the algebra artifacts (SageMath ≥ 10, with Singular). `build.py` is plain Python
that imports `sage.all`, so run it with the interpreter of your Sage installation — note that
`sage -python` was removed in Sage 10.9:

```bash
cd src && "$(dirname "$(command -v sage)")/python" -m multidegree.build -d 6
```

### Multidegree backends

The algorithm that computes $\mathcal{Q}_d$ is selectable, so a different route
— restriction equations, a resolution and pushforward, a parametrisation and elimination — or a
singularity family beyond Morin $A_d$ can be dropped in without touching the chamber assembly, the
artifact schema, or anything in `chernpp`.

```bash
cd src && "$(dirname "$(command -v sage)")/python" -m multidegree.build --list-backends
```

```bash
cd src && "$(dirname "$(command -v sage)")/python" -m multidegree.build -d 6 --backend basic-equations
```

The separation is: `multidegree/morin.py` holds the *model* (which ambient space, what torus
weights, which reference point, how the residue formula is rewritten in chamber coordinates), and
`multidegree/backends/` holds the *algorithms*. A backend implements one method,
`compute(family, order, base_field)`, returning a `Multidegree` — which validates that the
polynomial's degree really is the codimension the backend claims. Register it in
`multidegree/backends/__init__.py`; `--backend` then selects it, and each artifact records the
family and backend that produced it. A backend whose dependencies are missing is reported rather
than taking the registry down, so `--list-backends` works even in a partial install.

The Sage stage refuses to emit an artifact it cannot justify. It checks that every orbit equation
is multihomogeneous for the torus weights, that the saturated ideal has exactly the codimension
$\dim\widehat{N}_d - \binom{d}{2}$ forced by homogeneity of the residue formula, that random
$B_d$-translates of $\epsilon_{\mathrm{ref}}$ lie on it, that $\deg\mathcal{Q}_d$ matches, that the
chamber correction divides exactly, and that the resulting numerator has constant term 1.

## Tests

`tests/` is tiered, most foundational first:

1. `test_1_polynomial.py` — ring laws, truncation, geometric series, exact rationals.
2. `test_2_artifacts.py` — schema invariants and internal consistency of the mined algebras.
3. `test_3_thom.py` — Thom polynomials against both the classical $\ell=0$ values and Rimányi's
   published $\ell=1$ tables (`tests/data/published_thom_polynomials.json`), Chern multiset
   structure, $\ell$-independence, the overflow guard, and CRT reconstruction.
4. `test_4_chamber.py` — both conjectures, ballot combinatorics, the reductions at $d=5$ vs $d=6$,
   and a cross-validation of $C(M)$ between the two independent implementations.
5. `test_5_certificates.py` — certificate verification, tamper rejection, order obstructions.
6. `test_6_backends.py` — the backend registry: registration, lookup, family filtering, and the
   contract a backend's result must satisfy. Runs without SageMath.
7. `test_7_tail.py` — the unpaired-tail series, multiplicative certificates, and absorption.

## Caveats

- Coefficients grow fast in $\ell$. The fast evaluator accumulates in `int64` and **raises** once
  the values would wrap, rather than returning negative-looking garbage. `chernpp/crt.py` goes
  past that by residue arithmetic; it is slower, so the `int64` path stays the default.
- Everything reported as "holds" over a truncation range is a finite exact computation, never a
  proof of the infinite statement. `chernpp/certificates.py` and `chernpp/families.py` are the
  only components that produce proofs.
- $d = 7$ is out of reach by this method: $\deg\mathcal{Q}_7 = 13$ in a 34-variable ambient space.

## References

In `papers/`:

- Bérczi & Szenes, *Thom polynomials of Morin singularities*, Annals of Math. **175** (2012).
  §7.3–7.4 give the orbit equations used here.
- `rimanyi_positivity.pdf` — the two conjectures; $d=4$ proved, $d=5$ strong form disproved.
- `report.pdf` — the $\ell$-free reduction, verification for $\ell \le 11$ at $d=5$, three dead ends.
- `a5_weak_positivity_handoff.pdf` — the paired reduction and the proved unpaired tail.
- `summary_draft.md` — a self-contained summary of this work for a non-specialist reader.
