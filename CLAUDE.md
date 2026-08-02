# Chern++ — working notes for Claude

Thom polynomials of Morin singularities $A_d$ ($d \le 7$) via the Bérczi–Szenes residue
formula, and the positivity conjectures attached to them. `README.md` is the accurate,
maintained description of the mathematics and the results — **read it first**; this file
only covers how to work in the repo and where the live questions are.

## Environment

The project venv is **`~/.virtualenvs/chernplusplus`** — use it, not a local `.venv`.
A bare `python3` is the system 3.14 and cannot import `chernpp`.

Its editable install points at the **main checkout**, so when working in a git worktree
you must put the worktree's `src` ahead of it or you will silently test the wrong code:

```bash
PYTHONPATH=$PWD/src ~/.virtualenvs/chernplusplus/bin/python -m unittest discover -s tests
```

From the main checkout the `PYTHONPATH` prefix is unnecessary. Full suite is 196 tests,
~5m20s, and passes clean. It needs no SageMath.

- **git LFS is a hard prerequisite.** `src/chernpp/data/*.npz` and
  `tests/data/published_thom_polynomials.json` are LFS objects. Without them
  `load_algebra` fails on a confusing *parse* error, not a missing-file one. Verify with
  `file src/chernpp/data/a7_algebra.npz` → should say `Zip archive`, not `ASCII text`.
- **SageMath is not installed here** and is not needed. It regenerates the `.npz`
  artifacts, which are tracked. Don't try to run `multidegree/` without it.
- black runs at commit time via pre-commit; the hook is already installed in the shared
  `.git`. `src/examples.ipynb` is deliberately excluded and formatted by hand.

## The exploration loop

Fast enough to iterate on directly — $A_7$ to degree 12 is about 4 seconds.

```bash
~/.virtualenvs/chernplusplus/bin/python -c "from chernpp.chamber import chamber_series, chern_coefficient; F = chamber_series(7, max_deg=12); print(chern_coefficient(F, [1,1,1,0,-1,-1,-1], 12))"
```

That prints `35`, which is the worked $d=7$ cancellation example in the report
(§"Why Rimányi's conjecture survives it") and also the published coefficient of
$c_1c_2^3$ in $\mathrm{Tp}_{A_7}$ at $\ell = 0$. It is a good one-line smoke test.

`chamber_series(dim, max_deg, ...)` takes the **order first, not an algebra** — passing an
algebra positionally interpolates its whole repr into a file path and produces a 68 MB
error message. Pass a loaded algebra as the `algebra=` keyword.

Bulk runs go through the experiment runner, which has five independently selectable
sections:

```bash
.venv/bin/python -m chernpp.experiments --dim 6 --only laurent reductions
```

## What the code will and will not do for you

The invariant worth preserving: **this codebase refuses rather than guesses.** Several
components deliberately raise where a lesser one would return plausible garbage.

- `chern_coefficient` raises if any ballot ordering of the multiset exceeds the truncation
  degree, rather than silently summing over the orderings that happen to fit. If you see
  that error, recompute the series deeper — do not lower your expectations of the answer.
- The fast Chern evaluator accumulates in `int64` and raises on wrap. Past that ceiling use
  `chernpp/crt.py` (slower, exact by CRT, verified against a held-back prime).
- The Sage stage refuses to emit an artifact whose codimension, degree, or constant term
  fails its own checks.

Keep that posture in new code. A wrong number here is far more expensive than a crash,
because the whole point is evidence about open conjectures.

## Proof status — the distinction that matters most

Be precise in writing and in commit messages about which of these you have:

- **A finite exact verification** — "holds for every $\beta$ of degree $\le 10$". Almost
  everything in the repo is this. It is never a proof of the infinite statement.
- **A proof** — only two components emit these. Certificate *order obstructions* (an
  infeasible truncated LP rules out that order at *every* degree, because truncation is an
  exact projection of the feasible set), and the closed-form family arguments in
  `lemma1.py`. A *returned* certificate is also a proof: the remainder is recomputed in
  exact rationals and accepted only if it verifies, so it does not depend on the LP solver.

The README and report are careful about this; match them.

## Where the mathematics actually stands

Settled and not worth re-litigating:

- Bérczi–Szenes strong (all $A_\beta \ge 0$) is **false** for $d \ge 5$. Rimányi weak
  (Chern coefficients $\ge 0$) is open and is the target.
- The **$\tau$-pairing frame is dead as a general route.** Both the unpaired tail and the
  paired inequality hold at $d = 5, 6$ and fail at $d = 7$ — the paired one in the sharpest
  way possible, at a fixed point $\tau(\beta) = \beta$ where it reads $2A_\beta \ge 0$ with
  $A_\beta = -1$. Anything built on it is small-$d$. Prefix positivity fails a step earlier,
  at $d = 6$.
- **Additive certificates are obstructed on every target tried**, including provably true
  ones. The report explicitly recommends against more effort there.
- Multiplicative certificates mechanically reproduce both published proofs, but what they
  certify is the individual $A_\beta$ — exactly what stops being controllable at $d = 7$.

## Gauge freedom (`chernpp/gauge.py`)

The newest piece, and the one that reframes the negative results above. The idea, and the
$d=5$ kernel that first showed it works, are **due to the two external reports in
`papers/`**: `a6_status.pdf` (31 July 2026, §§4.2, 7, 13) and
`rimanyi_positivity_findings_summary.pdf` (29 July 2026, §§3.2, 4.4, 4.5), plus the KIAS
progress note. The second contains the load-bearing proofs — $GBC$ residue-nullity by two
chamber-safe contour swaps, and the $GDJ$ Stanley decomposition. Both are still
unrefereed; treat them as sources to check, and keep citing them for the gauge idea.

$\mathcal{Q}_d$ is
not the only numerator computing $\mathrm{Tp}_{A_d}$: numerators differing by a
**residue-null kernel** give the same Thom polynomial and a different chamber series. So
$A_\beta$ is numerator-dependent while $C(M)$ is not, and a negative $A_\beta$ may be an
artefact of the representative rather than a fact about the singularity.

This matters because every obstruction in `certificates.py` is scoped to the canonical
$\mathcal{Q}_d$. At $d=5$ the search finds a kernel making the series **coefficientwise
nonnegative** — no certificate needed at all, where the LP had proved none exists of order
$\le 7$.

Two things the module learned the hard way, both encoded in the code:

- **Search in monomial coordinates, not a nullspace basis.** A row-reduced basis of the
  null space has entries reaching $8\times10^9$ at $d=5$; the resulting LP is too badly
  scaled to recover exactly.
- **Use integer variables.** The continuous relaxation returns a fractional vertex (21
  nonzeros near $0.5$ at $d=5$) that no rounding makes feasible. MILP returns a kernel
  that verifies immediately.

**Always call `validate_gauge` at a higher truncation than the search used.** `null_kernel_basis`
imposes nullity only on packets that fit under the truncation, and that is badly
underdetermined. At $d=5$ (48 packets vs 34 monomials) the kernel holds out-of-sample to
degree 22. At $d=6$ (29 packets vs **766** monomials) the search returns a kernel that
looks perfect in-sample and breaks 5 of 21 packets one truncation up — a false positive.
Packet supply grows ~4 per degree, so no reachable truncation fixes it.

### The symmetry construction — the route that scales

`symmetry_kernels` *constructs* null kernels instead of fitting them. The Chern insertion
is symmetric and the Vandermonde antisymmetric, so if $P/D_d$ is invariant under a
transposition $s$ the residue is its own negative. Writing $A$ for the product of the
$D_d$ factors that $s$ moves off the factor set, this is exactly

$$P = A\cdot R,\qquad R \text{ any } s\text{-invariant polynomial,}$$

null in all degrees at once. `analyse_swap` also applies the contour test — after
absorption every remaining factor pinning $z_i$ or $z_j$ must locate it on the *inner*
scale, or the two contours cannot be exchanged.

Validated: at $d=5$, $s_{45}$ absorbs exactly $\{z_2+z_3-z_5,\; z_1+z_4-z_5\}$ and
reproduces the published $MBC$ summand. **Zero of the constructed kernels fail the packet
falsifier at any order** — 3 at $d=5$, 65 at $d=6$, 4167 at $d=7$. Contour-safe swaps
exist at every order ($s_{47}$, $s_{57}$, $s_{67}$ at $d=7$).

The first construction is tight: $A_sR$ is null **iff** $R$ is $s$-invariant. That is
sufficient but incomplete, so there is a second family.

### Partial absorption — the second swap

`partial_absorption_kernels` absorbs all the moved factors **but one**: $P = (A_s/u)\cdot R$
with $R$ still $s$-invariant. Then $P/D_d = R/(Cu)$, whose antisymmetric part pairs $u$ with
its image $s(u)$ — the source-multiset completion — so a *second* transposition can kill it.
This is the mechanism that reaches $FBC$, which full invariance provably cannot.

These are a **shape, not a theorem**: most of what the family proposes is not null, so
everything goes through the packet falsifier (`null_candidates`).

**`filter_at` must exceed the fit truncation.** The falsifier is only as strong as the packets
in range. At $d=6$, degree 12 offers 13 packets against thousands of candidates — far too weak,
and non-null kernels get through and corrupt the search (observed: 3, then 8, then 22 packet
sums moving). Filtering at degree 20 (62 packets) cuts 2297 candidates to 410 genuinely null
ones.

### Status

**$d=5$ is solved with no fitted kernels at all** — the two structural families suffice,
validated at 16, 18, 20. $GBC$ is among the valid gauges (the solver may return a different
equally valid one).

**$d=6$ is not solved.** With the depth-20 filter the 410 candidates are genuinely null (0
packets move at any depth to 22), but no combination found so far is positive out of sample —
and it degrades with depth: fitting at 14 leaves 13/42, 36/66, 83/99, then 165/147 negatives,
i.e. worse than canonical by degree 22. Either the structural space lacks a positive element or
the fit must be run at a depth where the MILP is currently too slow. That is the open question.

The live lead, from the report's Assessment §4: since the $\tau$-orbit is not the right
unit of cancellation, **what is?** The negative mass is strikingly small and bounded —
at $d = 7$ it is 0.19% of the positive mass, least value $-2$, spread over few ballot
orderings while the positive mass concentrates in one. *A bound on the negative part of
$C(M)$ in terms of its largest term would settle the conjecture with no pairing at all*,
and nothing computed so far argues against one existing. Note also that the empirical
$\min C(M)$ is $1$ rather than $0$ across $d = 4..7$ in `papers/tables/cancellation.tex`.

Other open questions, with more setup cost, are in the report's final section: what
$\mathcal{Q}_d$ is an invariant of (corank two has no canonical reference jet — four germs
of $I_{2,2}$ give three orbit closures), the acting group beyond 2-jets, whether
Chern-monomial positivity is the shadow of a basis interpolating to Schur (the $I_{a,b}$
are Schur-positive but Chern-mixed), and why $\mathcal{Q}_8$ needs different mathematics
(the estimate is $\sim 10^{11}\times$ the cost of the degree-9 Gröbner step; it wants
localisation on a resolution, not a faster basis).

## Writing

`papers/chernpp_report.tex` is the live report; `papers/tables/*.tex` are **generated** by
`tools/render_tables.py` and carry a do-not-edit header. Regenerate rather than hand-edit.
Prior commit messages are full sentences describing the mathematical content of the change
("Compute Q_7, and show the A_5 reduction machinery fails at d = 7"); match that register.
