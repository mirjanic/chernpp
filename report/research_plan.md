# Research plan: phase 3

_Living document, updated as work lands. Status as of this commit._

## Context

The target is Rimányi's conjecture: every Chern coefficient `C(M)` of the Morin Thom polynomials `Tp_{A_d}` is nonnegative. Here `M` runs over zero-sum multisets of `d` integers. For the notation and all proofs see `report/ballot.pdf`, which is self-contained.

**What is established:**

| Statement | Status | Where |
|---|---|---|
| **The ballot conjecture.** `C(M) ≥ b(M)` (the number of ballot orderings), with equality iff `max M ≤ 1`. It implies Rimányi. | **Proved for d ≤ 4.** Verified on every exact table with d ≤ 7 (A_7 to ℓ = 5, every d = 7 packet with max M ≤ 7, A_6 to ℓ = 12). | `ballot.pdf` §§3–4; `chernpp/ballot.py`; tier 15 |
| **Plane sector.** `C(M) = b(M)` (a Kreweras number) when `max M ≤ 1`. | Proved for all d, given two geometric inputs that are stated in full | `ballot.pdf` §2 |
| **Explicit positive product formula for `F_4`.** A one-line proof of strong positivity at d = 4. | Proved | `ballot.pdf`, Lemma 4 |
| **Zero insertion.** `C − b` is a nonnegative binomial combination in the number of zeros. | Verified on 4,925 bases | `ballot.pdf` §5 |
| **Dominance.** `N(M) ≤ κ_d A_dom`, with κ = 1/6, 1/6, 8/21. | Verified | `morin_d7.pdf` |
| **The corank filtration.** `ρ ≥ corank`, and `ρ = corank` is refuted. `C_2` is simplicial only in degree n ≤ 5. | Proved (refutation by exact certificates) | `morin_d7.pdf` §1 |

**Dead routes**, each recorded with its obstruction in the reports:
- τ-pairing;
- additive certificates;
- local step weights;
- formal Jucys–Murphy inverses;
- fusion;
- factorial weights;
- the closed-form tower class (it differs from 𝒬₅ outside the residue-null kernel).

**Constraints set by the user:**
- Q₈ gets a scoped feasibility study only.
- Overnight runs are fine.
- Every write-up must be self-contained.
- Commit and push after every piece of work.

## Workstream A: prove the ballot conjecture for small d (headline)

- **d = 2, 3: done.** Hand proof. `F_3` factors into three Lemma-1 ratios and dominates a comparison series `Z_3` whose packet sums are `b(M)`.
- **d = 4: done.** The proof has three steps:
  1. The explicit positive product formula gives `F_4 ≥ N` = the number of the 64 product supports that contain a cell.
  2. A packet-preserving bijection moves the zero set of `F_4` onto a target set T. This gives a comparison series with packet sums `b(M)`.
  3. `N ≥ w + 1_D` reduces to three Presburger-set inclusions, decided exactly by isl.
- **d = 5: in progress.** The chamber numerator factors into ten monomial-type factors, one d = 4-type factor, and one factor that carries 𝒬₅. The canonical `F_5` is not strongly positive, so step 1 needs a residue-null gauge `N_5 + K`, where K comes from the structural kernel families in `optimisation/gauge.py`.
  - First check whether the gauge preserves the simple factors, leaving a small block that a small exact certificate can handle.
  - Then redo steps 2–3 with the gauged zero set.
  - Gate: if there is no positive factorised gauge within ~2 days, record the obstruction and stop at d = 4.

## Workstream B: the structure of charge 2 (exploratory, time-boxed)

Charge 2 (`max M = 2`) is where the conjecture now lives. Geometrically it is the curvilinear locus in `C^3`.

1. **Combinatorial models, tested exhaustively** on d ≤ 5, each required to predict d = 6, 7 exactly:
   - degree-2 Jucys–Murphy elements (3-cycle class sums ending at k);
   - monotone factorisations into transpositions and 3-cycles with a ballot condition;
   - decorated ballot words guided by the zero-insertion Newton coefficients.
2. **Geometric side.** Treat `C_{n,2} ⊂ C_{n,3}` as the zero scheme of a section of the tautological bundle E. Aim for "(planar count) + (nonnegative correction)".
3. **Zero-insertion polynomiality.** Try to prove it, and test the degree guess `Σ_{a>0} a + #{a<0}` on the deeper A_6 data.

## Workstream C: 𝒬₈ feasibility (scoped, gated)

1. Choose a smooth tower mapping birationally onto the orbit closure `O_d`, and restate it self-contained. As a negative control, understand why the rank-one nonassociative tower fails.
2. Compute `𝒬_d` by equivariant localisation (`src/multidegree/localisation.py`, no Sage).
3. **Gates:**
   - reproduce 𝒬₄ and 𝒬₅ exactly;
   - then 𝒬₆ and 𝒬₇;
   - extrapolate the cost to d = 8.
4. Only if every gate passes: run 𝒬₈ overnight and validate it against:
   - degree 22;
   - `[z_8^9]𝒬_8 = −𝒬_7`;
   - Rimányi's `Tp_{A_8}` at ℓ = 0.

   Then test the ballot and dominance conjectures at d = 8.
5. Otherwise write a self-contained obstruction note.

## Workstream D: long runs (overnight)

| Run | Status |
|---|---|
| D1. A_7 charge box p = 7 (118.5M cells) | **Done.** 16,475 coefficients, minimum 1 (`results/deep_a7_charge7.json`). |
| D2. A_6 level box L = 13 | **Done.** `Tp_{A_6}` to ℓ = 12: 62,239 coefficients, minimum 1. |
| D3. Finish the corank survey (41 tables, codimension 19–26) | **Running.** Two shards with no cap. The exact vertex-Farkas fallback settled the three previously undecided tables. |
| Ballot and zero-insertion re-checks on the deep tables | **Running.** Results go to `results/ballot_conjecture.json` and `results/zero_insertion.json`. |
| D4. d = 6 gauge search with the ballot target | Queued. Waits on the Workstream A tooling. |

## Writing

- `report/ballot.tex` (+ `ballot_summary.md`): the theorems and conjectures around `b(M)`.
- `report/morin_d7.tex` (+ summary): the data, anatomy, landscape and corank.
- `report/q8_feasibility.tex`, if Workstream C runs.
- This file, updated whenever a workstream moves.

Rules: every document self-contained; statements labelled **proved**, **verified** (finite exact computation) or **conjectured**; commit and push after each deliverable.

## Order and cut line

1. Finish A at d = 5, or hit its gate.
2. Then B.1.
3. Then C.1–C.3.
4. Then either C.4 or the obstruction note.
5. D4 runs overnight once A's tooling exists.

Cut, in this order: B.3, then D4, then C beyond gate 3. A (d ≤ 4, done) and D1–D3 are protected.

## Verification

- **Proofs.** Rest on exact identities and certificates re-verified in rationals, and on exact isl inclusions (tier 15).
- **Models.** Must predict held-out d exactly.
- **Localisation.** Must reproduce the tracked 𝒬₄..𝒬₇ exactly.
- **Long runs.** CRT with a held-back prime.
- **Before every push.** The full suite passes and the PDFs build.
