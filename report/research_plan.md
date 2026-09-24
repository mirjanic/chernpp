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
| **Zero insertion.** `C − b` is a nonnegative binomial combination in the number of zeros. | Verified on 10,620 bases | `ballot.pdf` §5 |
| **Dominance.** `N(M) ≤ κ_d A_dom`, with κ = 1/6, 1/6, 8/21. | Verified | `morin_d7.pdf` |
| **The corank filtration.** `ρ ≥ corank`, and `ρ = corank` is refuted; `ρ − corank` reaches 3 (`B_{5,4}`). `C_2` is simplicial only in degree n ≤ 5. | Proved (refutation by exact certificates) | `morin_d7.pdf` §1 |

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
- **d = 5: partial result, then gate.**
  - **Achieved.**
    - The GBC-gauged numerator factors completely.
    - The one remaining six-factor block has an exact certificate with 17 parts.
    - So the gauged `F_5` has an explicit nonnegative expression (`results/d5_block_certificate.json`, `ballot.verify_d5_block`, tier 15).
  - **Blocking the d = 5 ballot theorem.**
    - The certificate's weights are fractional. No integer certificate of that shape exists with parts of degree ≤ 10: the integer program is infeasible while its LP relaxation is feasible. So the support-counting step does not transfer.
    - GBC's nullity is proved only in the external report.
  - **Possible continuations.** A weighted Step 3 (Presburger counting with weights), an integer certificate of higher degree, or a self-contained proof of GBC's nullity.

## Workstream B: the structure of charge 2 (exploratory, time-boxed)

**Result so far (negative).** At r = 2 the curvilinear integral functional is a content evaluation with Frobenius weights over hooks. That is the plane theorem restated.

At r = 3 the exact tables determine the whole functional, and **no content-evaluation weights over partitions of n fit it** at d = 3, 4, 5 (`tools/content_functional.py`). So the naive charge-2 Jucys–Murphy calculus does not exist. Local step weights (exact fits) also fail across d.

Remaining ideas: a larger algebra (e.g. evaluation at 3-dimensional/plane-partition contents), or the degeneration route.


Charge 2 (`max M = 2`) is where the conjecture now lives. Geometrically it is the curvilinear locus in `C^3`.

1. **Combinatorial models, tested exhaustively** on d ≤ 5, each required to predict d = 6, 7 exactly:
   - degree-2 Jucys–Murphy elements (3-cycle class sums ending at k);
   - monotone factorisations into transpositions and 3-cycles with a ballot condition;
   - decorated ballot words guided by the zero-insertion Newton coefficients.
2. **Geometric side.** Treat `C_{n,2} ⊂ C_{n,3}` as the zero scheme of a section of the tautological bundle E. Aim for "(planar count) + (nonnegative correction)".
3. **Zero-insertion polynomiality.** Try to prove it, and test the degree guess `Σ_{a>0} a + #{a<0}` on the deeper A_6 data.

## Workstream C: 𝒬₈ feasibility — gate failed, recorded

`report/q8_feasibility.pdf` is self-contained. The natural resolution candidate is the rank-one nonassociative tower, with explicit point class `P*_d`. Findings:
- It equals 𝒬₄ exactly.
- For d = 5, 6, 7 it is **not** residue-null equivalent to 𝒬_d: it changes 21 of 32, 11 of 13, and 1 of 3 packet sums.
- The defect is a product of three linear forms at d = 5, a linear form times an irreducible sextic at d = 6, and an irreducible polynomial of degree 13 at d = 7. So it cannot be fixed by subtracting a few linear components.
- Reproduce with `tools/tower_class.py`.

Decision: no 𝒬₈ run.

Open follow-up: do the known constraints determine the *packet class* of 𝒬₈? These are the degree, the normalisation, stabilisation, the plane-sector theorem, and Rimányi's `Tp_{A_8}` at ℓ = 0. Answering this is a constraint count against the admissible numerator space.

## Workstream D: long runs (overnight)

| Run | Status |
|---|---|
| D1. A_7 charge box p = 7 (118.5M cells) | **Done.** 16,475 coefficients, minimum 1 (`results/deep_a7_charge7.json`). |
| D2. A_6 level box L = 13 | **Done.** `Tp_{A_6}` to ℓ = 12: 62,239 coefficients, minimum 1. |
| D3. Finish the corank survey | **Stopped at 94 of 128 tables**, merged into `results/corank_survey.json`. The exact vertex-Farkas fallback settled the three previously undecided tables. New: `ρ(B_{5,4}) = 5` at ℓ = 2, so `ρ − corank` reaches 3. The remaining 34 tables (codimension 20–26) need column generation. |
| Ballot and zero-insertion re-checks on the deep tables | **Done.** 133,104 packet checks and 10,620 bases, no violation (`results/ballot_conjecture.json`, `results/zero_insertion.json`). |
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
