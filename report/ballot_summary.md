# Ballot words and the Morin Thom polynomials — summary

Companion to `report/ballot.pdf`.

**Setting.** Write `Tp^ℓ_{A_d} = Σ_M C(M) ∏_{a∈M} c_{ℓ+1+a}`, where `M` runs over zero-sum multisets of `d` integers. `C(M)` does not depend on `ℓ`, and Rimányi's conjecture says `C(M) ≥ 0`.

**Definition.** `b(M)` is the number of distinct orderings of `M` whose proper prefix sums are all nonnegative (ballot orderings).
- It is the packet sum of the pure chain series `∏_j (1 − x_j)^{−1}`, whose coefficients are all 1.
- `b(M) ≥ 1` always, because the decreasing ordering is ballot.

## Theorem (plane sector)

This is proved, given two geometric inputs, both stated in full in `ballot.pdf` §2.1: the curvilinear form of the Bérczi–Szenes incidence formula, and a multiplicity-one hyperplane lemma.

If `max M ≤ 1` then

```
C(M) = b(M) = d! / ((d−b+1)! ∏ m_i!)
```

where `M = (1^{d−b}, 1−s_1, …, 1−s_b)` and `m_i` is the number of `s` equal to `i`. This is Kreweras' count of noncrossing partitions.

**Proof.**
- **Geometric input** (`ballot.pdf` §2.1).
  - `C(M)` is an integral of a monomial symmetric function of the tautological bundle over the curvilinear component `C_{n,r}` of the punctual Hilbert scheme of `C^r`, with `n = d+1`.
  - The hyperplane lemma lowers `r` one step at a time; on the plane sector it reaches `r = 2`.
  - There, by Briançon and Lehn–Sorger, the integral is the coefficient of an `n`-cycle in `m_{1−M}(J_1, …, J_n)`, where the `J_k` are the Jucys–Murphy elements.
- **New lemma:** for the cycle `x ↦ x+1`, each ballot exponent vector has **exactly one** minimal monotone factorisation, and no other vector has any.
  - Peel off the transpositions through `n`.
  - Their other endpoints are forced to decrease.
  - That leaves increasing cycles on consecutive intervals.
  - The cut points are the last visits of the deficit walk `D(j) = j−1−Σ_{k≤j} e_k` to each level.
- **Last step:** the cycle lemma gives the product formula.
- **Checks:** exhaustive for `n ≤ 7` (the totals are Catalan numbers), and `C = b` on every plane packet with `d ≤ 7`.

## Conjecture (ballot)

**`C(M) ≥ b(M)` for every `M`, with equality exactly when `max M ≤ 1`.**

Equivalently, `F_d − ∏(1−x_j)^{−1}` has nonnegative packet sums, and they vanish exactly on the plane sector.

- It implies Rimányi's conjecture, with every occurring coefficient at least 1. This explains why the least Chern coefficient is 1, never 0.
- It holds with no violation and the exact equality case on every exact table (`results/ballot_conjecture.json`, `tools/ballot_check.py`):
  - d ≤ 6 on large level boxes;
  - `Tp_{A_7}` for ℓ ≤ 5;
  - every d = 7 packet of charge ≤ 6.
- Off the plane sector `C/b ≥ 2`, with minima 2, 2, 9/4, 12/5, 79/30, 58/21 for d = 2..7.

**Theorem (d ≤ 4), unconditional.** The ballot conjecture holds for d ≤ 4, with equality exactly on the plane sector and `C ≥ b + 1` off it.
- **Main input:** `F_3 = (1−x)(1−y)(1−xy)/((1−2x)(1−2xy)(1−y−xy))` factors into three ratios of the form `1 + v/(1−u−v)`.
- **Dominated series:** that product dominates `Z_3 = 1 + x/((1−x)(1−y)) + xy/(1−xy)`.
  - The packet sums of `Z_3` are exactly `b(M)`.
  - The zero face `(0,k)` of `F_3` is compensated by the diagonal `(k,k)` in the same packet.
- **Equality case:** at fixed d the plane sector is finite, so equality there is a finite check.

**The d = 4 case.**
- **Explicit formula.** `F_4 = (½/(1−2x₁) + ½/(1−2s) + x₁²s/((1−2x₁)(1−s−t)))` × four Lemma-1 ratios, with `s = x₂x₃` and `t = x₁x₂x₃`. This is a new one-line proof of strong positivity at d = 4.
- **Proof strategy.** `F_4` dominates the number of the 64 product supports containing each cell. That count dominates the weight of a comparison series `Z_4` whose packet sums are `b(M)`, plus one unit at every decreasing cell with `β₁ ≥ 2`.
- **Verification.** The last step is three inclusions of Presburger sets, decided exactly by isl (`chernpp.ballot.verify_d4`).

**Refinement (zero insertion).** For zero-free `M₀`, the Newton coefficients in `z` of both `C(M₀ ⊔ 0^z)` and `C − b` are ≥ 0. This holds on all 4,925 bases with d ≤ 7.

For example, `C(2,−2,0^z) = 2 + 5z + 4·C(z,2) + C(z,3)`.

**Failed refinements:**
- factorial weights on ballot words (they fail at d = 3);
- "fusion" of a 2 into 1 + 1 (it leaves negative residuals);
- local step weights for the 2-steps (none is consistent across `d`);
- the formal Jucys–Murphy formula with `J_k^{−1}` (wrong values, some fractional).

**Where a proof must go.** A charge-`p` packet lives on the curvilinear locus in `C^{p+1}`. The conjecture says the extra dimensions only *add* to the planar ballot count. A charge-2 analogue of the Jucys–Murphy calculus, on the curvilinear component of `Hilb_0(C³)`, is the natural target.
