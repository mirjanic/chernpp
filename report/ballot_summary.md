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
  - d ≤ 6 on large level boxes (A_6 to ℓ = 12);
  - `Tp_{A_7}` for ℓ ≤ 5;
  - every d = 7 packet of charge ≤ 7.
- Off the plane sector `C/b ≥ 2`, with minima 2, 2, 9/4, 12/5, 79/30, 58/21 for d = 2..7.

**Theorem (d ≤ 5), unconditional.** The ballot conjecture holds for d ≤ 5, with equality exactly on the plane sector and `C ≥ b + 1` off it.
- **Main input:** `F_3 = (1−x)(1−y)(1−xy)/((1−2x)(1−2xy)(1−y−xy))` factors into three ratios of the form `1 + v/(1−u−v)`.
- **Dominated series:** that product dominates `Z_3 = 1 + x/((1−x)(1−y)) + xy/(1−xy)`.
  - The packet sums of `Z_3` are exactly `b(M)`.
  - The zero face `(0,k)` of `F_3` is compensated by the diagonal `(k,k)` in the same packet.
- **Equality case:** at fixed d the plane sector is finite, so equality there is a finite check.

**The d = 4 case.**
- **Explicit formula.** `F_4 = (½/(1−2x₁) + ½/(1−2s) + x₁²s/((1−2x₁)(1−s−t)))` × four Lemma-1 ratios, with `s = x₂x₃` and `t = x₁x₂x₃`. This is a new one-line proof of strong positivity at d = 4.
- **Proof strategy.** `F_4` dominates the number of the 64 product supports containing each cell. That count dominates the weight of a comparison series `Z_4` whose packet sums are `b(M)`, plus one unit at every decreasing cell with `β₁ ≥ 2`.
- **The bijection.** The zero set of `F_4` is moved onto `T` by an explicit four-piece rearrangement of words (`chernpp.ballot.BIJECTION_D4`); isl checks that the pieces partition the zero set, are injective, and have images partitioning `T`.
- **Verification.** The last step is three inclusions of Presburger sets, decided exactly by isl (`chernpp.ballot.verify_d4`).

**The d = 5 case: the dominant cell.** The canonical `F_5` has negative coefficients, so the proof changes the numerator first.
1. **A residue-null kernel.** `Q₅ = GDJ + GBC`. `GBC` has zero packet sums, by a *contour-free swap lemma*. If `R = N/∏L` is invariant under `s = (i i+1)` and no denominator form with top variable `z_{i+1}` involves `z_i`, then the chamber expansion of `R` is `s`-invariant. So `V·R` is `s`-antisymmetric and its orbit sums vanish. Two swaps (`s₄`, then `s₁`) kill `GBC`. The hypotheses are finite exact checks.
2. **A nonnegative representative.** `F_5^{GDJ} = 𝒞 · ∏_{i≤6}(1 + V_i)`, with six Lemma-1 ratios and an eleven-term Stanley decomposition of the block `𝒞`. The decomposition is from the external findings summary, restated and checked exactly. The other pairing of the factors gives a second block with an exact 17-part order-5 certificate (`results/d5_block_certificate.json`); its weights are fractional, and no integer certificate of that shape exists up to degree 10, which is why the d = 4 support-counting argument does not transfer. The dominant-cell endgame below never counts supports, so fractional weights do not matter.
3. **One coefficient beats `b(M)`.** `C(M) ≥ F_5^{GDJ}(β_max(M))`. Every atomic factor has an explicit coefficient (`2^k`, or a binomial `≥ 2^{min}`), so "some product term is `≥ 2⁷`" is a Presburger set. isl shows it contains the whole decreasing cone with `β₁ ≥ 10`. Since `2⁷ = 128 > 5! ≥ b(M)`, every packet of charge ≥ 10 is done.
4. **Finite table.** The 2611 packets with `max M ≤ 9` are checked from the exact charge box.

The same argument re-proves d = 4 (threshold 9, bound `2⁵ > 4!`). Both run in `chernpp.ballot.verify_d5` / `verify_d4_dominant` (tier 15, about 20 s).

**What this says about d = 6, 7.** At fixed `d` the ballot conjecture follows from a residue-null kernel whose gauged series has a manifestly nonnegative product form. Growth along the decreasing cone then does the rest, and at d = 5 it comes almost entirely from the Lemma-1 ratios. So the only missing ingredient at d = 6 is a nonnegative gauge.

**Refinement (zero insertion).** For zero-free `M₀`, the Newton coefficients in `z` of both `C(M₀ ⊔ 0^z)` and `C − b` are ≥ 0. This holds on all 10,620 bases with d ≤ 7.

For example, `C(2,−2,0^z) = 2 + 5z + 4·C(z,2) + C(z,3)`.

**Failed refinements:**
- factorial weights on ballot words (they fail at d = 3);
- "fusion" of a 2 into 1 + 1 (it leaves negative residuals);
- local step weights for the 2-steps (none is consistent across `d`);
- the formal Jucys–Murphy formula with `J_k^{−1}` (wrong values, some fractional).

**Where a proof must go.** A charge-`p` packet lives on the curvilinear locus in `C^{p+1}`. The conjecture says the extra dimensions only *add* to the planar ballot count. A charge-2 analogue of the Jucys–Murphy calculus, on the curvilinear component of `Hilb_0(C³)`, is the natural target.
