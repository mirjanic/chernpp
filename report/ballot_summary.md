# Ballot words and the Morin Thom polynomials — summary

Companion to `report/ballot.pdf`.

**Setting.** Write `Tp^ℓ_{A_d} = Σ_M C(M) ∏_{a∈M} c_{ℓ+1+a}`, where `M` runs over zero-sum multisets of `d` integers. `C(M)` does not depend on `ℓ`, and Rimányi's conjecture says `C(M) ≥ 0`.

**Definition.** `b(M)` is the number of distinct orderings of `M` whose proper prefix sums are all nonnegative (ballot orderings).
- It is the packet sum of the pure chain series `∏_j (1 − x_j)^{−1}`, whose coefficients are all 1.
- `b(M) ≥ 1` always, because the decreasing ordering is ballot.

## Theorem (plane sector)

This is proved, given Theorem 5.4 of the unrefereed external findings summary.

If `max M ≤ 1` then

```
C(M) = b(M) = d! / ((d−b+1)! ∏ m_i!)
```

where `M = (1^{d−b}, 1−s_1, …, 1−s_b)` and `m_i` is the number of `s` equal to `i`. This is Kreweras' count of noncrossing partitions.

**Proof.**
- **External input:** the summary identifies `C(M)` with the coefficient of an `n`-cycle in `m_ν(J_1, …, J_n)`, where the `J_k` are the Jucys–Murphy elements.
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

**Refinement (zero insertion).** For zero-free `M₀`, the Newton coefficients in `z` of both `C(M₀ ⊔ 0^z)` and `C − b` are ≥ 0. This holds on all 4,925 bases with d ≤ 7.

For example, `C(2,−2,0^z) = 2 + 5z + 4·C(z,2) + C(z,3)`.

**Failed refinements:**
- factorial weights on ballot words (they fail at d = 3);
- "fusion" of a 2 into 1 + 1 (it leaves negative residuals);
- local step weights for the 2-steps (none is consistent across `d`);
- the formal Jucys–Murphy formula with `J_k^{−1}` (wrong values, some fractional).

**Where a proof must go.** A charge-`p` packet lives on the curvilinear locus in `C^{p+1}`. The conjecture says the extra dimensions only *add* to the planar ballot count. A charge-2 analogue of the Jucys–Murphy calculus, on the curvilinear component of `Hilb_0(C³)`, is the natural target.
