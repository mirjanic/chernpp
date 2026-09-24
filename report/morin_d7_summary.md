# Beyond Morin, and deeper into A₇: summary

A companion to `report/morin_d7.pdf`. Each item is labelled **proved**, **verified** (a finite exact computation) or **conjectured**.

## 1. The corank filtration: Rimányi's conjecture beyond Morin

Read the Chern class `c_k` as `h_k`. With that convention Pragacz–Weber says every Thom polynomial is Schur-positive. Define:

- `C_r` is the cone spanned by products of Schur functions with at most `r` rows;
- `ρ(η) = min{ r : Tp_η ∈ C_r }`.

`C_1` is the Chern-monomial cone. Rimányi's conjecture is therefore exactly **ρ(A_d) = 1**.

- **Proved.** `ρ(η) ≥ corank η`. The row-sum partition of each generator appears with coefficient 1 and cannot cancel. Every component of a corank-`r` Thom polynomial contains `(ℓ+r)^r`, and this was checked on every registry table.
- **Proved.** The sum of the Chern coefficients equals `[s_(n)] Tp`, because every Kostka number `K_{(n),μ}` is 1.
  - It is `(d!)^{ℓ+1}` for `A_d`, verified on all published and new tables.
  - It is `0` at corank ≥ 2. So **every non-Morin Thom polynomial has negative Chern coefficients, necessarily**.
- **Proved by exact certificate.** The natural generalisation `ρ = corank` is **false**.
  - `Tp_{I_{2,4}} = 16s₄₂ + 4s₃₃ + 12s₃₂₁ + 5s₂₂₂ + 2s₂₂₁₁` is not in `C_2`. The functional `[s₃₃] + [s₂₂₂] − [s₃₂₁]` is ≥ 0 on every usable generator and equals −3 on Tp.
  - Likewise `Tp_{III_{2,3}} = 8s₅₃ + 4s₄₃₁ + 2s₃₃₂` is not in `C_2`.
- **Verified** (the ρ table on the registry is `results/corank_survey.json`):
  - `ρ(I_{a,b}) = 2` iff `b ≤ a+1`, otherwise 3.
  - `ρ(III_{a,b}) = 2` iff `a = b`, otherwise 3.
  - `ρ(B_4)` rises from 2 to 3 between ℓ = 1 and ℓ = 2.
  - `ρ − corank` reaches 2 (for example `B_{5,3}`).

**Why this matters.** A Bérczi–Szenes-type residue formula on `Gr(r, TX)` with nonnegative packet sums would put `Tp_η` in `C_r`. Each Farkas certificate therefore rules out such a formula for that singularity. The open question is **what invariant of the local algebra predicts ρ**.

- *Negative result.* The basis variant (products of `r`-row slices, which has unique coefficients) is not positive in general.

## 2. New exact data (verified)

- **The sweep.** `chernpp/boxes.py` computes one pass per denominator factor, which is 34 passes at d = 7 instead of a fixed point of ~4500. It is 50–100× faster.
- **Exactness.** Values are reconstructed by CRT over word-sized primes and checked against a held-back prime.
- **New tables.**
  - **Tp_{A₇} for ℓ ≤ 5**: 8033 Chern coefficients.
  - **Tp_{A₆} for ℓ ≤ 8**: 12692 Chern coefficients.

  All are positive with minimum 1, and they agree with every published table. Before this work the published A₇ tables stopped at ℓ = 1.

## 3. Where positivity is tight

- **Verified, d ≤ 7.** On the plane sector `max M ≤ 1`, `C(M)` is a Kreweras number:

  ```
  C(M) = d! / ((d−b+1)! ∏ m_i!)
  ```

  This counts noncrossing partitions with the block sizes read from `M`. The only values equal to 1 are the global minima.
- **Margin outside the plane sector.** For `max M = 2` the minimum is 12, 17 and 27 at d = 5, 6, 7.
- **Conjectured (dominance).** Write:
  - `N(M)` for the negative mass of a packet;
  - `A_dom(M)` for the coefficient of its decreasing ordering.

  Then `N(M) ≤ κ_d · A_dom(M)` with `κ_d < 1`.

  **Evidence.** `κ₅ = κ₆ = 1/6` and `κ₇ = 8/21` hold exactly on every packet with `max M ≤ 8, 6, 5` respectively. The ratio falls 3–4× per unit of charge, so the conjecture is a small-charge statement. It implies Rimányi's conjecture and needs no pairing.
- **Correction.** Individual `A_β` are unbounded below: −3622 at d = 7. The earlier "least value −2" is wrong.
- **0-Hecke landscape.**
  - `P₅ = S₅ ∖ ⟨s₃, s₄⟩`.
  - `|P₆| = 643`, with 11 minimal elements; this is stable in the box.
  - At d = 7, `s₁ ∉ P₇`.
  - **Minimal parabolic symmetrisations that clear every negative:**
    - d = 5: {1} and {2};
    - d = 6: {1} and {2,5};
    - d = 7: {1,2,k} and {1,3,5} (stable on L = 4, 5).

    The cancellation unit grows from one transposition to three, which is why fixed-size pairings die.

## 4. What this means for Q₈

The corank filtration gets its evidence from breadth across coranks, and the registry supplies that. Q₈ is now needed for one sharp question: **is κ₈ < 1?**

Stabilisation `F_d|_{x_{d−1}=0} = F_{d−1}` is verified at d = 5, 6, 7. It gives Q₈ a hard constraint: `[z₈⁹] Q₈ = −Q₇`.

## Reproduce

- `tools/morin_tables.py`
- `tools/packet_anatomy.py`
- `tools/hecke_landscape.py`
- `tools/corank_survey.py`
- `tools/render_d7_tables.py`

Test tiers 10–14 cover the new modules.
