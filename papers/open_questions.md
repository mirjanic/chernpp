# Open questions and sharp edges

Points where the computations in this repository run past what we can justify from
the literature we have. Each states what was checked, what it settles, and what it
does not. Intended for discussion rather than as claims.

---

## 1. What $\mathcal{Q}_d$ is an invariant of

**Checked.** Take the corank-two 2-jet space $\mathrm{Hom}(\mathbb{C}^2,\mathrm{Sym}^2\mathbb{C}^2)$,
of dimension 6, and the class $I_{2,2}$ with local algebra
$\mathbb{C}[[x,y]]/(xy,\,x^2+y^2) \cong \mathbb{C}[[x,y]]/(x^2,y^2)$. Four germs of this
one class give three different $B_2\times B_2$-orbit closures:

| germ | orbit closure | codim |
|---|---|---|
| $(xy,\;x^2+y^2)$ | $q^u_{02}$ | 1 |
| $(x^2+xy,\;y^2)$ | $q^u_{02}$ | 1 |
| $(x^2,\;y^2)$ | $q^u_{02},\,q^u_{11}$ | 2 |
| $(y^2,\;x^2)$ | three quadrics, determinantal | 2 |

The two standard presentations of the *same* algebra already disagree.

**The obvious repair fails.** One would demand the representative be generic against
the flags. But $B_2\times B_2$ has no dense orbit here: its generic orbit is
5-dimensional in a 6-dimensional space, so the action has modality $\ge 1$ and the
generic jet lies on a one-parameter family of orbits, whose closures are
hypersurfaces moving with the representative. By contrast $GL_2\times GL_2$ *is*
transitive on generic corank-two 2-jets — so $I_{2,2}$ is the generic corank-two
2-jet, of codimension 0, and its $GL\times GL$ class carries no information either.

**Reconciliation.** This is not a defect in Bérczi–Szenes. Their $\mathcal{Q}_d$ is
not "the class of the $A_d$ locus"; it is the equivariant class of *one specific*
Borel orbit closure, that of a canonically defined reference jet
$\epsilon_{\mathrm{ref}}$, and the residue formula is what converts it into a Thom
polynomial. The choice of $\epsilon_{\mathrm{ref}}$ is part of the theorem, not a
normalisation one is free to vary.

**Question.** Is there a corank-two analogue of $\epsilon_{\mathrm{ref}}$ — a
canonical reference jet — together with a residue formula proved against it? Does
the fibration of Bérczi–Szenes Lemma 7.1, under which $\mathrm{pr}_0$ restricted to
the orbit fibres over the toric orbit $T_d\cdot\epsilon_{\mathrm{ref}}$, have a
corank-two counterpart? Without both halves a corank-two multidegree is a number
without a theorem attached, which is why `multidegree/corank2.py` stops there and
emits no artifact.

---

## 2. The acting group beyond 2-jets

On 2-jets only the *linear* parts of source and target diffeomorphisms act: for $f$
with no constant or linear term, $f(\varphi_1+\varphi_2+\cdots) = f(\varphi_1)+O(3)$
and $\psi(f) = \psi_1(f) + O(4)$. From order 3 the higher parts contribute, so the
acting group is the jet group, not $GL_2\times GL_2$.

Every $I_{a,b}$ with $b \ge 3$ lives in the $b$-jet space — $I_{2,3}$ and $I_{3,3}$ in
dimension 14, $I_{2,4}$ in dimension 24 — so the whole family past $I_{2,2}$ needs
this. `orbit_closure` refuses order $> 2$ rather than answering for the wrong group.

**Question.** What is the right Borel of the jet group here, and does the
defect-zero saturation trick (Section 4 below) have an analogue that isolates the
orbit component?

---

## 3. Which positivity, in which basis

Rimányi's conjecture is that $\mathrm{Tp}_{A_d}$ has nonnegative coefficients in the
relative *Chern monomial* basis. That is a corank-one statement and does not survive
contact with corank two. From Rimányi's own registry, at relative dimension 0:

$$\mathrm{Tp}(I_{2,2}) = c_2^2 - c_1c_3 = s_{2,2},$$

the Giambelli–Thom–Porteous class of $\Sigma^2$. We expanded the published
$I_{2,2}, I_{2,3}, I_{2,4}, I_{3,3}$ in the Schur basis: every one is Schur-positive,
and every one has mixed signs in Chern monomials. So the $I$ family tests
Schur positivity — the Pragacz–Weber conjecture — and refutes any naive extension of
the Chern-monomial statement.

**Question.** Is Chern-monomial positivity genuinely special to Morin
singularities, or is it the shadow of a statement about a basis interpolating
between Chern monomials and Schur polynomials? The $A_d$ Thom polynomials are
positive in *both*; the $I_{a,b}$ only in one.

---

## 4. Where the $d=7$ evidence leaves the programme

Both reductions of the $A_5$ handoff note fail at $d=7$: the unpaired tail
($i>j \Rightarrow A_\beta \ge 0$) and the paired inequality
($A_\beta + A_{\tau\beta} \ge 0$). Both hold at $d=5$ and $d=6$, which is why they
looked structural. Rimányi's conjecture itself survives — the negative $A_\beta$
cancel inside the Chern coefficients, which match the published tables at every
order through $d=7$ and at two relative dimensions.

The $\tau$-fixed point $\beta = (1,2,1,2,2,1)$ is the sharpest case: $\tau$ fixes it,
so the paired inequality reads $2A_\beta \ge 0$ with no partner to appeal to, and
$A_\beta = -1$.

**Question.** Is there a replacement reduction? Concretely: can the negative part of
$C(M)$ be bounded by its largest term, uniformly in $d$? Every violation we see is
by exactly $-1$, against Chern coefficients in the tens.

---

## 5. The cost of $\mathcal{Q}_8$, and whether the equations are the right ones

$\mathcal{Q}_7$ takes about 85 s. Two configuration choices carry it: Singular's
`std` over `slimgb`, and ordering the variables so the defect-zero ones come last,
then saturating **back to front** so each step perturbs a cheap order slightly.

$d=8$ does not follow. The ambient dimension goes $34 \to 50$, $\deg\mathcal{Q}_d$
goes $13 \to 22$, the basic equations $21 \to 39$, and the defect-zero variables
$12 \to 16$. The first saturation step — 0.5 s at $d=7$ — does not finish in an hour.

**Question.** Is the obstruction the equations or the method? Bérczi–Szenes
Remark 7.2 note that standard algorithms exist for the equivariant dual of a
*toric* orbit but that none is known for Borel orbits. The toric route was tried
here and closed off: $\mathrm{eP}[T_d]$ computes in 0.13 s even at $d=7$ and matches
the published codimension table, but does not divide $\mathcal{Q}_6$, so it is not a
factor of the Borel class.
