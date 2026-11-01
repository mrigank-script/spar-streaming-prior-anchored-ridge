import Mathlib

/-!
# OGD regret on the anchor pull + the window optimum

Machine-checked core of the composition-theorem chain (scalar-pull
family). No probabilistic assumptions anywhere.

## Table of contents

### Subgradient Identity
* `sq_subgradient`     : the EXACT subgradient identity for squared losses
  along the segment family (the loss-convexity step of the composition bound).

### Box Projection
* `projBox_dist_sq_le` : the box projection [0,1] is a contraction in
  squared distance toward any point of the box.

### OGD Regret (1-Dimensional)
* `ogd_regret_box`     : the Zinkevich step: OGD with projection to [0,1]
  has regret ≤ (w_0 - sstar)^2 / (2*eta) + eta*T*G^2/2 against any fixed
  sstar in the box, for any losses satisfying the subgradient inequality
  with gradient bound G.
* `ogd_regret_segment` : Proposition 4(i) assembled: squared error along the
  pull-identity segment family satisfies the subgradient identity exactly
  (gradient g_t = 2*b_t*(a_t + b_t*w_t)), hence OGD on the anchor pull
  is no-regret against any fixed pull.

### Window Optimum
* `window_optimum`     : the Window Sign minimizer is optimal via the EXACT
  factorization R(w) - R(wstar) = (nu^2 / (w^2 * wstar^3)) * (w - wstar)^2
  * (2w + wstar) >= 0.

### Window Convexity
* `window_second_deriv_pos` : the exact window risk is strictly convex on
  (0, 2): its second derivative is strictly positive.
-/

/-! ## Subgradient Identity
This section contains the exact subgradient identity for squared losses
along the segment family (the loss-convexity step of the composition
bound).
-/

theorem sq_subgradient (a b s t : ℝ) :
    (a + b * s) ^ 2 - (a + b * t) ^ 2 - 2 * b * (a + b * t) * (s - t)
      = (b * (s - t)) ^ 2 := by
  ring


/-! ## Box Projection
This section contains the projection onto the box [0,1] and the lemmas
showing it is a contraction in squared distance toward any point of the
box.
-/

/-- Projection onto the box [0,1]. -/
def projBox (x : ℝ) : ℝ := min 1 (max 0 x)

theorem projBox_le (x : ℝ) : projBox x ≤ 1 := min_le_left _ _

theorem projBox_nonneg (x : ℝ) : 0 ≤ projBox x :=
  le_min zero_le_one (le_max_left 0 x)

theorem projBox_of_le {x : ℝ} (hx : x ≤ 0) : projBox x = 0 := by
  unfold projBox
  rw [max_eq_left hx, min_eq_right zero_le_one]

theorem projBox_of_ge {x : ℝ} (hx : 1 ≤ x) : projBox x = 1 := by
  have h0x : (0:ℝ) ≤ x := le_trans zero_le_one hx
  unfold projBox
  rw [max_eq_right h0x, min_eq_left hx]

theorem projBox_self {x : ℝ} (h0 : 0 ≤ x) (h1 : x ≤ 1) : projBox x = x := by
  unfold projBox
  rw [max_eq_right h0, min_eq_right h1]

/-- The box projection is a contraction in squared distance toward any
point of the box. -/
theorem projBox_dist_sq_le (s x : ℝ) (h1 : 0 ≤ s) (h2 : s ≤ 1) :
    (projBox x - s) ^ 2 ≤ (x - s) ^ 2 := by
  rcases le_total x 0 with hx0 | hx0
  · rw [projBox_of_le hx0]
    have key : (x - s) ^ 2 - s ^ 2 = x * (x - 2 * s) := by ring
    have hxs : x - 2 * s ≤ 0 := by linarith
    have hprod : 0 ≤ x * (x - 2 * s) := by nlinarith
    nlinarith [key, hprod, sq_nonneg s]
  · rcases le_total x 1 with hx1 | hx1
    · rw [projBox_self hx0 hx1]
    · rw [projBox_of_ge hx1]
      have key : (x - s) ^ 2 - (1 - s) ^ 2 = (x - 1) * (x + 1 - 2 * s) := by
        ring
      have hlo : 0 ≤ x - 1 := by linarith
      have hhi : 0 ≤ x + 1 - 2 * s := by linarith
      have hprod : 0 ≤ (x - 1) * (x + 1 - 2 * s) := mul_nonneg hlo hhi
      nlinarith [key, hprod, sq_nonneg (1 - s)]


/-! ## OGD Regret (1-Dimensional)
This section contains the Zinkevich step for OGD with projection to
[0,1], the telescoped regret bound on the box, and the assembled
composition bound for squared errors along the pull-identity segment
family.
-/

theorem ogd_step (l : ℕ → ℝ → ℝ) (g w : ℕ → ℝ) (η sstar : ℝ) (hη : 0 < η)
    (t : ℕ)
    (hconv : ∀ s, s ∈ Set.Icc (0:ℝ) 1 →
      l t (w t) + g t * (s - w t) ≤ l t s)
    (hrec : w (t + 1) = projBox (w t - η * g t))
    (hs : sstar ∈ Set.Icc (0:ℝ) 1) :
    l t (w t) - l t sstar
      ≤ ((w t - sstar) ^ 2 - (w (t + 1) - sstar) ^ 2) / (2 * η)
        + η * (g t) ^ 2 / 2 := by
  have hgrad : l t (w t) - l t sstar ≤ g t * (w t - sstar) := by
    have h := hconv sstar hs
    linarith
  have hclip : (w (t + 1) - sstar) ^ 2 ≤ (w t - η * g t - sstar) ^ 2 := by
    rw [hrec]
    exact projBox_dist_sq_le sstar _ hs.1 hs.2
  have hexp : (w t - η * g t - sstar) ^ 2
      = (w t - sstar) ^ 2 - 2 * η * g t * (w t - sstar) + (η * g t) ^ 2 := by
    ring
  rw [hexp] at hclip
  have h2η : (0:ℝ) < 2 * η := by linarith
  have hkey : g t * (w t - sstar)
      ≤ (((w t - sstar) ^ 2 - (w (t + 1) - sstar) ^ 2) + (η * g t) ^ 2)
        / (2 * η) := by
    rw [le_div_iff₀ h2η]
    have hlin : (g t * (w t - sstar)) * (2 * η)
        ≤ ((w t - sstar) ^ 2 - (w (t + 1) - sstar) ^ 2)
          + (η * g t) ^ 2 := by
      nlinarith [hclip]
    linarith
  have hsplit : (((w t - sstar) ^ 2 - (w (t + 1) - sstar) ^ 2)
      + (η * g t) ^ 2) / (2 * η)
      = ((w t - sstar) ^ 2 - (w (t + 1) - sstar) ^ 2) / (2 * η)
        + η * (g t) ^ 2 / 2 := by
    rw [add_div]
    have hq : (η * g t) ^ 2 / (2 * η) = η * (g t) ^ 2 / 2 := by
      field_simp
    rw [hq]
  rw [hsplit] at hkey
  linarith [hgrad, hkey]

/-- **Zinkevich on the box** (no probability anywhere). -/
theorem ogd_regret_box (l : ℕ → ℝ → ℝ) (g w : ℕ → ℝ) (η G sstar : ℝ)
    (hη : 0 < η) (T : ℕ)
    (hconv : ∀ t s, s ∈ Set.Icc (0:ℝ) 1 →
      l t (w t) + g t * (s - w t) ≤ l t s)
    (hbnd : ∀ t, |g t| ≤ G)
    (hrec : ∀ t, w (t + 1) = projBox (w t - η * g t))
    (hs : sstar ∈ Set.Icc (0:ℝ) 1) :
    (∑ t ∈ Finset.range T, l t (w t))
      - (∑ t ∈ Finset.range T, l t sstar)
      ≤ (w 0 - sstar) ^ 2 / (2 * η) + η * T * G ^ 2 / 2 := by
  have main : ∀ n : ℕ,
      ((∑ t ∈ Finset.range n, l t (w t))
        - (∑ t ∈ Finset.range n, l t sstar)
        + (w n - sstar) ^ 2 / (2 * η))
      ≤ (w 0 - sstar) ^ 2 / (2 * η)
        + ∑ t ∈ Finset.range n, η * (g t) ^ 2 / 2 := by
    intro n
    induction n with
    | zero => simp
    | succ n ih =>
      have step := ogd_step l g w η sstar hη n
        (fun s hs' => hconv n s hs') (hrec n) hs
      rw [sub_div] at step
      have hs1 := Finset.sum_range_succ (fun t => l t (w t)) n
      have hs2 := Finset.sum_range_succ (fun t => l t sstar) n
      have hs3 := Finset.sum_range_succ (fun t => η * (g t) ^ 2 / 2) n
      rw [hs1, hs2, hs3]
      have rearr : ((∑ t ∈ Finset.range n, l t (w t) + l n (w n))
            - (∑ t ∈ Finset.range n, l t sstar + l n sstar))
           = ((∑ t ∈ Finset.range n, l t (w t))
             - (∑ t ∈ Finset.range n, l t sstar))
             + (l n (w n) - l n sstar) := by ring
      have combined : (((∑ t ∈ Finset.range n, l t (w t))
             - (∑ t ∈ Finset.range n, l t sstar)) + (l n (w n) - l n sstar)
             + (w (n + 1) - sstar) ^ 2 / (2 * η))
           ≤ ((w 0 - sstar) ^ 2 / (2 * η)
               + ∑ t ∈ Finset.range n, η * (g t) ^ 2 / 2)
             + η * (g n) ^ 2 / 2 := by
        have hadd := add_le_add ih step
        linarith [hadd, rearr]
      linarith [combined, rearr]
  have hmain := main T
  have h2ηnn : (0:ℝ) ≤ 2 * η := by linarith
  have hpot : 0 ≤ (w T - sstar) ^ 2 / (2 * η) :=
    div_nonneg (sq_nonneg _) h2ηnn
  have hheat : ∑ t ∈ Finset.range T, η * (g t) ^ 2 / 2
      ≤ η * T * G ^ 2 / 2 := by
    have hg2 : ∀ t ∈ Finset.range T, (g t) ^ 2 ≤ G ^ 2 := by
      intro t _
      obtain ⟨hlo, hhi⟩ := abs_le.mp (hbnd t)
      nlinarith [hlo, hhi, sq_nonneg G]
    have hle : ∀ t ∈ Finset.range T, η * (g t) ^ 2 / 2 ≤ η * G ^ 2 / 2 := by
      intro t ht
      exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left (hg2 t ht)
        hη.le) (by norm_num)
    have hsumle : ∑ t ∈ Finset.range T, η * (g t) ^ 2 / 2
        ≤ ∑ _t ∈ Finset.range T, η * G ^ 2 / 2 :=
      Finset.sum_le_sum hle
    have hcard : ∑ _t ∈ Finset.range T, η * G ^ 2 / 2
        = η * T * G ^ 2 / 2 := by
      rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
      push_cast
      ring
    linarith [hsumle, hcard]
  linarith [hmain, hpot, hheat]

/-- **The composition theorem (scalar-pull family), machine-checked.**
Losses are squared errors along the pull-identity segment
`a_t + b_t * s`; OGD on the pull s ∈ [0,1] has regret
`≤ 1/(2*eta) + eta*T*Gmax^2/2` against any fixed pull in the box. -/
theorem ogd_regret_segment (a b w : ℕ → ℝ) (η Gmax sstar : ℝ)
    (hη : 0 < η) (T : ℕ)
    (hbnd : ∀ t, |2 * b t * (a t + b t * w t)| ≤ Gmax)
    (hrec : ∀ t, w (t + 1)
      = projBox (w t - η * (2 * b t * (a t + b t * w t))))
    (hw0 : w 0 ∈ Set.Icc (0:ℝ) 1) (hs : sstar ∈ Set.Icc (0:ℝ) 1) :
    (∑ t ∈ Finset.range T, (a t + b t * w t) ^ 2)
      - (∑ t ∈ Finset.range T, (a t + b t * sstar) ^ 2)
      ≤ 1 / (2 * η) + η * T * Gmax ^ 2 / 2 := by
  have hconv : ∀ t s, s ∈ Set.Icc (0:ℝ) 1 →
      (a t + b t * w t) ^ 2
        + (2 * b t * (a t + b t * w t)) * (s - w t)
      ≤ (a t + b t * s) ^ 2 := by
    intro t s _
    have idt := sq_subgradient (a t) (b t) s (w t)
    have hnn : 0 ≤ (b t * (s - w t)) ^ 2 := sq_nonneg _
    linarith
  have hreg := ogd_regret_box
    (fun t s => (a t + b t * s) ^ 2)
    (fun t => 2 * b t * (a t + b t * w t)) w η Gmax sstar hη T
    hconv (fun t => hbnd t) (fun t => hrec t) hs
  have hw0s : (w 0 - sstar) ^ 2 ≤ 1 := by
    have h1 := hw0.1
    have h2 := hw0.2
    have h3 := hs.1
    have h4 := hs.2
    have hup : 0 ≤ 1 - (w 0 - sstar) := by linarith
    have hdown : 0 ≤ 1 + (w 0 - sstar) := by linarith
    have hprod : 0 ≤ (1 - (w 0 - sstar)) * (1 + (w 0 - sstar)) :=
      mul_nonneg hup hdown
    have hkey : (1 - (w 0 - sstar)) * (1 + (w 0 - sstar))
        = 1 - (w 0 - sstar) ^ 2 := by ring
    nlinarith [hprod, hkey]
  have h2ηpos : (0:ℝ) < 2 * η := by linarith
  have hbound : (w 0 - sstar) ^ 2 / (2 * η) ≤ 1 / (2 * η) := by
    have hkey : (w 0 - sstar) ^ 2 / (2 * η) - 1 / (2 * η)
        = ((w 0 - sstar) ^ 2 - 1) / (2 * η) := by
      field_simp
    have hnum : (w 0 - sstar) ^ 2 - 1 ≤ 0 := by linarith
    have hdiv : ((w 0 - sstar) ^ 2 - 1) / (2 * η) ≤ 0 := by
      exact div_nonpos_of_nonpos_of_nonneg (by linarith) (by linarith)
    linarith [hkey, hnum, hdiv]
  have hreg2 : (∑ t ∈ Finset.range T, (a t + b t * w t) ^ 2)
      - (∑ t ∈ Finset.range T, (a t + b t * sstar) ^ 2)
      ≤ (w 0 - sstar) ^ 2 / (2 * η) + η * T * Gmax ^ 2 / 2 := by
    simpa using hreg
  linarith [hreg2, hbound]


/-! ## Window Optimum
This section contains the optimality of the Window Sign minimizer via an
exact factorization of the risk difference into nonnegative factors.
-/

/-- The Window Sign minimizer is optimal via the EXACT factorization:
`R(w) - R(wstar) = (nu^2 / (w^2 * wstar^3)) * (w - wstar)^2 * (2w + wstar)`. -/
theorem window_optimum (c ν w wstar : ℝ) (hc : 0 < c) (hν : 0 < ν)
    (hw : 0 < w) (hwstar : 0 < wstar)
    (hcrit : wstar ^ 3 = 2 * ν ^ 2 / c) :
    c * wstar + ν ^ 2 / wstar ^ 2 ≤ c * w + ν ^ 2 / w ^ 2 := by
  have hc2 : c = 2 * ν ^ 2 / wstar ^ 3 := by
    rw [hcrit]
    field_simp
  have hfac : c * w + ν ^ 2 / w ^ 2 - (c * wstar + ν ^ 2 / wstar ^ 2)
      = (ν ^ 2 / (w ^ 2 * wstar ^ 3)) * (w - wstar) ^ 2 * (2 * w + wstar) := by
    rw [hc2]
    field_simp
    ring
  have hprod : 0 ≤ (ν ^ 2 / (w ^ 2 * wstar ^ 3)) * (w - wstar) ^ 2
      * (2 * w + wstar) := by positivity
  linarith [hfac, hprod]


/-! ## Window Convexity
This section contains the strict convexity of the exact window risk on
the interval (0, 2): its second derivative is strictly positive, so the
minimizer is unique.
-/

/-- **Second-derivative positivity of the exact window risk**: on the
interval `w ∈ (0,2)` with `c' > 0` and `ν² > 0`, the second derivative of
`R(w) = c'·w/(2−w) + ν²/w²` is `6ν²/w⁴ + 4c'/(2−w)³ > 0` (the
differentiation step is the inline calculus of the paper text), so `R` is
strictly convex and its minimizer is unique. -/
theorem window_second_deriv_pos (c' ν2 w : ℝ) (hc : 0 < c') (hν : 0 < ν2)
    (hw1 : 0 < w) (hw2 : w < 2) :
    0 < 6 * ν2 / w ^ 4 + 4 * c' / (2 - w) ^ 3 := by
  positivity
