import Mathlib

/-!
# Hedge regret on the expert simplex (the EG-arm guarantee)

The expert-simplex composition predicts with EG weights over K expert
paths. The weights have a CLOSED FORM (no recursion): after t steps,
w i t = exp(-eta * L i t) / Z t where L i t is expert i's cumulative loss
and Z t = sum_j exp(-eta * L j t).

## Table of contents

### Exponential Bound
* `exp_neg_le_one_sub_add_sq` : e^{-x} ≤ 1 - x + x² on [0,∞).

### Zfun Potential
* `Zfun_step` : the potential identity Z(t+1) = Z(t) · Σ_i w i t e^{-ηℓ i t}.

### Product Bound
* `Zfun_step_bound` : the per-step ratio bound via Step 1 + ℓ ≤ M.
* `Zfun_prod_bound` : the product bound by induction.

### Hedge Regret
* `hedge_regret` : Σ_t m t ≤ (L j T + ln K / η) / (1 - ηM) for any
  expert j, arbitrary adaptive losses in [0,M], no probability.
-/

/-! ## Exponential Bound
This section contains the elementary exponential inequality
`e^{-x} ≤ 1 - x + x²` on [0,∞), the analytic engine of the Hedge
potential argument.
-/

/-- e^{-x} ≤ 1 - x + x² for x ≥ 0: since exp x ≥ 1 + x, we have
e^{-x} ≤ 1/(1+x), and (1-x+x²)(1+x) = 1 + x³ ≥ 1 gives
1/(1+x) ≤ 1 - x + x². -/
theorem exp_neg_le_one_sub_add_sq (x : ℝ) (hx : 0 ≤ x) :
    Real.exp (-x) ≤ 1 - x + x ^ 2 := by
  have h1 : x + 1 ≤ Real.exp x := Real.add_one_le_exp x
  have hpos : 0 < 1 + x := by linarith
  have hexp0 : Real.exp (-x) * Real.exp x = 1 := by
    rw [← Real.exp_add]
    simp
  have hinv : Real.exp (-x) ≤ 1 / (1 + x) := by
    rw [le_div_iff₀ hpos]
    have hle : Real.exp (-x) * (1 + x) ≤ Real.exp (-x) * Real.exp x :=
      mul_le_mul_of_nonneg_left (by linarith [h1]) (Real.exp_nonneg (-x))
    linarith [hle, hexp0]
  have hdiv : 1 / (1 + x) ≤ 1 - x + x ^ 2 := by
    rw [div_le_iff₀ hpos]
    have hexpand : (1 - x + x ^ 2) * (1 + x) = 1 + x ^ 3 := by ring
    rw [hexpand]
    nlinarith [hx]
  linarith [hinv, hdiv]


noncomputable section

variable {K : ℕ} [NeZero K]

/-! ## Zfun Potential
This section contains the closed-form Hedge machinery: expert cumulative
losses, the normalizer potential `Z`, the closed-form EG weights, and the
potential step identity `Z(t+1) = Z(t) · Σ_i w i t · e^{-η ℓ i t}`.
-/

/-- Expert i's cumulative loss over the first t steps. -/
def Lcum (ell : Fin K → ℕ → ℝ) (i : Fin K) (t : ℕ) : ℝ :=
  ∑ s∈ Finset.range t, ell i s

/-- The normalizer Z t = Σ_j exp(-η L j t). -/
def Zfun (ell : Fin K → ℕ → ℝ) (η : ℝ) (t : ℕ) : ℝ :=
  ∑ j, Real.exp (-η * Lcum ell j t)

/-- The closed-form EG weight of expert i at time t. -/
def wgt (ell : Fin K → ℕ → ℝ) (η : ℝ) (i : Fin K) (t : ℕ) : ℝ :=
  Real.exp (-η * Lcum ell i t) / Zfun ell η t

/-- The weighted-mean loss at time t. -/
def mloss (ell : Fin K → ℕ → ℝ) (η : ℝ) (t : ℕ) : ℝ :=
  ∑ i, wgt ell η i t * ell i t

theorem Lcum_succ (ell : Fin K → ℕ → ℝ) (i : Fin K) (t : ℕ) :
    Lcum ell i (t + 1) = Lcum ell i t + ell i t := by
  simp [Lcum, Finset.sum_range_succ]

theorem Zfun_pos (ell : Fin K → ℕ → ℝ) (η : ℝ) (t : ℕ) :
    0 < Zfun ell η t := by
  obtain ⟨i₀⟩ : (Finset.univ : Finset (Fin K)).Nonempty :=
    ⟨⟨0, Nat.pos_of_ne_zero (NeZero.ne _)⟩, Finset.mem_univ 0⟩
  have hle : Real.exp (-η * Lcum ell i₀ t) ≤ Zfun ell η t :=
    Finset.single_le_sum
      (f := fun i => Real.exp (-η * Lcum ell i t))
      (fun i _ => le_of_lt (Real.exp_pos (-η * Lcum ell i t)))
      (Finset.mem_univ i₀)
  exact lt_of_lt_of_le (Real.exp_pos _) hle

theorem wgt_nonneg (ell : Fin K → ℕ → ℝ) (η : ℝ) (i : Fin K) (t : ℕ) :
    0 ≤ wgt ell η i t :=
  div_nonneg (Real.exp_nonneg _) (le_of_lt (Zfun_pos ell η t))

theorem weights_sum_one (ell : Fin K → ℕ → ℝ) (η : ℝ) (t : ℕ) :
    ∑ i, wgt ell η i t = 1 := by
  unfold wgt
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt (Zfun_pos ell η t))

/-- The potential identity: Z(t+1) = Z(t) · Σ_i w i t · e^{-η ℓ i t}. -/
theorem Zfun_step (ell : Fin K → ℕ → ℝ) (η : ℝ) (t : ℕ) :
    Zfun ell η (t + 1)
      = Zfun ell η t * ∑ i, wgt ell η i t * Real.exp (-η * ell i t) := by
  have hexp : ∀ i : Fin K, Real.exp (-η * Lcum ell i (t + 1))
      = Real.exp (-η * Lcum ell i t) * Real.exp (-η * ell i t) := by
    intro i
    rw [Lcum_succ, ← Real.exp_add]
    congr 1
    ring
  calc Zfun ell η (t + 1)
      = ∑ i, Real.exp (-η * Lcum ell i (t + 1)) := rfl
    _ = ∑ i, Real.exp (-η * Lcum ell i t) * Real.exp (-η * ell i t) :=
        Finset.sum_congr rfl (fun i _ => hexp i)
    _ = ∑ i, Zfun ell η t * (wgt ell η i t * Real.exp (-η * ell i t)) := by
        refine Finset.sum_congr rfl (fun i _ => ?_)
        have hz' : Zfun ell η t ≠ 0 := ne_of_gt (Zfun_pos ell η t)
        have hcancel : Zfun ell η t
            * (Real.exp (-η * Lcum ell i t) / Zfun ell η t)
            = Real.exp (-η * Lcum ell i t) := mul_div_cancel₀ _ hz'
        rw [wgt, ← mul_assoc, hcancel]
    _ = Zfun ell η t * ∑ i, wgt ell η i t * Real.exp (-η * ell i t) := by
          rw [Finset.mul_sum]


/-! ## Product Bound
This section contains the per-step ratio bound on the potential via the
exponential bound, and the product bound obtained by induction over the
horizon.
-/

/-- The per-step ratio bound: Z(t+1) ≤ Z(t) · e^{-η(1-ηM) · m t}. -/
theorem Zfun_step_bound (ell : Fin K → ℕ → ℝ) (η M : ℝ) (hη : 0 < η)
    (hloss : ∀ i t, 0 ≤ ell i t ∧ ell i t ≤ M) (t : ℕ) :
    Zfun ell η (t + 1)
      ≤ Zfun ell η t * Real.exp (-η * (1 - η * M) * mloss ell η t) := by
  rw [Zfun_step]
  have hquad : ∀ i : Fin K, Real.exp (-η * ell i t)
      ≤ 1 - η * ell i t + (η * ell i t) ^ 2 := by
    intro i
    have h := exp_neg_le_one_sub_add_sq (η * ell i t)
      (mul_nonneg (le_of_lt hη) (hloss i t).1)
    rw [← neg_mul] at h
    exact h
  have hterm : ∀ i : Fin K,
      wgt ell η i t * Real.exp (-η * ell i t)
        ≤ wgt ell η i t
            * (1 - η * ell i t + (η * ell i t) ^ 2) :=
    fun i => mul_le_mul_of_nonneg_left (hquad i) (wgt_nonneg ell η i t)
  have hsumle : ∑ i, wgt ell η i t * Real.exp (-η * ell i t)
      ≤ ∑ i, wgt ell η i t
            * (1 - η * ell i t + (η * ell i t) ^ 2) :=
    Finset.sum_le_sum (fun i _ => hterm i)
  -- split per term: w (1 - ηℓ + η²ℓ²) = w + (-η + η²ℓ)(w ℓ)
  have hsplit : ∑ i, wgt ell η i t
        * (1 - η * ell i t + (η * ell i t) ^ 2)
      = ∑ i, wgt ell η i t
          + ∑ i, (-η + η ^ 2 * ell i t) * (wgt ell η i t * ell i t) := by
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl (fun i _ => by ring)
  -- coefficient bound: -η + η²ℓ ≤ -η + η²M (from ℓ ≤ M, η² ≥ 0)
  have hcoef : ∀ i : Fin K,
      (-η + η ^ 2 * ell i t) ≤ (-η + η ^ 2 * M) := by
    intro i
    have hM := (hloss i t).2
    have hη2 : (0:ℝ) ≤ η ^ 2 := sq_nonneg η
    nlinarith [hM, hη2]
  have hprod : ∀ i : Fin K, 0 ≤ wgt ell η i t * ell i t :=
    fun i => mul_nonneg (wgt_nonneg ell η i t) (hloss i t).1
  have hsum2 : ∑ i, (-η + η ^ 2 * ell i t) * (wgt ell η i t * ell i t)
      ≤ ∑ i, (-η + η ^ 2 * M) * (wgt ell η i t * ell i t) :=
    Finset.sum_le_sum (fun i _ =>
      mul_le_mul_of_nonneg_right (hcoef i) (hprod i))
  have hsum3 : ∑ i, (-η + η ^ 2 * M) * (wgt ell η i t * ell i t)
      = (-η + η ^ 2 * M) * mloss ell η t := by
    rw [mloss, Finset.mul_sum]
  have hsum3' : (-η + η ^ 2 * M) * mloss ell η t
      = (-η * (1 - η * M)) * mloss ell η t := by ring
  have hw : ∑ i, wgt ell η i t = 1 := weights_sum_one ell η t
  -- assemble: the inner sum ≤ 1 + (-η(1-ηM)) m
  have hfinal : ∑ i, wgt ell η i t * Real.exp (-η * ell i t)
      ≤ 1 + (-η * (1 - η * M)) * mloss ell η t := by
    rw [hsplit] at hsumle
    have hsum2' := le_trans hsum2 (le_of_eq hsum3)
    rw [hsum3'] at hsum2'
    linarith [hsumle, hsum2', hw]
  -- x + 1 ≤ exp x, then multiply by Z ≥ 0
  have hexp1 : (1:ℝ) + (-η * (1 - η * M)) * mloss ell η t
      ≤ Real.exp ((-η * (1 - η * M)) * mloss ell η t) := by
    linarith [Real.add_one_le_exp ((-η * (1 - η * M)) * mloss ell η t)]
  have hmul : Zfun ell η t * ∑ i, wgt ell η i t * Real.exp (-η * ell i t)
      ≤ Zfun ell η t * Real.exp ((-η * (1 - η * M)) * mloss ell η t) :=
    mul_le_mul_of_nonneg_left (le_trans hfinal hexp1)
      (le_of_lt (Zfun_pos ell η t))
  exact hmul

/-- The product bound: Z T ≤ Z 0 · e^{-η(1-ηM) Σ_{t<T} m t}. -/
theorem Zfun_prod_bound (ell : Fin K → ℕ → ℝ) (η M : ℝ) (hη : 0 < η)
    (hloss : ∀ i t, 0 ≤ ell i t ∧ ell i t ≤ M) (T : ℕ) :
    Zfun ell η T ≤ Zfun ell η 0
      * Real.exp (-η * (1 - η * M)
          * ∑ t∈ Finset.range T, mloss ell η t) := by
  induction T with
  | zero =>
    simp only [Finset.sum_range_zero, mul_zero, Real.exp_zero, mul_one]
    exact le_refl (Zfun ell η 0)
  | succ T ih =>
    have hstep := Zfun_step_bound ell η M hη hloss T
    have hpos : (0:ℝ) ≤ Real.exp (-η * (1 - η * M) * mloss ell η T) :=
      le_of_lt (Real.exp_pos _)
    calc Zfun ell η (T + 1)
        ≤ Zfun ell η T * Real.exp (-η * (1 - η * M) * mloss ell η T) :=
          hstep
      _ ≤ (Zfun ell η 0
            * Real.exp (-η * (1 - η * M)
                * ∑ t ∈ Finset.range T, mloss ell η t))
          * Real.exp (-η * (1 - η * M) * mloss ell η T) :=
          mul_le_mul_of_nonneg_right ih hpos
      _ = Zfun ell η 0
          * Real.exp (-η * (1 - η * M)
              * ∑ t ∈ Finset.range (T + 1), mloss ell η t) := by
          have harg : (-η * (1 - η * M))
                * ∑ t ∈ Finset.range T, mloss ell η t
                + (-η * (1 - η * M)) * mloss ell η T
              = (-η * (1 - η * M))
                  * ∑ t ∈ Finset.range (T + 1), mloss ell η t := by
            rw [Finset.sum_range_succ]
            ring
          rw [mul_assoc, ← Real.exp_add, harg]


/-! ## Hedge Regret
This section contains the Hedge regret theorem: for any expert j, the
weighted-mean loss never exceeds `(L j T + ln K / η) / (1 - ηM)`.
-/

/-- **Hedge regret (closed-form weights, adaptive losses, no probability).**
For arbitrary expert losses in [0, M] (adaptivity allowed), any expert j,
any horizon T, and η M < 1:
`Σ_t m t ≤ (L j T + ln K / η) / (1 - η M)`. -/
theorem hedge_regret (ell : Fin K → ℕ → ℝ) (η M : ℝ) (hη : 0 < η)
    (hloss : ∀ i t, 0 ≤ ell i t ∧ ell i t ≤ M)
    (hηM : η * M < 1) (T : ℕ) (j : Fin K) :
    (∑ t∈ Finset.range T, mloss ell η t)
      ≤ (Lcum ell j T + Real.log (K : ℝ) / η) / (1 - η * M) := by
  have hM0 : (0:ℝ) ≤ M := le_trans (hloss j 0).1 (hloss j 0).2
  have hc : 0 < η * (1 - η * M) := by
    have h1 : 0 < 1 - η * M := by linarith
    exact mul_pos hη h1
  have hsingle : Real.exp (-η * Lcum ell j T) ≤ Zfun ell η T :=
    Finset.single_le_sum
      (f := fun i => Real.exp (-η * Lcum ell i T))
      (fun i _ => le_of_lt (Real.exp_pos (-η * Lcum ell i T)))
      (Finset.mem_univ j)
  have hprod := Zfun_prod_bound ell η M hη hloss T
  have hZ0 : Zfun ell η 0 = (K : ℝ) := by
    unfold Zfun Lcum
    simp
  rw [hZ0] at hprod
  have hchain : Real.exp (-η * Lcum ell j T)
      ≤ (K : ℝ) * Real.exp (-η * (1 - η * M)
          * ∑ t∈ Finset.range T, mloss ell η t) :=
    le_trans hsingle hprod
  have hKpos : (0 : ℝ) < (K : ℝ) :=
    Nat.cast_pos.mpr (Nat.pos_of_ne_zero (NeZero.ne _))
  have hlp : (-η * Lcum ell j T)
      ≤ Real.log (K : ℝ)
          + (-η * (1 - η * M)
              * ∑ t∈ Finset.range T, mloss ell η t) := by
    have h1 : (0:ℝ) < Real.exp (-η * Lcum ell j T) := Real.exp_pos _
    have h2 : (0:ℝ) < (K : ℝ) * Real.exp (-η * (1 - η * M)
        * ∑ t∈ Finset.range T, mloss ell η t) :=
      mul_pos hKpos (Real.exp_pos _)
    have h3 := Real.log_le_log h1 hchain
    rw [Real.log_exp, Real.log_mul (ne_of_gt hKpos)
      (Real.exp_ne_zero _), Real.log_exp] at h3
    exact h3
  -- final algebra: c S ≤ η L j + ln K, then divide through
  have hfinal : η * (1 - η * M)
      * (∑ t∈ Finset.range T, mloss ell η t)
      ≤ η * Lcum ell j T + Real.log (K : ℝ) := by
    nlinarith [hlp]
  field_simp
  nlinarith [hfinal, hc]

end
