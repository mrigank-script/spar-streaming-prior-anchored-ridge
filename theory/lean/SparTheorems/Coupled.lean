import SparTheorems.Ogd
import SparTheorems.Hedge

/-!
# Coupled two-dial contraction

The closed-loop stability core: two dials, each a contraction toward its
own target given the other dial frozen, couple through cross-Lipschitz
terms `a` and `b`. The joint iteration on the error pair contracts in a
weighted max-norm whenever the coupling product `a * b < beta * eta`,
with the explicit weight `r = (a / beta + eta / b) / 2` (the midpoint of
the compatible interval `a / beta < r < eta / b`, which exists exactly
when `a * b < beta * eta`).

## Table of contents

### Weighted Norm
* `joint_r`               : the midpoint weight.
* `joint_r_pos`           : positivity of the weight.
* `joint_bounds`          : the compatible interval is nonempty.
* `joint_weight_lo`       : the window-side compatibility `a / r < beta`.
* `joint_weight_hi`       : the strength-side compatibility `r * b < eta`.

### Rate
* `joint_rate`            : the joint contraction rate.
* `joint_rate_pos`        : positivity of the rate.
* `joint_rate_lt_one`     : the rate is contractive.

### Joint Step
* `joint_step_gen`        : the one-step weighted max-norm contraction
  (generic weight and rate).
* `joint_step`            : the main instantiation.

### Geometric Decay
* `joint_geometric_gen`   : noiseless geometric decay (generic).
* `joint_geometric`       : the main instantiation.
* `joint_geometric_drift_mode` : the pinned-dial specialization.
* `joint_geometric_triangular` : the one-way-coupling specialization.

### Tracking Radius
* `pow_le_one_of_le_one`  : auxiliary power bound.
* `joint_tracking_gen`    : steady-state radius under bounded noise.
* `joint_tracking`        : the main instantiation.
-/

/-! ## Weighted Norm
This section sets the weight and proves the two compatibility bounds
that make the joint iteration contractive: the midpoint of the interval
`(a / beta, eta / b)` lies strictly inside it, which is exactly the
coupling-product condition `a * b < beta * eta`.
-/

/-- The midpoint weight between the two compatibility bounds. -/
noncomputable def joint_r (β η a b : ℝ) : ℝ := (a / β + η / b) / 2

theorem joint_r_pos (β η a b : ℝ)
    (hβ : 0 < β) (hη : 0 < η) (ha : 0 < a) (hb : 0 < b) :
    0 < joint_r β η a b := by
  unfold joint_r
  have h1 : 0 < a / β := div_pos ha hβ
  have h2 : 0 < η / b := div_pos hη hb
  have h3 : 0 < a / β + η / b := add_pos h1 h2
  exact half_pos h3

/-- **Compatible interval nonempty**: the coupling-product condition is
exactly the statement that the two one-dial bounds cross. -/
theorem joint_bounds (β η a b : ℝ)
    (hβ : 0 < β) (hb : 0 < b)
    (hab : a * b < β * η) :
    a / β < η / b := by
  rw [div_lt_div_iff₀ hβ hb, mul_comm η β]
  exact hab

theorem joint_weight_lo (β η a b : ℝ)
    (hβ : 0 < β) (hη : 0 < η) (ha : 0 < a) (hb : 0 < b)
    (hab : a * b < β * η) :
    a / joint_r β η a b < β := by
  have hr : 0 < joint_r β η a b := joint_r_pos β η a b hβ hη ha hb
  rw [div_lt_iff₀ hr]
  unfold joint_r
  have hkey : β * ((a / β + η / b) / 2) = (a + β * η / b) / 2 := by
    field_simp
  rw [hkey]
  have hβb : a < β * η / b := by
    rw [lt_div_iff₀ hb]
    exact hab
  linarith

theorem joint_weight_hi (β η a b : ℝ)
    (hβ : 0 < β) (hη : 0 < η) (ha : 0 < a) (hb : 0 < b)
    (hab : a * b < β * η) :
    joint_r β η a b * b < η := by
  unfold joint_r
  have hkey : ((a / β + η / b) / 2) * b = (a * b / β + η) / 2 := by
    field_simp
  rw [hkey]
  have habβ : a * b / β < η := by
    rw [div_lt_iff₀ hβ, mul_comm η β]
    exact hab
  linarith


/-! ## Rate
This section defines the joint contraction rate as the max of the two
row coefficients and proves it is strictly below one.
-/

/-- The joint contraction rate in the weighted max-norm. -/
noncomputable def joint_rate (β η a b : ℝ) : ℝ :=
  max ((1 - β) + a / joint_r β η a b) ((1 - η) + joint_r β η a b * b)

theorem joint_rate_pos (β η a b : ℝ)
    (hβ : 0 < β) (hβ1 : β ≤ 1) (hη : 0 < η) (ha : 0 < a) (hb : 0 < b) :
    0 < joint_rate β η a b := by
  have hr : 0 < joint_r β η a b := joint_r_pos β η a b hβ hη ha hb
  have h1 : 0 < (1 - β) + a / joint_r β η a b := by
    have hpos : 0 < a / joint_r β η a b := div_pos ha hr
    have hnn : 0 ≤ (1 : ℝ) - β := sub_nonneg.2 hβ1
    linarith
  have h2 : 0 < max ((1 - β) + a / joint_r β η a b)
      ((1 - η) + joint_r β η a b * b) :=
    lt_of_lt_of_le h1 (le_max_left _ _)
  exact h2

theorem joint_rate_lt_one (β η a b : ℝ)
    (hβ : 0 < β) (hη : 0 < η)
    (ha : 0 < a) (hb : 0 < b) (hab : a * b < β * η) :
    joint_rate β η a b < 1 := by
  have hlo := joint_weight_lo β η a b hβ hη ha hb hab
  have hhi := joint_weight_hi β η a b hβ hη ha hb hab
  unfold joint_rate
  rw [max_lt_iff]
  constructor
  · linarith
  · linarith


/-! ## Joint Step
This section contains the one-step contraction: in the weighted
max-norm, the joint error pair decreases at the joint rate plus the
weighted noise. The generic version is parameterized over the weight and
rate; the main instantiation follows.
-/

/-- **One-step joint contraction** (generic weight and rate): in the
weighted max-norm with weight `r`, the coupled error pair satisfies the
single-step bound at rate `γ` plus the weighted perturbation. -/
theorem joint_step_gen (β η a b r γ : ℝ)
    (ew eρ ξw ξρ : ℕ → ℝ)
    (hβ1 : β ≤ 1) (hη1 : η ≤ 1) (ha : 0 ≤ a) (hb : 0 ≤ b) (hr : 0 < r)
    (hγ : γ = max ((1 - β) + a / r) ((1 - η) + r * b))
    (h1 : ∀ m, ew (m + 1) ≤ (1 - β) * ew m + a * eρ m + ξw m)
    (h2 : ∀ m, eρ (m + 1) ≤ b * ew m + (1 - η) * eρ m + ξρ m)
    (hew : ∀ m, 0 ≤ ew m)
    (m : ℕ) :
    max (ew (m + 1)) (r * eρ (m + 1))
      ≤ γ * max (ew m) (r * eρ m) + max (ξw m) (r * ξρ m) := by
  have hMw : ew m ≤ max (ew m) (r * eρ m) := le_max_left _ _
  have hMp : r * eρ m ≤ max (ew m) (r * eρ m) := le_max_right _ _
  have hMnn : 0 ≤ max (ew m) (r * eρ m) :=
    le_trans (hew m) (le_max_left _ _)
  have hpeρ : eρ m ≤ max (ew m) (r * eρ m) / r := by
    rw [le_div_iff₀ hr, mul_comm]
    exact hMp
  have hγ1 : ((1 - β) + a / r) ≤ γ := by rw [hγ]; exact le_max_left _ _
  have hγ2 : ((1 - η) + r * b) ≤ γ := by rw [hγ]; exact le_max_right _ _
  have hbnn : 0 ≤ r * b := mul_nonneg (le_of_lt hr) hb
  have h1βnn : 0 ≤ (1 : ℝ) - β := sub_nonneg.2 hβ1
  have h1ηnn : 0 ≤ (1 : ℝ) - η := sub_nonneg.2 hη1
  have hsum1 : ((1 - β) + a / r) * max (ew m) (r * eρ m)
      = (1 - β) * max (ew m) (r * eρ m)
        + (a / r) * max (ew m) (r * eρ m) := by ring
  have hsum2 : ((1 - η) + r * b) * max (ew m) (r * eρ m)
      = (r * b) * max (ew m) (r * eρ m)
        + (1 - η) * max (ew m) (r * eρ m) := by ring
  have hγM1 : ((1 - β) + a / r) * max (ew m) (r * eρ m)
      ≤ γ * max (ew m) (r * eρ m) :=
    mul_le_mul_of_nonneg_right hγ1 hMnn
  have hγM2 : ((1 - η) + r * b) * max (ew m) (r * eρ m)
      ≤ γ * max (ew m) (r * eρ m) :=
    mul_le_mul_of_nonneg_right hγ2 hMnn
  have harm1 : ew (m + 1)
      ≤ γ * max (ew m) (r * eρ m) + max (ξw m) (r * ξρ m) := by
    have hcross : a * eρ m ≤ (a / r) * max (ew m) (r * eρ m) := by
      have hmul := mul_le_mul_of_nonneg_left hpeρ ha
      have heq : (a / r) * max (ew m) (r * eρ m)
          = a * (max (ew m) (r * eρ m) / r) := by
        field_simp
      rw [heq]
      exact hmul
    have hA : (1 - β) * ew m ≤ (1 - β) * max (ew m) (r * eρ m) :=
      mul_le_mul_of_nonneg_left hMw h1βnn
    have hstep := h1 m
    linarith [hstep, hA, hcross, hsum1, hγM1, le_max_left (ξw m) (r * ξρ m)]
  have harm2 : r * eρ (m + 1)
      ≤ γ * max (ew m) (r * eρ m) + max (ξw m) (r * ξρ m) := by
    have hstep2 := h2 m
    have hmul : r * (b * ew m + (1 - η) * eρ m + ξρ m)
        = (r * b) * ew m + (1 - η) * (r * eρ m) + r * ξρ m := by ring
    have hpre : r * eρ (m + 1)
        ≤ (r * b) * ew m + (1 - η) * (r * eρ m) + r * ξρ m := by
      rw [← hmul]
      exact mul_le_mul_of_nonneg_left hstep2 (le_of_lt hr)
    have hA2 : (r * b) * ew m ≤ (r * b) * max (ew m) (r * eρ m) :=
      mul_le_mul_of_nonneg_left hMw hbnn
    have hB2 : (1 - η) * (r * eρ m) ≤ (1 - η) * max (ew m) (r * eρ m) :=
      mul_le_mul_of_nonneg_left hMp h1ηnn
    linarith [hpre, hA2, hB2, hsum2, hγM2,
      le_max_right (ξw m) (r * ξρ m)]
  exact max_le harm1 harm2

/-- **One-step joint contraction** (main instantiation): the coupled
error pair contracts at `joint_rate` in the `joint_r`-weighted max-norm. -/
theorem joint_step (β η a b : ℝ)
    (ew eρ ξw ξρ : ℕ → ℝ)
    (hβ : 0 < β) (hβ1 : β ≤ 1) (hη : 0 < η) (hη1 : η ≤ 1)
    (ha : 0 < a) (hb : 0 < b)
    (h1 : ∀ m, ew (m + 1) ≤ (1 - β) * ew m + a * eρ m + ξw m)
    (h2 : ∀ m, eρ (m + 1) ≤ b * ew m + (1 - η) * eρ m + ξρ m)
    (hew : ∀ m, 0 ≤ ew m)
    (m : ℕ) :
    max (ew (m + 1)) (joint_r β η a b * eρ (m + 1))
      ≤ joint_rate β η a b * max (ew m) (joint_r β η a b * eρ m)
        + max (ξw m) (joint_r β η a b * ξρ m) :=
  joint_step_gen β η a b (joint_r β η a b) (joint_rate β η a b)
    ew eρ ξw ξρ hβ1 hη1 (le_of_lt ha) (le_of_lt hb)
    (joint_r_pos β η a b hβ hη ha hb) rfl
    h1 h2 hew m


/-! ## Geometric Decay
This section contains the noiseless geometric decay of the joint error
pair, the pinned-dial specialization, and the one-way-coupling
specialization.
-/

/-- **Geometric decay** (generic weight and rate): with zero noise the
joint error pair decays geometrically in the weighted max-norm. -/
theorem joint_geometric_gen (β η a b r γ : ℝ)
    (ew eρ : ℕ → ℝ)
    (hβ1 : β ≤ 1) (hη1 : η ≤ 1) (ha : 0 ≤ a) (hb : 0 ≤ b) (hr : 0 < r)
    (hγ : γ = max ((1 - β) + a / r) ((1 - η) + r * b))
    (h1 : ∀ m, ew (m + 1) ≤ (1 - β) * ew m + a * eρ m)
    (h2 : ∀ m, eρ (m + 1) ≤ b * ew m + (1 - η) * eρ m)
    (hew : ∀ m, 0 ≤ ew m)
    (n : ℕ) :
    max (ew n) (r * eρ n) ≤ γ ^ n * max (ew 0) (r * eρ 0) := by
  have h1' : ∀ m, ew (m + 1) ≤ (1 - β) * ew m + a * eρ m + 0 :=
    fun m => by linarith [h1 m]
  have h2' : ∀ m, eρ (m + 1) ≤ b * ew m + (1 - η) * eρ m + 0 :=
    fun m => by linarith [h2 m]
  have hγpos : 0 ≤ γ := by
    have har : 0 ≤ a / r := div_nonneg ha (le_of_lt hr)
    have h1β : 0 ≤ (1 : ℝ) - β := sub_nonneg.2 hβ1
    have hpos : 0 ≤ (1 - β) + a / r := by linarith
    have hle : (1 - β) + a / r ≤ γ := by rw [hγ]; exact le_max_left _ _
    linarith
  have hstep : ∀ m, max (ew (m + 1)) (r * eρ (m + 1))
      ≤ γ * max (ew m) (r * eρ m) :=
    fun m => by
      have h := joint_step_gen β η a b r γ ew eρ (fun _ => 0) (fun _ => 0)
        hβ1 hη1 ha hb hr hγ h1' h2' hew m
      simpa using h
  induction n with
  | zero => simp
  | succ n ih =>
    have h := hstep n
    rw [pow_succ]
    calc max (ew (n + 1)) (r * eρ (n + 1))
        ≤ γ * max (ew n) (r * eρ n) := h
      _ ≤ γ * (γ ^ n * max (ew 0) (r * eρ 0)) :=
          mul_le_mul_of_nonneg_left ih hγpos
      _ = γ ^ n * γ * max (ew 0) (r * eρ 0) := by ring

/-- **Geometric decay** (main instantiation). -/
theorem joint_geometric (β η a b : ℝ)
    (ew eρ : ℕ → ℝ)
    (hβ : 0 < β) (hβ1 : β ≤ 1) (hη : 0 < η) (hη1 : η ≤ 1)
    (ha : 0 < a) (hb : 0 < b)
    (h1 : ∀ m, ew (m + 1) ≤ (1 - β) * ew m + a * eρ m)
    (h2 : ∀ m, eρ (m + 1) ≤ b * ew m + (1 - η) * eρ m)
    (hew : ∀ m, 0 ≤ ew m)
    (n : ℕ) :
    max (ew n) (joint_r β η a b * eρ n)
      ≤ joint_rate β η a b ^ n * max (ew 0) (joint_r β η a b * eρ 0) :=
  joint_geometric_gen β η a b (joint_r β η a b) (joint_rate β η a b)
    ew eρ hβ1 hη1 (le_of_lt ha) (le_of_lt hb)
    (joint_r_pos β η a b hβ hη ha hb) rfl
    h1 h2 hew n

/-- **Pinned-dial specialization**: with the strength-dial error
identically zero (the drift-mode pin), the window error alone decays
geometrically. -/
theorem joint_geometric_drift_mode (β η a b : ℝ)
    (ew : ℕ → ℝ)
    (hβ : 0 < β) (hβ1 : β ≤ 1) (hη : 0 < η) (hη1 : η ≤ 1)
    (ha : 0 < a) (hb : 0 < b)
    (h1 : ∀ m, ew (m + 1) ≤ (1 - β) * ew m)
    (hew : ∀ m, 0 ≤ ew m)
    (n : ℕ) :
    ew n ≤ joint_rate β η a b ^ n * ew 0 := by
  have h2 : ∀ m, (0 : ℝ) ≤ b * ew m + (1 - η) * 0 := by
    intro m
    have h := mul_nonneg (le_of_lt hb) (hew m)
    simp only [mul_zero, add_zero]
    exact h
  have h := joint_geometric β η a b ew (fun _ => 0)
    hβ hβ1 hη hη1 ha hb
    (fun m => by simpa using h1 m) h2 hew n
  simp only [mul_zero] at h
  rw [max_eq_left (hew n), max_eq_left (hew 0)] at h
  exact h

/-- **Pinned strength dial, free window rate**: when the strength-dial
error is identically zero, the window error contracts at the free dial's
own rate `1−β`, not at the slower joint rate of
`joint_geometric_drift_mode`. -/
theorem joint_geometric_pinned (β : ℝ) (ew : ℕ → ℝ)
    (_hβ : 0 < β) (hβ1 : β ≤ 1)
    (h1 : ∀ m, ew (m + 1) ≤ (1 - β) * ew m)
    (n : ℕ) :
    ew n ≤ (1 - β) ^ n * ew 0 := by
  have hnonneg : 0 ≤ 1 - β := by linarith
  induction n with
  | zero => simp
  | succ n ih =>
      calc ew (n + 1) ≤ (1 - β) * ew n := h1 n
        _ ≤ (1 - β) * ((1 - β) ^ n * ew 0) :=
            mul_le_mul_of_nonneg_left ih hnonneg
        _ = (1 - β) ^ (n + 1) * ew 0 := by
            rw [pow_succ]
            ring

/-- **One-way-coupling specialization** (`b = 0`): when the strength
dial does not see the window dial, the weight `r = 2 * a / β` and rate
`max (1 - β / 2) (1 - η)` give the joint contraction. -/
theorem joint_geometric_triangular (β η a : ℝ)
    (ew eρ : ℕ → ℝ)
    (hβ : 0 < β) (hβ1 : β ≤ 1) (hη1 : η ≤ 1)
    (ha : 0 < a)
    (h1 : ∀ m, ew (m + 1) ≤ (1 - β) * ew m + a * eρ m)
    (h2 : ∀ m, eρ (m + 1) ≤ (1 - η) * eρ m)
    (hew : ∀ m, 0 ≤ ew m)
    (n : ℕ) :
    max (ew n) ((2 * a / β) * eρ n)
      ≤ (max (1 - β / 2) (1 - η)) ^ n
        * max (ew 0) ((2 * a / β) * eρ 0) := by
  have hr : 0 < 2 * a / β := by
    apply div_pos _ hβ
    linarith
  have hγ : max (1 - β / 2) (1 - η)
      = max ((1 - β) + a / (2 * a / β)) ((1 - η) + (2 * a / β) * 0) := by
    have ha2β : a / (2 * a / β) = β / 2 := by
      field_simp
    rw [ha2β, mul_zero, add_zero]
    congr 1
    ring
  have h2' : ∀ m, eρ (m + 1) ≤ 0 * ew m + (1 - η) * eρ m :=
    fun m => by simpa using h2 m
  exact joint_geometric_gen β η a 0 (2 * a / β) (max (1 - β / 2) (1 - η))
    ew eρ hβ1 hη1 (le_of_lt ha) (le_refl 0) hr hγ
    h1 h2' hew n


/-! ## Tracking Radius
This section contains the steady-state bound: under bounded weighted
noise, the joint error pair stays within a geometric transient plus the
noise-to-gap ratio.
-/

/-- Auxiliary: powers of a number in `[0, 1]` stay in `[0, 1]`. -/
theorem pow_le_one_of_le_one (a : ℝ) (h0 : 0 ≤ a) (h1 : a ≤ 1) :
    ∀ n : ℕ, a ^ n ≤ 1 := by
  intro n
  induction n with
  | zero => simp
  | succ n ih =>
    rw [pow_succ]
    have h1' := mul_le_mul_of_nonneg_left ih h0
    have h2' : a ^ n * a ≤ 1 := by
      have ha1 : a ^ n * a ≤ 1 * a := mul_le_mul_of_nonneg_right ih h0
      have ha2 : (1 : ℝ) * a ≤ 1 := by nlinarith
      linarith [ha1, ha2]
    exact h2'

/-- **Tracking radius** (generic weight and rate): with weighted noise
bounded by `ξ` and rate below one, the joint error pair stays within the
geometric transient plus `ξ / (1 - γ)`. -/
theorem joint_tracking_gen (β η a b r γ ξ : ℝ)
    (ew eρ ξw ξρ : ℕ → ℝ)
    (hβ1 : β ≤ 1) (hη1 : η ≤ 1) (ha : 0 ≤ a) (hb : 0 ≤ b) (hr : 0 < r)
    (hγ : γ = max ((1 - β) + a / r) ((1 - η) + r * b))
    (hγlt : γ < 1)
    (h1 : ∀ m, ew (m + 1) ≤ (1 - β) * ew m + a * eρ m + ξw m)
    (h2 : ∀ m, eρ (m + 1) ≤ b * ew m + (1 - η) * eρ m + ξρ m)
    (hew : ∀ m, 0 ≤ ew m)
    (hξ : ∀ m, max (ξw m) (r * ξρ m) ≤ ξ) (hξnn : 0 ≤ ξ)
    (n : ℕ) :
    max (ew n) (r * eρ n)
      ≤ γ ^ n * max (ew 0) (r * eρ 0) + ξ / (1 - γ) := by
  have hγpos : 0 ≤ γ := by
    have har : 0 ≤ a / r := div_nonneg ha (le_of_lt hr)
    have h1β : 0 ≤ (1 : ℝ) - β := sub_nonneg.2 hβ1
    have hpos : 0 ≤ (1 - β) + a / r := by linarith
    have hle : (1 - β) + a / r ≤ γ := by rw [hγ]; exact le_max_left _ _
    linarith
  have h1γ : (1 : ℝ) - γ > 0 := by linarith
  have hstep : ∀ m, max (ew (m + 1)) (r * eρ (m + 1))
      ≤ γ * max (ew m) (r * eρ m) + ξ := fun m => by
    have h := joint_step_gen β η a b r γ ew eρ ξw ξρ
      hβ1 hη1 ha hb hr hγ h1 h2 hew m
    have hnoise : max (ξw m) (r * ξρ m) ≤ ξ := hξ m
    linarith [h, hnoise]
  have main : ∀ n : ℕ,
      max (ew n) (r * eρ n)
        ≤ γ ^ n * max (ew 0) (r * eρ 0)
          + ξ * (1 - γ ^ n) / (1 - γ) := by
    intro n
    induction n with
    | zero => simp
    | succ n ih =>
      have h := hstep n
      rw [pow_succ]
      have halg : γ * (γ ^ n * max (ew 0) (r * eρ 0)
          + ξ * (1 - γ ^ n) / (1 - γ)) + ξ
          = γ ^ n * γ * max (ew 0) (r * eρ 0)
            + ξ * (1 - γ ^ n * γ) / (1 - γ) := by
        field_simp
        ring
      calc max (ew (n + 1)) (r * eρ (n + 1))
          ≤ γ * max (ew n) (r * eρ n) + ξ := h
        _ ≤ γ * (γ ^ n * max (ew 0) (r * eρ 0)
            + ξ * (1 - γ ^ n) / (1 - γ)) + ξ :=
            add_le_add (mul_le_mul_of_nonneg_left ih hγpos) (le_refl _)
        _ = γ ^ n * γ * max (ew 0) (r * eρ 0)
            + ξ * (1 - γ ^ n * γ) / (1 - γ) := halg
  have hfinal := main n
  have hgnn : 0 ≤ γ ^ n := pow_nonneg hγpos n
  have hgn1 : γ ^ n ≤ 1 := pow_le_one_of_le_one γ hγpos (le_of_lt hγlt) n
  have hconv : ξ * (1 - γ ^ n) / (1 - γ) ≤ ξ / (1 - γ) := by
    have hnum : 1 - γ ^ n ≤ 1 := by linarith
    have hmul : ξ * (1 - γ ^ n) ≤ ξ * 1 :=
      mul_le_mul_of_nonneg_left hnum hξnn
    rw [mul_one] at hmul
    rw [div_le_div_iff₀ h1γ h1γ]
    exact mul_le_mul_of_nonneg_right hmul (le_of_lt h1γ)
  linarith [hfinal, hconv]

/-- **Tracking radius** (main instantiation). -/
theorem joint_tracking (β η a b ξ : ℝ)
    (ew eρ ξw ξρ : ℕ → ℝ)
    (hβ : 0 < β) (hβ1 : β ≤ 1) (hη : 0 < η) (hη1 : η ≤ 1)
    (ha : 0 < a) (hb : 0 < b) (hab : a * b < β * η)
    (h1 : ∀ m, ew (m + 1) ≤ (1 - β) * ew m + a * eρ m + ξw m)
    (h2 : ∀ m, eρ (m + 1) ≤ b * ew m + (1 - η) * eρ m + ξρ m)
    (hew : ∀ m, 0 ≤ ew m)
    (hξ : ∀ m, max (ξw m) (joint_r β η a b * ξρ m) ≤ ξ) (hξnn : 0 ≤ ξ)
    (n : ℕ) :
    max (ew n) (joint_r β η a b * eρ n)
      ≤ joint_rate β η a b ^ n * max (ew 0) (joint_r β η a b * eρ 0)
        + ξ / (1 - joint_rate β η a b) :=
  joint_tracking_gen β η a b (joint_r β η a b) (joint_rate β η a b) ξ
    ew eρ ξw ξρ hβ1 hη1 (le_of_lt ha) (le_of_lt hb)
    (joint_r_pos β η a b hβ hη ha hb) rfl
    (joint_rate_lt_one β η a b hβ hη ha hb hab)
    h1 h2 hew hξ hξnn n
