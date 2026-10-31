import Mathlib

/-!
# SPAR (Spectrally-Adaptive Anchor Refresh): algebraic core

Machine-checked algebraic core of the SPAR theory chapter
(`theory/ch4_spar_theorems.tex`). Every statement is pure algebra or
elementary real analysis over ℝ; no probability is mechanized.

## Table of contents

### Pull Identity
* `pull_identity`                : exact per-direction decomposition (γ = 1)
* `pull_identity_general`       : the same, with anchor shrinkage γ

### Interpolation and Calibration
* `convex_interp_sq`            : exact interpolation identity, excess = s(1−s)(u−d)²
* `convex_interp_sq_le`         : squared convex combination ≤ linear interpolation
* `quadratic_calibration`       : R(s) = R_min + (v+c)(s−s*)²
* `quadratic_calibration_delta` : δ-parameterized form (c = δ²)
* `optimal_pull_min`           : minimal value is a lower bound
* `optimal_pull_attained`      : s* = v/(v+c) attains the minimum

### Oracle Dominance
* `dominance`                   : (1−s)²·v ≤ v on [0,1]
* `dominance_strict`            : strict in the interior

### Minimax Calibration
* `minimax_lower`               : lumped minimax value DvC/(Dv+C)

### Safety Threshold
* `safety_threshold`            : R(s) ≤ v ↔ s ≤ 2s* (multiplicative form)

### Pull Inversion
* `pull_inverse`                : ρ ↔ s inversion, ρ* = μv/c

### Tracking
* `lag_invariance`              : continuous-refresh equilibrium adds no lag

### Finite-Cadence Refresh Equilibrium
* `refresh_rate_identity`       : a = s(2−s) = 1 − (1−s)²
* `refresh_fixed_point`         : g* = −L − νk·a/(1−a)
* `refresh_closed_form`         : geometric closed form with rate a^m
* `refresh_contraction_rate`    : a ∈ (0,1) for s ∈ (0,1)

### Dial Fixed-Point Dichotomy
* `naive_dial_no_fixed_point`   : naive dial cubed value at least doubles per step
* `debiased_dial_contraction`   : debiased dial contracts geometrically to the optimum

### Correlated Finite-Cadence Variance
* `refresh_anchor_variance`     : steady-state anchor variance system
* `refresh_variance_form`       : closed-form correlated solution variance
* `refresh_variance_lt`         : strict variance reduction at every cadence
-/

/-! ## Pull Identity
This section contains the exact per-direction decomposition of the
anchored solution's error into data error and anchor error, with and
without anchor shrinkage γ.
-/

/-- **Pull identity (γ = 1)**: the anchored solution's error is the convex
combination of the data error and the anchor error, per eigen-direction.
`μ` is the excitation, `ρ` the anchor strength, `b` the accumulated
cross-moment, `g` the anchor coordinate, `w` the truth. -/
theorem pull_identity (μ ρ b g w : ℝ) (hμ : μ ≠ 0) (hden : μ + ρ ≠ 0) :
    (b + ρ * g) / (μ + ρ) - w
      = (μ / (μ + ρ)) * (b / μ - w) + (ρ / (μ + ρ)) * (g - w) := by
  field_simp
  ring

/-- **Pull identity (general γ)**: with anchor weight γ the pulled anchor term
carries the shrinkage factor γ — this is exactly why γ < 1 is biased. -/
theorem pull_identity_general (μ ρ b g w γ : ℝ) (hμ : μ ≠ 0) (hden : μ + ρ ≠ 0) :
    (b + ρ * γ * g) / (μ + ρ) - w
      = (μ / (μ + ρ)) * (b / μ - w) + (ρ / (μ + ρ)) * (γ * g - w) := by
  field_simp
  ring


/-! ## Interpolation and Calibration
This section contains the interpolation identity for squared convex
combinations and the quadratic calibration identity
`R(s) = (1−s)²·v + s²·c = v·c/(v+c) + (v+c)·(s−s*)²`, together with its
two corollaries: the minimal value is a lower bound, and the optimal pull
`s* = v/(v+c)` attains it.
-/

/-- Exact identity behind the interpolation bound: the squared convex
combination falls short of the linear interpolation of the squares by
exactly `s(1−s)(u−d)²`. -/
theorem convex_interp_sq (s u d : ℝ) :
    (1 - s) * u ^ 2 + s * d ^ 2 - ((1 - s) * u + s * d) ^ 2
      = s * (1 - s) * (u - d) ^ 2 := by
  ring

/-- The squared convex combination never exceeds the linear interpolation of
the squares, for `s ∈ [0,1]`. -/
theorem convex_interp_sq_le (s u d : ℝ) (hs : 0 ≤ s) (hs' : s ≤ 1) :
    ((1 - s) * u + s * d) ^ 2 ≤ (1 - s) * u ^ 2 + s * d ^ 2 := by
  have h := convex_interp_sq s u d
  have hsd : 0 ≤ s * (1 - s) * (u - d) ^ 2 :=
    mul_nonneg (mul_nonneg hs (by linarith)) (sq_nonneg _)
  linarith

/-- **Calibration identity**: the per-direction risk
`R(s) = (1−s)²·v + s²·c` (data variance `v`, prior squared error `c`)
equals its minimum `v·c/(v+c)` plus the quadratic excess
`(v+c)·(s − v/(v+c))²`. -/
theorem quadratic_calibration (v c s : ℝ) (hvc : v + c ≠ 0) :
    (1 - s) ^ 2 * v + s ^ 2 * c
      = v * c / (v + c) + (v + c) * (s - v / (v + c)) ^ 2 := by
  field_simp
  ring

/-- δ-parameterized form of the calibration identity (c = δ²). -/
theorem quadratic_calibration_delta (v δ s : ℝ) (hvδ : v + δ ^ 2 ≠ 0) :
    (1 - s) ^ 2 * v + s ^ 2 * δ ^ 2
      = v * δ ^ 2 / (v + δ ^ 2) + (v + δ ^ 2) * (s - v / (v + δ ^ 2)) ^ 2 :=
  quadratic_calibration v (δ ^ 2) s hvδ

/-- The minimal risk `v·c/(v+c)` is a lower bound for the risk at any pull
`s`, whenever `v > 0` and `c ≥ 0`. -/
theorem optimal_pull_min (v c s : ℝ) (hv : 0 < v) (hc : 0 ≤ c) :
    v * c / (v + c) ≤ (1 - s) ^ 2 * v + s ^ 2 * c := by
  have hvc : v + c ≠ 0 := by linarith
  rw [quadratic_calibration v c s hvc]
  have hnn : 0 ≤ (v + c) * (s - v / (v + c)) ^ 2 :=
    mul_nonneg (by linarith) (sq_nonneg _)
  linarith

/-- The optimal pull `s* = v/(v+c)` attains the minimal risk. -/
theorem optimal_pull_attained (v c : ℝ) (hv : 0 < v) (hc : 0 ≤ c) :
    (1 - v / (v + c)) ^ 2 * v + (v / (v + c)) ^ 2 * c = v * c / (v + c) := by
  have hvc : v + c ≠ 0 := by linarith
  have h := quadratic_calibration v c (v / (v + c)) hvc
  have h0 : (v / (v + c) - v / (v + c)) ^ 2 = 0 := by ring
  rw [h0] at h
  simp only [mul_zero, add_zero] at h
  linarith [h]


/-! ## Oracle Dominance
This section contains the oracle dominance bounds: with a correct anchor,
the anchored variance `(1−s)²·v` never exceeds the data variance `v`, and
the inequality is strict in the interior.
-/

/-- With a correct anchor, the anchored variance `(1−s)²·v` never exceeds
the data variance `v`, for every pull `s ∈ [0,1]`. -/
theorem dominance (s v : ℝ) (hs : 0 ≤ s) (hs' : s ≤ 1) (hv : 0 ≤ v) :
    (1 - s) ^ 2 * v ≤ v := by
  have h2 : (1 - s) ^ 2 ≤ 1 := by nlinarith
  have h3 : (1 - s) ^ 2 * v ≤ 1 * v := mul_le_mul_of_nonneg_right h2 hv
  simpa using h3

/-- Strict dominance in the interior `0 < s < 1` with `v > 0`. -/
theorem dominance_strict (s v : ℝ) (hs : 0 < s) (hs' : s < 1) (hv : 0 < v) :
    (1 - s) ^ 2 * v < v := by
  have h2 : (1 - s) ^ 2 < 1 := by nlinarith
  have h3 : (1 - s) ^ 2 * v < 1 * v := mul_lt_mul_of_pos_right h2 hv
  simpa using h3


/-! ## Minimax Calibration
This section contains the lumped minimax lower bound of the isotropic
problem.
-/

/-- **Lumped minimax lower bound**: in the isotropic lumped problem with
total data risk `D·v` and conflict budget `C`, every fixed pull `s` pays at
least the harmonic mean `DvC/(Dv+C)`, attained at `s* = Dv/(Dv+C)` (see
`optimal_pull_attained`). Together with monotonicity of the value in `C`
(single-variable calculus, paper text), this is the computational core of
`min_s max_{C'≤C} R = max_{C'≤C} min_s R = DvC/(Dv+C)`. -/
theorem minimax_lower (D v C s : ℝ) (hD : 0 < D) (hv : 0 < v) (hC : 0 < C) :
    D * v * C / (D * v + C) ≤ (1 - s) ^ 2 * (D * v) + s ^ 2 * C :=
  optimal_pull_min (D * v) C s (by positivity) (le_of_lt hC)


/-! ## Safety Threshold
This section contains the safety threshold characterization: for a
nonnegative pull the anchored risk never exceeds the data-path variance
exactly when the pull undershoots `2s*`.
-/

/-- **Safety threshold**: for a nonnegative pull `s`, the anchored risk
`(1−s)²·v + s²·c` never exceeds the data-path variance `v` if and only if
`s·(v+c) ≤ 2·v`, i.e. `s ≤ 2s*` at the optimal pull `s* = v/(v+c)`.
Underestimating the pull is safe; overshooting the threshold is punished
quadratically by the calibration identity. -/
theorem safety_threshold (v c s : ℝ) (hv : 0 < v) (hc : 0 ≤ c) (hs : 0 ≤ s) :
    (1 - s) ^ 2 * v + s ^ 2 * c ≤ v ↔ s * (v + c) ≤ 2 * v := by
  have hkey : (1 - s) ^ 2 * v + s ^ 2 * c - v = s * (s * (v + c) - 2 * v) := by
    ring
  constructor
  · intro h
    by_contra hgt
    have hgt2 : 2 * v < s * (v + c) := lt_of_not_ge hgt
    have hs0 : 0 < s := by
      by_contra hs1
      have hs2 : s ≤ 0 := le_of_not_gt hs1
      have hzc : s * (v + c) ≤ 0 := by nlinarith
      linarith
    have hxp : 0 < s * (v + c) - 2 * v := by linarith
    have hpos : 0 < s * (s * (v + c) - 2 * v) := by nlinarith
    linarith
  · intro h
    have hf : s * (v + c) - 2 * v ≤ 0 := by linarith
    have hle : s * (s * (v + c) - 2 * v) ≤ 0 := by
      nlinarith
    linarith


/-! ## Pull Inversion
This section contains the inversion between anchor strength `ρ` and pull
`s` at the optimum: `ρ* = μ·v/c`.
-/

/-- **Pull inversion** (multiplicative form): at the optimal pull
`s* = v/(v+c)` the per-direction pull `s = ρ/(μ+ρ)` inverts to
`ρ = μ·s/(1−s)`; here `μ·s* = ρ*·(1−s*)` with `ρ* = μ·v/c`, i.e. the
anchor strength equals `μ·v/c` at the optimum. -/
theorem pull_inverse (μ v c : ℝ) (hvc : v + c ≠ 0) (hc : c ≠ 0) :
    μ * (v / (v + c)) = (μ * v / c) * (1 - v / (v + c)) := by
  have h1 : 1 - v / (v + c) = c / (v + c) := by
    field_simp
    ring
  rw [h1]
  field_simp


/-! ## Tracking
This section contains the tracking analysis: at the continuous-refresh
equilibrium the anchor contributes no additional lag.
-/

/-- At the continuous-refresh equilibrium the anchored solution satisfies
`W − w* = (1−s)(−L) + s(W − w*)`, where `L` is the data path's lag. For
`s ≠ 1` the unique solution is `W − w* = −L`: the anchor contributes
**no** additional tracking lag. -/
theorem lag_invariance (s L x : ℝ) (hs : s ≠ 1)
    (h : (1 - s) * (-L) + s * x = x) : x = -L := by
  have h2 : (s - 1) * x = (s - 1) * (-L) := by
    linear_combination (norm := ring_nf) h
  exact mul_left_cancel₀ (sub_ne_zero.mpr hs) h2


/-! ## Finite-Cadence Refresh Equilibrium
This section contains the finite-cadence refresh equilibrium: the per-cycle
contraction factor, the moving-frame fixed point, the geometric closed
form of the affine refresh recursion, and the contraction rate.
-/

/-- **Refresh rate identity**: the per-cycle refresh contraction factor
`a = s(2−s)` equals `1 − (1−s)²`. -/
theorem refresh_rate_identity (s : ℝ) : 1 - s * (2 - s) = (1 - s) ^ 2 := by
  ring

/-- **Finite-cadence fixed point**: with pull `s`, data-path lag `L`,
target velocity `ν`, and cadence `k`, the moving-frame anchor offset
`g = G − w*(t)` at the fixed point satisfies
`(1−a)·g* = −(1−a)·L − a·ν·k` with `a = s(2−s)`, i.e.
`g* = −L − νk·a/(1−a)` (requires `s ≠ 1`). -/
theorem refresh_fixed_point (s L ν k : ℝ) (ha : 1 - s * (2 - s) ≠ 0) :
    (1 - s * (2 - s)) * (-L - ν * k * s * (2 - s) / (1 - s * (2 - s)))
      = - (1 - s * (2 - s)) * L - s * (2 - s) * ν * k := by
  field_simp

/-- **Closed form of the affine refresh recursion**: if the anchor offsets
follow `g_{m+1} = a·g_m + b` from `g₀`, then for every `m`
`g_m = g* + a^m·(g₀ − g*)` with fixed point `g* = b/(1−a)` — the
geometric convergence of Theorem 2(i) at rate `a` per refresh cycle. -/
theorem refresh_closed_form (a b g0 gstar : ℝ) (ha : a ≠ 1)
    (hgs : gstar = b / (1 - a)) (m : ℕ)
    (g : ℕ → ℝ) (hg0 : g 0 = g0)
    (hrec : ∀ n, g (n + 1) = a * g n + b) :
    g m = gstar + a ^ m * (g0 - gstar) := by
  have hfix : a * gstar + b = gstar := by
    rw [hgs]
    field_simp
    ring
  induction m with
  | zero => rw [hg0, pow_zero]; ring
  | succ m ih =>
      rw [hrec, ih, pow_succ]
      linear_combination hfix

/-- **Contraction rate**: for every pull `s ∈ (0,1)` the per-cycle refresh
factor `a = s(2−s) = 1 − (1−s)²` lies in `(0,1)`, so the closed form of
`refresh_closed_form` converges geometrically. -/
theorem refresh_contraction_rate (s : ℝ) (hs0 : 0 < s) (hs1 : s < 1) :
    0 < s * (2 - s) ∧ s * (2 - s) < 1 := by
  constructor <;> nlinarith


/-! ## Dial Fixed-Point Dichotomy
This section contains the dial dichotomy: the naive window dial has no
fixed point (its cubed value at least doubles each step), while the
debiased dial contracts geometrically to the optimum.
-/

/-- **Naive dial has no fixed point**: the undebiased window dial
`T_naive(w) = ((w*)^3 + 4w^3/(2-w))^(1/3)` has cubed value exceeding
`2w^3` for every `w ∈ (0,2)`, so the cubed trajectory at least doubles
each iteration and hits the cap in logarithmically many steps. -/
theorem naive_dial_no_fixed_point (w wstar : ℝ)
    (hw0 : 0 < w) (hw2 : w < 2) (hwstar : 0 ≤ wstar) :
    wstar ^ 3 + 4 * w ^ 3 / (2 - w) > 2 * w ^ 3 := by
  have h1 : 0 < 2 - w := by linarith
  have hkey : 4 * w ^ 3 / (2 - w) - 2 * w ^ 3 = 2 * w ^ 4 / (2 - w) := by
    field_simp
    ring
  have h2 : 0 < 2 * w ^ 4 / (2 - w) := by positivity
  have h3 : 4 * w ^ 3 / (2 - w) > 2 * w ^ 3 := by
    have h4 : 0 < 4 * w ^ 3 / (2 - w) - 2 * w ^ 3 := by
      rw [hkey]
      exact h2
    linarith
  have h5 : 0 ≤ wstar ^ 3 := by positivity
  linarith

/-- **Debiased dial contraction**: with the variance-floor-subtracted
estimator, the cubed window state converges geometrically to the optimum
`ustar` from any initialization. -/
theorem debiased_dial_contraction (β u0 ustar : ℝ) (m : ℕ)
    (u : ℕ → ℝ) (hu0 : u 0 = u0)
    (hrec : ∀ n, u (n + 1) = (1 - β) * u n + β * ustar) :
    u m = ustar + (1 - β) ^ m * (u0 - ustar) := by
  induction m with
  | zero => rw [hu0]; simp [pow_zero]
  | succ m ih =>
      have hfix : (1 - β) * ustar + β * ustar = ustar := by ring
      rw [hrec, ih, pow_succ]
      linear_combination hfix


/-! ## Correlated Finite-Cadence Variance
This section contains the correlated finite-cadence variance analysis:
the steady-state anchor variance system, its closed-form solution
variance, and the strict variance reduction at every finite cadence.
-/

/-- **Correlated anchor variance**: the closed forms
`q = (1−s)²·v·(1+aρ)/((1+a)(1−aρ))` (anchor variance) and
`C = (1−s)²·ρ·v/(1−aρ)` (cross-covariance, the collapsed geometric sum
of the cross-cycle noise correlations) satisfy the steady-state
variance system `q = (1−s)⁴·v + a²·q + 2a·(1−s)²·C`; the system is
linear in `q`, so the solution is unique. -/
theorem refresh_anchor_variance (s ρ v : ℝ)
    (hne1 : 1 + s * (2 - s) ≠ 0) (hne : 1 - s * (2 - s) * ρ ≠ 0) :
    (1 - s) ^ 4 * v + (s * (2 - s)) ^ 2
        * ((1 - s) ^ 2 * v * (1 + s * (2 - s) * ρ)
             / ((1 + s * (2 - s)) * (1 - s * (2 - s) * ρ)))
      + 2 * s * (2 - s) * (1 - s) ^ 2
        * ((1 - s) ^ 2 * ρ * v / (1 - s * (2 - s) * ρ))
    = (1 - s) ^ 2 * v * (1 + s * (2 - s) * ρ)
          / ((1 + s * (2 - s)) * (1 - s * (2 - s) * ρ)) := by
  field_simp
  ring

/-- **Correlated solution variance**: with the anchor variance and
cross-covariance closed forms, the solution variance
`(1−s)²·v + s²·q + 2s(1−s)·C` equals
`v·(1 − 2s(1+s−s²)(1−ρ)/((1+a)(1−aρ)))`. -/
theorem refresh_variance_form (s ρ v : ℝ)
    (hne1 : 1 + s * (2 - s) ≠ 0) (hne : 1 - s * (2 - s) * ρ ≠ 0) :
    (1 - s) ^ 2 * v
      + s ^ 2 * ((1 - s) ^ 2 * v * (1 + s * (2 - s) * ρ)
             / ((1 + s * (2 - s)) * (1 - s * (2 - s) * ρ)))
      + 2 * s * (1 - s) * ((1 - s) ^ 2 * ρ * v
             / (1 - s * (2 - s) * ρ))
    = v * (1 - 2 * s * (1 + s - s ^ 2) * (1 - ρ)
             / ((1 + s * (2 - s)) * (1 - s * (2 - s) * ρ))) := by
  field_simp
  ring

/-- **Strict variance reduction at every finite cadence**: the correlated
solution variance is strictly below `v` for every pull `s ∈ (0,1)` and
every cross-cycle correlation `ρ ∈ [0,1)`. -/
theorem refresh_variance_lt (s ρ v : ℝ) (hs0 : 0 < s) (hs1 : s < 1)
    (hv : 0 < v) (hρ1 : ρ < 1) :
    v * (1 - 2 * s * (1 + s - s ^ 2) * (1 - ρ)
             / ((1 + s * (2 - s)) * (1 - s * (2 - s) * ρ))) < v := by
  have h1 : 0 < 1 + s * (2 - s) := by nlinarith
  have h2 : 0 < 1 + s - s ^ 2 := by nlinarith
  have h3 : 0 < 1 - s * (2 - s) := by nlinarith
  have ha0 : 0 ≤ s * (2 - s) := by nlinarith
  have harho : s * (2 - s) * ρ < 1 := by
    have h5 : s * (2 - s) * ρ ≤ s * (2 - s) * 1 :=
      mul_le_mul_of_nonneg_left hρ1.le ha0
    have h6 : s * (2 - s) * 1 < 1 := by
      have h7 : s * (2 - s) < 1 := by linarith
      linarith
    linarith
  have h4 : 0 < 1 - s * (2 - s) * ρ := by linarith
  have hn : 0 < 2 * s * (1 + s - s ^ 2) * (1 - ρ) :=
    mul_pos (mul_pos (by nlinarith : 0 < 2 * s) h2)
      (by linarith : 0 < 1 - ρ)
  have hd : 0 < (1 + s * (2 - s)) * (1 - s * (2 - s) * ρ) :=
    mul_pos h1 h4
  have hfrac : 0 < 2 * s * (1 + s - s ^ 2) * (1 - ρ)
      / ((1 + s * (2 - s)) * (1 - s * (2 - s) * ρ)) :=
    div_pos hn hd
  have hvfrac : 0 < v * (2 * s * (1 + s - s ^ 2) * (1 - ρ)
      / ((1 + s * (2 - s)) * (1 - s * (2 - s) * ρ))) :=
    mul_pos hv hfrac
  linarith

/-- **Lower variance endpoint**: the correlated solution variance is at
least `(1−s)²·v` for every pull `s ∈ (0,1)` and correlation `ρ ∈ [0,1)`,
so the frozen-anchor plug-in `c'` lower-bounds the exact variance and the
window dial errs toward larger (safer) windows.  The factorisation is
`Ψ − (1−s)² = s(1−s)²(ρ(s³−4s²+2s+2)+s)/((1+a)(1−aρ))`, and
`s³−4s²+2s+2 = 1 + (s−1)(s²−3s−1) ≥ 1` on `[0,1]`. -/
theorem refresh_variance_lb (s ρ v : ℝ) (hs0 : 0 < s) (hs1 : s < 1)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (hv : 0 ≤ v) :
    (1 - s) ^ 2 * v ≤
      v * (1 - 2 * s * (1 + s - s ^ 2) * (1 - ρ)
             / ((1 + s * (2 - s)) * (1 - s * (2 - s) * ρ))) := by
  have h1 : 0 < 1 + s * (2 - s) := by nlinarith
  have ha0 : 0 ≤ s * (2 - s) := by nlinarith
  have harho : s * (2 - s) * ρ < 1 := by
    have h5 : s * (2 - s) * ρ ≤ s * (2 - s) * 1 :=
      mul_le_mul_of_nonneg_left hρ1.le ha0
    have h6 : s * (2 - s) * 1 < 1 := by
      have h7 : s * (2 - s) < 1 := by nlinarith
      linarith
    linarith
  have h4 : 0 < 1 - s * (2 - s) * ρ := by linarith
  have hd : 0 < (1 + s * (2 - s)) * (1 - s * (2 - s) * ρ) :=
    mul_pos h1 h4
  set D : ℝ := (1 + s * (2 - s)) * (1 - s * (2 - s) * ρ) with hDdef
  have hDpos : 0 < D := by rw [hDdef]; exact hd
  have hs2 : s ^ 2 ≤ 1 := by nlinarith
  have hA : s - 1 ≤ 0 := by linarith
  have hB : s ^ 2 - 3 * s - 1 ≤ 0 := by nlinarith
  have hprod : 0 ≤ (s - 1) * (s ^ 2 - 3 * s - 1) :=
    mul_nonneg_of_nonpos_of_nonpos hA hB
  have hf : 0 ≤ s ^ 3 - 4 * s ^ 2 + 2 * s + 2 := by nlinarith
  have hnum : 0 ≤ ρ * (s ^ 3 - 4 * s ^ 2 + 2 * s + 2) + s := by
    have := mul_nonneg hρ0 hf
    linarith
  have hfac : D - 2 * s * (1 + s - s ^ 2) * (1 - ρ) - (1 - s) ^ 2 * D
      = s * (1 - s) ^ 2
          * (ρ * (s ^ 3 - 4 * s ^ 2 + 2 * s + 2) + s) := by
    rw [hDdef]
    ring
  have hmain : (1 - s) ^ 2 * v * D
      ≤ v * (D - 2 * s * (1 + s - s ^ 2) * (1 - ρ)) := by
    have hdiff : v * (D - 2 * s * (1 + s - s ^ 2) * (1 - ρ))
        - (1 - s) ^ 2 * v * D
        = v * (s * (1 - s) ^ 2
            * (ρ * (s ^ 3 - 4 * s ^ 2 + 2 * s + 2) + s)) := by
      rw [← hfac]
      ring
    have hs1sq : 0 ≤ (1 - s) ^ 2 := sq_nonneg _
    have h3 : 0 ≤ s * (1 - s) ^ 2 := mul_nonneg hs0.le hs1sq
    have h4n : 0 ≤ s * (1 - s) ^ 2
        * (ρ * (s ^ 3 - 4 * s ^ 2 + 2 * s + 2) + s) :=
      mul_nonneg h3 hnum
    have hnonneg : 0 ≤ v * (s * (1 - s) ^ 2
        * (ρ * (s ^ 3 - 4 * s ^ 2 + 2 * s + 2) + s)) :=
      mul_nonneg hv h4n
    linarith
  have hrhsdiv : v * (1 - 2 * s * (1 + s - s ^ 2) * (1 - ρ) / D)
      = v * (D - 2 * s * (1 + s - s ^ 2) * (1 - ρ)) / D := by
    field_simp [ne_of_gt hDpos]
  rw [hrhsdiv]
  exact (le_div_iff₀ hDpos).mpr hmain
