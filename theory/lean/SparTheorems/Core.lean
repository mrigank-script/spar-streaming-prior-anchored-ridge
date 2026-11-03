import Mathlib

/-!
# SPAR (Spectrally-Adaptive Anchor Refresh): algebraic core

Machine-checked algebraic core of the SPAR theory chapter
(`paper/appendix.tex`). Every statement is pure algebra or
elementary real analysis over ℝ; no probability is mechanized.

## Table of contents

### Pull Identity
* `pull_identity`                : exact per-direction decomposition (γ = 1)
* `pull_identity_general`       : the same, with anchor shrinkage γ
* `eigenbasis_solve_coord`     : diagonal anchored solve decouples coordinatewise
* `eigenbasis_pull_identity`  : γ = 1 pull identity in the eigenbasis

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
* `sum_range_half`              : Σ_{j<k}(j + 1/2) = k2/2
* `moving_frame_reduction`      : g_{m+1} = a·g_m − (1−s)2·L − a·ν·k
* `cycle_midpoint_average`      : cycle average g_m − ν·k/2
* `cycle_lag_assembly`          : Δ_k = s·ν·k·(a/(1−a) + 1/2)

### Dial Fixed-Point Dichotomy
* `naive_dial_no_fixed_point`   : naive dial cubed value at least doubles per step
* `debiased_dial_contraction`   : debiased dial contracts geometrically to the optimum

### Correlated Finite-Cadence Variance
* `refresh_anchor_variance`     : steady-state anchor variance system
* `refresh_variance_form`       : closed-form correlated solution variance
* `refresh_variance_lt`         : strict variance reduction at every cadence
* `deff_sub_plugin`             : d_eff − D(1−s̄)2 = Σ(s_i−s̄)2 (exact variance decomposition)
* `deff_sub_plugin_nonneg`      : the mean-field plug-in D(1−s̄)2 ≤ d_eff (Jensen lower bound)

### Spectral Invariance and Conditioning
* `conj_trace`                 : orthogonal conjugation preserves the trace
* `conj_det`                   : orthogonal conjugation preserves the determinant
* `windup_ratio_le`            : conditioning ratio ≤ 1 + μmax/ρ
* `pull_weight_bounds`         : μ/(μ+ρ) ∈ [0,1), ρ/(μ+ρ) ∈ (0,1]

### Deterministic-Equivalent Fixed-Point Algebra
* `de_log_identity`           : e^(zeta q) = 1 + zeta theta at q = log(1+zeta theta)/zeta
* `de_collapse`               : rho/rho_eff = kappa = log(1+zeta theta)/(zeta theta)

### Smooth Effective Dimension
* `smooth_dim_lipschitz`      : μ/(μ+ρ) is 1/ρ-Lipschitz on μ ≥ 0

### Weighted Cross-Covariance Bound
* `weighted_cross_le`         : r|e_w e_r| ≤ V² for V = max(|e_w|, r|e_r|)

### Degenerate Active Dimension and Branch Selection
* `branch_max_lipschitz`      : the branch `max` is Lipschitz at ties
* `strength_floor_pos`        : the conflict floor is strictly positive
* `strength_floor_bound`      : the floor bounds the strength target

### Basis-Invariant Comparators
* `conj_const_diagonal`      : a constant profile is c*I in every basis
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

/-! ## Exact Effective Dimension and the Mean-Field Plug-In

The variance coefficient of the window law is
`c' = σ2 · d_eff(ρ)` with `d_eff = Σ_i (1 − s_i)^2`, while the runtime
rule uses the mean-field plug-in `D(1 − s̄)^2`, where
`s̄ = (Σ_i s_i)/D` is the mean pull.  The decomposition below shows that
the plug-in is the exact coefficient minus the population variance of the
pulls: it is a lower bound (Jensen), the discrepancy is exactly
`D · Var(s) ≥ 0`, it vanishes on an isotropic spectrum (all pulls equal),
and it is largest when the pulls are spread out, as on a rank-deficient
spectrum whose dead directions carry `s_i = 1`.  Consequently the exact
coefficient is available from the eigenvalues already computed at each
refresh, and substituting it removes the mean-field error from the
window iteration. -/

open Finset

/-- **Exact effective-dimension decomposition**: the difference between the
exact variance coefficient `d_eff = Σ_i (1 − s_i)^2` and the mean-field
plug-in `D(1 − s̄)^2` equals the population variance of the pulls,
`Σ_i (s_i − s̄)^2 = D · Var(s)`.  The identity is algebraic and holds for
every finite spectrum; no full-rank or isotropy assumption is used. -/
theorem deff_sub_plugin {ι : Type*} (T : Finset ι) (s : ι → ℝ) (hcard : T.card ≠ 0) :
    (∑ i ∈ T, (1 - s i) ^ 2)
      - (T.card : ℝ) * (1 - (∑ i ∈ T, s i) / (T.card : ℝ)) ^ 2
      = ∑ i ∈ T, (s i - (∑ i ∈ T, s i) / (T.card : ℝ)) ^ 2 := by
  have hn : (T.card : ℝ) ≠ 0 := by exact_mod_cast hcard
  set m : ℝ := (∑ i ∈ T, s i) / (T.card : ℝ) with hm
  have hmul : m * (T.card : ℝ) = ∑ i ∈ T, s i := by
    rw [hm]; field_simp
  have hL : (∑ i ∈ T, (1 - s i) ^ 2)
      = (T.card : ℝ) - 2 * (∑ i ∈ T, s i) + ∑ i ∈ T, s i ^ 2 := by
    have hpt : ∀ i, (1 - s i) ^ 2 = s i ^ 2 - 2 * s i + 1 := by intro i; ring
    simp_rw [hpt]
    rw [Finset.sum_add_distrib, Finset.sum_sub_distrib]
    rw [show (∑ i ∈ T, 2 * s i) = 2 * ∑ i ∈ T, s i by
      rw [show (∑ i ∈ T, 2 * s i) = ∑ i ∈ T, s i * 2 by
        apply Finset.sum_congr rfl; intro i _; ring]
      rw [← Finset.sum_mul]; ring]
    rw [show (∑ _i ∈ T, (1:ℝ)) = (T.card : ℝ) by simp]
    ring
  have hR : (∑ i ∈ T, (s i - m) ^ 2)
      = (∑ i ∈ T, s i ^ 2) - 2 * m * (∑ i ∈ T, s i) + (T.card : ℝ) * m ^ 2 := by
    have hpt : ∀ i, (s i - m) ^ 2 = s i ^ 2 - 2 * m * s i + m ^ 2 := by intro i; ring
    simp_rw [hpt]
    rw [Finset.sum_add_distrib, Finset.sum_sub_distrib]
    rw [show (∑ i ∈ T, 2 * m * s i) = 2 * m * ∑ i ∈ T, s i by
      rw [show (∑ i ∈ T, 2 * m * s i) = ∑ i ∈ T, s i * (2 * m) by
        apply Finset.sum_congr rfl; intro i _; ring]
      rw [← Finset.sum_mul]; ring]
    rw [show (∑ _i ∈ T, m ^ 2) = (T.card : ℝ) * m ^ 2 by
      rw [Finset.sum_const, nsmul_eq_mul]]
  rw [hL, hR, ← hmul]
  ring

/-- **Jensen lower bound**: the mean-field plug-in never exceeds the exact
variance coefficient, `D(1 − s̄)^2 ≤ d_eff(ρ)`, with equality iff all pulls
are equal.  Hence the runtime rule underestimates the variance coefficient
and returns a window no smaller than the exact-optimal one. -/
theorem deff_sub_plugin_nonneg {ι : Type*} (T : Finset ι) (s : ι → ℝ) (hcard : T.card ≠ 0) :
    0 ≤ (∑ i ∈ T, (1 - s i) ^ 2)
      - (T.card : ℝ) * (1 - (∑ i ∈ T, s i) / (T.card : ℝ)) ^ 2 := by
  rw [deff_sub_plugin T s hcard]
  exact Finset.sum_nonneg (fun i _ => sq_nonneg _)

/-! ## Moving-Frame Reduction and Cycle Assembly

The refresh chain of the finite-cadence equilibrium follows from a change
of frame.  Writing `g_m = G_m − w*(t_m)` for the anchor offset in the
moving frame, the within-cycle anchored solution is
`W_t − w*(t) = (1−s)(−L) + s(g_m − ν(t−t_m))`.  Substituting it into the
refresh `G_{m+1} = (1−s)W_{t_{m+1}} + s G_m` at `t_{m+1} = t_m + k` and
using the constant target velocity `w*(t_{m+1}) − w*(t_m) = ν k` gives the
affine recursion `g_{m+1} = a g_m − (1−s)2 L − a ν k` with `a = s(2−s)`.
Averaging the within-cycle advance over the cycle is the midpoint
assembly, exact for the linear advance, and its steady-state value is the
cycle-averaged lag `Δ_k(s) = s ν k (a/(1−a) + 1/2)`. -/

/-- **Gauss sum for the within-cycle midpoint**: the sum of the within-cycle
midpoint offsets is `Σ_{j<k}(j + 1/2) = k2/2`. -/
theorem sum_range_half (k : ℕ) :
    (∑ j ∈ Finset.range k, ((j : ℝ) + 1 / 2)) = (k : ℝ) ^ 2 / 2 := by
  induction k with
  | zero => simp
  | succ n ih =>
      rw [Finset.sum_range_succ, ih]
      push_cast
      ring

/-- **Moving-frame reduction**: the refresh update together with the
moving-frame within-cycle solution reduce to the affine recursion
`g_{m+1} = a g_m − (1−s)2·L − a·ν·k` with `a = s(2−s)`, where
`g_m = G_m − w*(t_m)` is the anchor offset in the moving frame and the
target advances by `ν·k` over the cycle. -/
theorem moving_frame_reduction
    (s L ν k gm Gm W1 wm wm1 Gm1 : ℝ)
    (hg : gm = Gm - wm)
    (hW1 : W1 - wm1 = (1 - s) * (-L) + s * (gm - ν * k))
    (hrefresh : Gm1 = (1 - s) * W1 + s * Gm)
    (htarget : wm1 - wm = ν * k) :
    Gm1 - wm1
      = s * (2 - s) * gm - (1 - s) ^ 2 * L - s * (2 - s) * ν * k := by
  have hW1' : W1 = wm1 + ((1 - s) * (-L) + s * (gm - ν * k)) := by
    linarith [hW1]
  have hG : Gm = gm + wm := by linarith [hg]
  have hwm1 : wm1 = wm + ν * k := by linarith [htarget]
  rw [hrefresh, hW1', hG, hwm1]
  ring

/-- **Midpoint cycle assembly**: the cycle average of the frozen-anchor
advance `g_m − ν·(t − t_m)` over the `k` within-cycle steps, taken at the
midpoints (exact for a linear advance), equals `g_m − ν·k/2`. -/
theorem cycle_midpoint_average (k : ℕ) (hk : k ≠ 0) (gm ν : ℝ) :
    (1 / (k : ℝ)) * (∑ j ∈ Finset.range k, (gm - ν * ((j : ℝ) + 1 / 2)))
      = gm - ν * (k : ℝ) / 2 := by
  have hk' : (k : ℝ) ≠ 0 := by exact_mod_cast hk
  rw [Finset.sum_sub_distrib]
  rw [show (∑ j ∈ Finset.range k, ν * ((j : ℝ) + 1 / 2))
        = ν * ∑ j ∈ Finset.range k, ((j : ℝ) + 1 / 2) by
      rw [Finset.mul_sum]]
  rw [sum_range_half, Finset.sum_const, Finset.card_range]
  simp only [nsmul_eq_mul]
  field_simp

/-- **Cycle-averaged lag**: substituting the stationary offset
`g* = −L − νk·a/(1−a)` into the midpoint assembly, the anchored gap
exceeds the pure-leaky lag `−L` by `Δ_k(s) = s·ν·k·(a/(1−a) + 1/2)`,
the cycle-averaged lag of the finite-cadence refresh equilibrium. -/
theorem cycle_lag_assembly (s ν L k : ℝ) (hne : 1 - s * (2 - s) ≠ 0) :
    - ((1 - s) * (-L)
        + s * ((-L - ν * k * (s * (2 - s)) / (1 - s * (2 - s)))
               - ν * k / 2) - (-L))
      = s * ν * k * ((s * (2 - s)) / (1 - s * (2 - s)) + 1 / 2) := by
  field_simp
  ring

open Matrix

/-! ## Spectral Invariance and Conditioning

The scalar method uses only the sufficient statistics `A, B, G`; rotating the
stream by an orthogonal `Q` conjugates the accumulator, and its scalar
statistics are invariant to that conjugation.  The pull weights stay in the
unit interval for every excitation, and the anchored conditioning ratio is
bounded by `1 + μmax/ρ`, independent of the drift. -/

/-- **Trace invariance**: orthogonal conjugation preserves the trace. -/
theorem conj_trace {n : Type*} [Fintype n] [DecidableEq n]
    (Q A : Matrix n n ℝ) (hQ : Q * Qᵀ = 1) :
    (Qᵀ * A * Q).trace = A.trace := by
  rw [Matrix.trace_mul_cycle, hQ, one_mul]

/-- **Determinant invariance**: orthogonal conjugation preserves the determinant. -/
theorem conj_det {n : Type*} [Fintype n] [DecidableEq n]
    (Q A : Matrix n n ℝ) (hQ : Q * Qᵀ = 1) :
    (Qᵀ * A * Q).det = A.det := by
  rw [Matrix.det_mul, Matrix.det_mul, Matrix.det_transpose]
  have h : Q.det * Q.det = 1 := by
    have := congrArg Matrix.det hQ
    rwa [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at this
  linear_combination A.det * h

/-- **Conditioning bound**: the eigenvalue-interval ratio is at most `1 + μmax/ρ`. -/
theorem windup_ratio_le (μ μmax ρ : ℝ) (hρ : 0 < ρ) (h : μ ≤ μmax) :
    (μ + ρ) / ρ ≤ 1 + μmax / ρ := by
  have hsub : (μ + ρ) / ρ = 1 + μ / ρ := by field_simp; ring
  rw [hsub]
  exact add_le_add (le_refl (1 : ℝ)) ((div_le_div_iff_of_pos_right hρ).mpr h)

/-- **Pull weights in the unit interval**: `μ/(μ+ρ) ∈ [0,1)` and `ρ/(μ+ρ) ∈ (0,1]`. -/
theorem pull_weight_bounds (μ ρ : ℝ) (hμ : 0 ≤ μ) (hρ : 0 < ρ) :
    0 ≤ μ / (μ + ρ) ∧ μ / (μ + ρ) < 1 ∧ 0 < ρ / (μ + ρ) ∧ ρ / (μ + ρ) ≤ 1 := by
  have hd : 0 < μ + ρ := by linarith
  refine ⟨by positivity, ?_, by positivity, ?_⟩
  · rw [div_lt_one hd]; linarith
  · rw [div_le_one hd]; linarith

/-! ## Eigenbasis Specialization of the Pull Identity

In the eigenbasis of the accumulator the anchored solve is a diagonal system,
so it decouples coordinatewise into scalar anchored solves.  Each coordinate
is the `γ = 1` pull interpolation of the data solve and the anchor with weight
`s_i = ρ/(μ_i + ρ)`. -/

/-- **Eigenbasis decoupling**: in a diagonal basis the anchored solve's
`i`-th coordinate `(b_i + ρ g_i)/(μ_i + ρ)` satisfies the scalar anchored
equation `(μ_i + ρ) w_i = b_i + ρ g_i`. -/
theorem eigenbasis_solve_coord {n : Type*} [Fintype n] [DecidableEq n]
    (μ b g : n → ℝ) (ρ : ℝ) (hμ : ∀ i, μ i + ρ ≠ 0) (i : n) :
    (Matrix.diagonal (fun j => μ j + ρ)
        *ᵥ (fun j => (b j + ρ * g j) / (μ j + ρ))) i
      = b i + ρ * g i := by
  rw [Matrix.mulVec_diagonal]
  field_simp [hμ i]

/-- **Eigenbasis pull identity (`γ = 1`)**: the coordinatewise anchored solve
interpolates the data solve and the anchor with weight `s_i = ρ/(μ_i + ρ)`. -/
theorem eigenbasis_pull_identity {n : Type*} (μ b g wstar : n → ℝ) (ρ : ℝ)
    (hμ : ∀ i, μ i ≠ 0) (hμρ : ∀ i, μ i + ρ ≠ 0) (i : n) :
    (b i + ρ * g i) / (μ i + ρ) - wstar i
      = (μ i / (μ i + ρ)) * (b i / μ i - wstar i)
        + (ρ / (μ i + ρ)) * (g i - wstar i) := by
  have h := pull_identity (μ i) ρ (b i) (g i) (wstar i) (hμ i) (hμρ i)
  simpa using h

/-! ## Deterministic-Equivalent Fixed-Point Algebra

The separable-covariance deterministic equivalent collapses to a scalar fixed
point.  With weight sum `κ = log(1+ζθ)/(ζθ)` and `q = κθ`, the retention
identity `κθ + s̄ = 1` fixes `q`, and the anchored strength renormalizes as
`ρ/ρ_eff = κ`, equivalently `ρ_eff = ρ(e^{ζq}-1)/(ζq)`. -/

/-- **DE log identity**: with the substitution `q = log(1+ζθ)/ζ`, the argument
of the renormalization returns the weight sum, `e^{ζ q} = 1 + ζθ`. -/
theorem de_log_identity (ζ θ : ℝ) (h1 : 0 < 1 + ζ * θ) (hζ : ζ ≠ 0) :
    Real.exp (ζ * (Real.log (1 + ζ * θ) / ζ)) = 1 + ζ * θ := by
  have hcancel : ζ * (Real.log (1 + ζ * θ) / ζ) = Real.log (1 + ζ * θ) := by
    field_simp
  rw [hcancel, Real.exp_log h1]

/-- **DE collapse / renormalization**: with `κ = log(1+ζθ)/(ζθ)` and `q = κθ`,
the anchored strength renormalizes as `ρ/ρ_eff = κ`, equivalently
`ρ_eff = ρ (e^{ζ q} - 1)/(ζ q)`. -/
theorem de_collapse (ρ ζ θ : ℝ) (h1 : 0 < 1 + ζ * θ) (hζθ : ζ * θ ≠ 0) :
    ρ / (Real.log (1 + ζ * θ) / (ζ * θ))
      = ρ * (Real.exp (ζ * ((Real.log (1 + ζ * θ) / (ζ * θ)) * θ)) - 1)
          / (ζ * ((Real.log (1 + ζ * θ) / (ζ * θ)) * θ)) := by
  have hζ : ζ ≠ 0 := fun h => hζθ (by rw [h, zero_mul])
  have hθ : θ ≠ 0 := fun h => hζθ (by rw [h, mul_zero])
  have hlog : Real.log (1 + ζ * θ) ≠ 0 := by
    intro h
    have h1' : (1 : ℝ) + ζ * θ = 1 := by
      have := Real.exp_log h1
      rw [h, Real.exp_zero] at this
      exact this.symm
    exact hζθ (by linarith)
  have hkey : ζ * ((Real.log (1 + ζ * θ) / (ζ * θ)) * θ)
      = Real.log (1 + ζ * θ) := by
    field_simp [hθ, hζ]
  have hexp : Real.exp (ζ * ((Real.log (1 + ζ * θ) / (ζ * θ)) * θ))
      = 1 + ζ * θ := by
    rw [hkey, Real.exp_log h1]
  rw [hexp, hkey]
  field_simp
  ring

/-! ## Smooth Effective Dimension

The effective dimension in the strength estimator is the resolvent trace
`Σ_i μ_i/(μ_i + ρ_ridge)`, whose summands lie in `[0,1)` and are
`1/ρ_ridge`-Lipschitz in the excitation.  It is smooth in the eigenvalues
with no spectral-gap assumption, unlike an integer active-dimension count. -/

/-- **Smooth effective-dimension Lipschitz**: each resolvent term `μ/(μ+ρ)`
is `1/ρ`-Lipschitz on the excitation half-line `μ ≥ 0`.  The range bound
`μ/(μ+ρ) ∈ [0,1)` is `pull_weight_bounds`. -/
theorem smooth_dim_lipschitz (μ μ' ρ : ℝ) (hμ : 0 ≤ μ) (hμ' : 0 ≤ μ')
    (hρ : 0 < ρ) :
    |μ / (μ + ρ) - μ' / (μ' + ρ)| ≤ |μ - μ'| / ρ := by
  have h1 : μ + ρ ≠ 0 := by positivity
  have h2 : μ' + ρ ≠ 0 := by positivity
  have hρ2 : 0 < (μ + ρ) * (μ' + ρ) := by positivity
  have hkey : μ / (μ + ρ) - μ' / (μ' + ρ)
      = ρ * (μ - μ') / ((μ + ρ) * (μ' + ρ)) := by
    field_simp
    ring
  rw [hkey, abs_div, abs_mul, abs_of_pos hρ, abs_of_pos hρ2]
  field_simp
  nlinarith [abs_nonneg (μ - μ'), mul_nonneg hμ hμ', mul_nonneg hμ hρ.le,
    mul_nonneg hμ' hρ.le]

/-! ## Weighted Cross-Covariance Bound

The coupled contraction of Theorem `thm:closedloop` uses the weighted
max-norm `V = max(|e_ω|, r|e_ρ|)`.  That same norm bounds the cross moment:
`|e_ω| ≤ V` and `r|e_ρ| ≤ V` give `r|e_ω e_ρ| ≤ V²`, hence
`E[e_ω e_ρ] ≤ E[V²]/r`, so the cross-quantity covariance is controlled by
the scalar Lyapunov bound of part (iii), not left unasserted. -/

/-- **Cross-covariance bound from the weighted max-norm**: if the window and
strength errors satisfy `|e_ω| ≤ V` and `r|e_ρ| ≤ V`, then `r|e_ω e_ρ| ≤ V²`,
so the cross moment is at most `V²/r`; with `E[V²] ≤ q_*/(1-γ)²` the
stationary cross-covariance obeys `|E[e_ω e_ρ]| ≤ q_*/(r(1-γ)²)`. -/
theorem weighted_cross_le (a b r V : ℝ) (hr : 0 < r) (ha : |a| ≤ V)
    (hb : r * |b| ≤ V) :
    r * |a * b| ≤ V * V := by
  have hV : 0 ≤ V := le_trans (abs_nonneg a) ha
  rw [abs_mul]
  calc r * (|a| * |b|) = |a| * (r * |b|) := by ring
    _ ≤ V * V :=
      mul_le_mul ha hb (mul_nonneg hr.le (abs_nonneg b)) hV

/-! ## Degenerate Active Dimension and Branch Selection

Two degenerate concerns of the strength estimator are resolved by
construction.  The conflict floor `Ĉ = max(C_ref, C_resid, C_floor)` keeps
`Ĉ ≥ C_floor > 0`, removing the `Ĉ → 0` singularity of `ρ* = μ̄ V̂/Ĉ`; and the
`max` selection of the active branch is Lipschitz even where the active branch
switches, so the derivative jump at a tie cannot break the contraction, which
uses only Lipschitzness.  The active dimension is floored at one in the
runtime, and `μ̄ = tr(Ā)/d_act > 0` whenever `Ā ≠ 0`, so `V̂ > 0`; the
no-excitation limit `Ā = 0` gives `μ̄ = 0` and `ρ* = 0`, the unanchored
(pure-leaky) degenerate case. -/

/-- **Branch selection is Lipschitz**: the `max` of the conflict branches is
Lipschitz even where the active branch switches. -/
theorem branch_max_lipschitz (a a' b b' c c' : ℝ) :
    |max (max a b) c - max (max a' b') c'|
      ≤ max (max |a - a'| |b - b'|) |c - c'| := by
  calc |max (max a b) c - max (max a' b') c'|
      ≤ max |max a b - max a' b'| |c - c'| :=
        abs_max_sub_max_le_max _ _ _ _
    _ ≤ max (max |a - a'| |b - b'|) |c - c'| :=
        max_le_max (abs_max_sub_max_le_max _ _ _ _) (le_refl _)

/-- **Floor removes the strength singularity**: the conflict floor keeps the
selected estimate strictly positive. -/
theorem strength_floor_pos (a b floor : ℝ) (h : 0 < floor) :
    0 < max (max a b) floor :=
  lt_of_lt_of_le h (le_max_right _ _)

/-- **Floor keeps the strength target bounded**: with a strictly positive
floor, `μ V / max(max a b) floor ≤ μ V / floor` whenever `μ V ≥ 0`. -/
theorem strength_floor_bound (mu V a b floor : ℝ) (hV : 0 ≤ mu * V)
    (hfloor : 0 < floor) :
    mu * V / max (max a b) floor ≤ mu * V / floor := by
  have hle : floor ≤ max (max a b) floor := le_max_right _ _
  exact div_le_div_of_nonneg_left hV hfloor hle

/-! ## Basis-Invariant No-Regret Comparators

The projected-gradient comparator of Theorem `prop:ocowrap(i)` is a pull
profile `s ∈ [0,1]^D` read in the instantaneous eigenbasis `U_t`, i.e. the
realized physical operator `U_t diag(s) U_tᵀ`.  A *constant* profile `s ≡ c`
is basis-invariant: `U diag(c) Uᵀ = c·I` in every orthonormal basis, so the
isotropic profiles, in particular the data path `c = 0` and the anchor path
`c = 1`, are basis-invariant comparators of the no-regret bound; a general
fixed `s` is a comparator through its realized sequence. -/

/-- **Basis-invariant comparators**: a constant pull profile `s ≡ c` realizes
the physical operator `Q · diag(c) · Qᵀ = c·I` in every orthonormal basis `Q`. -/
theorem conj_const_diagonal {n : Type*} [Fintype n] [DecidableEq n]
    (Q : Matrix n n ℝ) (c : ℝ) (hQ : Q * Qᵀ = 1) :
    Q * Matrix.diagonal (fun _ : n => c) * Qᵀ = c • (1 : Matrix n n ℝ) := by
  have hdiag : Matrix.diagonal (fun _ : n => c) = c • (1 : Matrix n n ℝ) := by
    rw [show (fun _ : n => c) = c • (fun _ => (1 : ℝ)) by ext; simp,
        Matrix.diagonal_smul, Matrix.diagonal_one]
  rw [hdiag, Matrix.mul_smul, Matrix.smul_mul, mul_one, hQ]