import SparTheorems.Ogd
import SparTheorems.Hedge

/-!
# D-dimensional box extension

The key geometric fact: the coordinatewise box projection on (Fin D → ℝ)
is a contraction in squared ℓ₂ distance toward any point of the box,
because it is the sum of per-coordinate 1-dim contractions
(`projBox_dist_sq_le`). The OGD analysis then carries through with the
norm as a sum of squares, by the same argument as the 1-dim case.

## Table of contents

### Geometric Primitives
* `projBoxD`            : coordinatewise box projection.
* `normSq`              : squared ℓ₂ norm as a sum of squares.
* `dot`                 : Euclidean inner product.
* `inBox`               : box membership predicate.

### Box Projection
* `projBoxD_dist_sq_le` : the D-dim contraction (coordinatewise sum).

### Norm Expansion
* `sum_expand_sq`       : the norm expansion (the algebra step).

### OGD Step
* `ogd_step_D`          : the D-dimensional OGD step lemma.

### Regret Theorem
* `ogd_regret_box_D`    : Zinkevich on the D-dimensional box.
-/

/-! ## Geometric Primitives
This section contains the D-dimensional geometric definitions: the
coordinatewise box projection, the squared ℓ₂ norm, the Euclidean inner
product, and the box membership predicate.
-/

def projBoxD {D : ℕ} (x : Fin D → ℝ) : Fin D → ℝ :=
  fun i => projBox (x i)

def normSq {D : ℕ} (f : Fin D → ℝ) : ℝ :=
  ∑ i, (f i) ^ 2

def dot {D : ℕ} (f g : Fin D → ℝ) : ℝ :=
  ∑ i, f i * g i

def inBox {D : ℕ} (s : Fin D → ℝ) : Prop :=
  ∀ i, s i ∈ Set.Icc (0:ℝ) 1


/-! ## Box Projection
This section contains the projection lemmas: the coordinatewise
projection is a contraction in squared ℓ₂ distance toward any point of
the box, being the sum of per-coordinate 1-dim contractions.
-/

theorem projBoxD_le {D : ℕ} (x : Fin D → ℝ) (i : Fin D) :
    projBoxD x i ≤ 1 := projBox_le (x i)

theorem projBoxD_nonneg {D : ℕ} (x : Fin D → ℝ) (i : Fin D) :
    0 ≤ projBoxD x i := projBox_nonneg (x i)

/-- **D-dimensional box contraction**: the coordinatewise
projection is a contraction in squared ℓ₂ distance toward any point
of the box. This is the sum of per-coordinate 1-dim contractions. -/
theorem projBoxD_dist_sq_le {D : ℕ} (s x : Fin D → ℝ)
    (h1 : ∀ i, 0 ≤ s i) (h2 : ∀ i, s i ≤ 1) :
    normSq (projBoxD x - s) ≤ normSq (x - s) := by
  unfold normSq projBoxD
  simp only [Pi.sub_apply]
  exact Finset.sum_le_sum (fun i _ =>
    projBox_dist_sq_le (s i) (x i) (h1 i) (h2 i))


/-! ## Norm Expansion
This section contains the norm expansion identity — the algebra step that
expands the squared norm of the OGD update direction.
-/

/-- **Norm expansion** (the algebra step): expanding the squared norm of
(a - η·b - c). The per-coordinate identity is `ring`; the sum-level step
distributes the finite sum and factors the constants. -/
theorem sum_expand_sq {D : ℕ} (a b c : Fin D → ℝ) (η : ℝ) :
    normSq (a - (fun i => η * b i) - c)
      = normSq (a - c) - 2 * η * dot b (a - c) + η ^ 2 * normSq b := by
  unfold normSq dot
  simp only [Pi.sub_apply]
  have hper : ∀ i : Fin D,
      (a i - η * b i - c i) ^ 2
        = (a i - c i) ^ 2 + (-(2 * η * (b i * (a i - c i))))
            + η ^ 2 * (b i) ^ 2 := by
    intro i; ring
  rw [Finset.sum_congr rfl (fun i _ => hper i)]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib,
      Finset.sum_neg_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
  ring


/-! ## OGD Step
This section contains the D-dimensional OGD step lemma: with coordinatewise
clipping, the instantaneous regret is bounded by the potential decrease
plus `η ||g_t||² / 2`.
-/

/-- **D-dimensional OGD step lemma**: with coordinatewise clipping,
the instantaneous regret is bounded by the potential decrease plus
`η ||g_t||² / 2`. -/
theorem ogd_step_D {D : ℕ}
    (l : ℕ → (Fin D → ℝ) → ℝ) (g : ℕ → (Fin D → ℝ))
    (w : ℕ → (Fin D → ℝ)) (η : ℝ) (ss : Fin D → ℝ)
    (hη : 0 < η) (t : ℕ)
    (hconv : ∀ v, inBox v →
      l t (w t) + dot (g t) (v - w t) ≤ l t v)
    (hrec : w (t + 1) = projBoxD (w t - fun i => η * g t i))
    (hs : inBox ss) :
    l t (w t) - l t ss
      ≤ (normSq (w t - ss) - normSq (w (t + 1) - ss)) / (2 * η)
        + η * normSq (g t) / 2 := by
  have hgrad : l t (w t) - l t ss ≤ dot (g t) (w t - ss) := by
    have h := hconv ss hs
    have hneg : dot (g t) (ss - w t) + dot (g t) (w t - ss) = 0 := by
      unfold dot
      simp only [Pi.sub_apply]
      have hper : ∀ i : Fin D,
          g t i * (ss i - w t i) + g t i * (w t i - ss i) = 0 := by
        intro i; ring
      have hsum : ∑ i : Fin D,
          (g t i * (ss i - w t i) + g t i * (w t i - ss i)) = 0 := by
        rw [Finset.sum_congr rfl (fun i _ => hper i)]
        simp
      rw [← Finset.sum_add_distrib]
      exact hsum
    linarith
  have hclip : normSq (w (t + 1) - ss)
      ≤ normSq (w t - (fun i => η * g t i) - ss) := by
    rw [hrec]
    exact projBoxD_dist_sq_le ss _ (fun i => (hs i).1) (fun i => (hs i).2)
  have hnorm := sum_expand_sq (w t) (g t) ss η
  rw [hnorm] at hclip
  have h2η : (0:ℝ) < 2 * η := by linarith
  have hkey : dot (g t) (w t - ss)
      ≤ (normSq (w t - ss) - normSq (w (t + 1) - ss)
          + η ^ 2 * normSq (g t)) / (2 * η) := by
    rw [le_div_iff₀ h2η]
    nlinarith [hclip]
  have hsplit : (normSq (w t - ss) - normSq (w (t + 1) - ss)
      + η ^ 2 * normSq (g t)) / (2 * η)
      = (normSq (w t - ss) - normSq (w (t + 1) - ss)) / (2 * η)
        + η * normSq (g t) / 2 := by
    rw [add_div]
    have hq : η ^ 2 * normSq (g t) / (2 * η) = η * normSq (g t) / 2 := by
      field_simp
    rw [hq]
  rw [hsplit] at hkey
  linarith [hgrad, hkey]


/-! ## Regret Theorem
This section contains Zinkevich on the D-dimensional box: the telescoped
regret bound with the norm replacing the squared modulus.
-/

/-- **Zinkevich on the D-dimensional box** (no probability anywhere).
For arbitrary losses satisfying the subgradient inequality with
gradient norm bound, OGD with coordinatewise clipping has regret
≤ normSq(w_0 - s*)/(2η) + η·T·G²/2 against any fixed s* in the box.
The proof mirrors the 1-dim `ogd_regret_box` with `normSq` replacing
the squared norm and `dot` replacing the product. -/
theorem ogd_regret_box_D {D : ℕ}
    (l : ℕ → (Fin D → ℝ) → ℝ) (g : ℕ → (Fin D → ℝ))
    (w : ℕ → (Fin D → ℝ)) (η G : ℝ) (ss : Fin D → ℝ)
    (hη : 0 < η) (T : ℕ)
    (hconv : ∀ t v, inBox v →
      l t (w t) + dot (g t) (v - w t) ≤ l t v)
    (hbnd : ∀ t, normSq (g t) ≤ G ^ 2)
    (hrec : ∀ t, w (t + 1) = projBoxD (w t - fun i => η * g t i))
    (hs : inBox ss) :
    (∑ t ∈ Finset.range T, l t (w t)) - (∑ t ∈ Finset.range T, l t ss)
      ≤ normSq (w 0 - ss) / (2 * η) + η * T * G ^ 2 / 2 := by
  have main : ∀ n : ℕ,
      ((∑ t ∈ Finset.range n, l t (w t))
        - (∑ t ∈ Finset.range n, l t ss)
        + normSq (w n - ss) / (2 * η))
      ≤ normSq (w 0 - ss) / (2 * η)
        + ∑ t ∈ Finset.range n, η * normSq (g t) / 2 := by
    intro n
    induction n with
    | zero => simp
    | succ n ih =>
      have step := ogd_step_D l g w η ss hη n
        (fun v hv => hconv n v hv) (hrec n) hs
      rw [sub_div] at step
      have hs1 := Finset.sum_range_succ (fun t => l t (w t)) n
      have hs2 := Finset.sum_range_succ (fun t => l t ss) n
      have hs3 := Finset.sum_range_succ (fun t => η * normSq (g t) / 2) n
      rw [hs1, hs2, hs3]
      have rearr : ((∑ t ∈ Finset.range n, l t (w t) + l n (w n))
            - (∑ t ∈ Finset.range n, l t ss + l n ss))
           = ((∑ t ∈ Finset.range n, l t (w t))
             - (∑ t ∈ Finset.range n, l t ss))
             + (l n (w n) - l n ss) := by ring
      have combined : (((∑ t ∈ Finset.range n, l t (w t))
             - (∑ t ∈ Finset.range n, l t ss)) + (l n (w n) - l n ss)
             + normSq (w (n + 1) - ss) / (2 * η))
           ≤ ((normSq (w 0 - ss) / (2 * η)
               + ∑ t ∈ Finset.range n, η * normSq (g t) / 2)
             + η * normSq (g n) / 2) := by
        have hadd := add_le_add ih step
        linarith [hadd, rearr]
      linarith [combined, rearr]
  have hmain := main T
  have h2ηnn : (0:ℝ) ≤ 2 * η := by linarith
  have hpot : 0 ≤ normSq (w T - ss) / (2 * η) := by
    unfold normSq
    exact div_nonneg (Finset.sum_nonneg (fun i _ => sq_nonneg _)) h2ηnn
  have hheat : ∑ t ∈ Finset.range T, η * normSq (g t) / 2
      ≤ η * T * G ^ 2 / 2 := by
    have hle : ∀ t ∈ Finset.range T, η * normSq (g t) / 2 ≤ η * G ^ 2 / 2 := by
      intro t _
      have hg' := hbnd t
      have hmul : η * normSq (g t) ≤ η * G ^ 2 :=
        mul_le_mul_of_nonneg_left hg' (le_of_lt hη)
      have hkey : η * normSq (g t) / 2 - η * G ^ 2 / 2
          = (η * normSq (g t) - η * G ^ 2) / 2 := by
        field_simp
      have h2pos : (0:ℝ) < 2 := by norm_num
      have hdiv : (η * normSq (g t) - η * G ^ 2) / 2 ≤ 0 := by
        apply div_nonpos_of_nonpos_of_nonneg
        · linarith
        · linarith
      linarith [hkey, hmul, hdiv]
    have hsumle : ∑ t ∈ Finset.range T, η * normSq (g t) / 2
        ≤ ∑ _t ∈ Finset.range T, η * G ^ 2 / 2 :=
      Finset.sum_le_sum hle
    have hcard : ∑ _t ∈ Finset.range T, η * G ^ 2 / 2
        = η * T * G ^ 2 / 2 := by
      rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
      push_cast
      ring
    linarith [hsumle, hcard]
  linarith [hmain, hpot, hheat]
