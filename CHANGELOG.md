# Changelog

## 0.2.0

Repository and manuscript upgrade cycle.

### Theory
- Exact effective-dimension identity `d_eff - D(1-s_bar)^2 = sum (s_i - s_bar)^2`, formalized
  (`deff_sub_plugin`, `deff_sub_plugin_nonneg`).
- Active-subspace deterministic equivalent for rank-deficient spectra (corollary), with the
  diagnostic spectrum aligned to the stream generator.
- False-alarm calibration for the sliding two-scale EMA (weighted Freedman bound, time-uniform).
- Moving-frame reduction and cycle-average assembly formalized (`moving_frame_reduction`,
  `cycle_midpoint_average`, `cycle_lag_assembly`, `sum_range_half`).
- Exact finite-window variance term in the cycle-averaged risk; the small-aspect-ratio
  restriction is retired and replaced by the random-design variance.
- Global closed-loop contraction on the parameter box, with the clip as a nonexpansive projection.
- Explicit finite-`n` correction for the reference branch of the strength estimator.
- Closed-form characterization of the coupled stationary law as the fixed point of the target-map
  composition, with a convergent fixed-point series.
- Window surrogate value-gap bound (proposition) with a certified numerical bound.
- Smooth effective dimension replaces the spectral-gap assumption in the closed-loop Lipschitzness.
- Weighted cross-covariance bound for the coupled fluctuation.
- Degenerate active-dimension and branch-selection lemmas; matrix pull-weight, resolvent, and floor identities over Mathlib matrices.
- Basis-invariant no-regret comparators (constant profiles realize `c*I`).
- Deterministic-equivalent collapse algebra formalized (`de_collapse`).
- Moving-target gap characterized by spectrum and margin.
- Concentration bound extended to mixing and heavy-tailed streams; mismatch suite scoped.

### Experiments and diagnostics
- Exact random-design variance diagnostic, finite-`D` concentration sweep.
- Moving-target decomposition, window surrogate-gap certificate.
- Sustained-drift sticky-floor detector with a constant-velocity stream.
- Causal end-to-end EEG Hedge mixture and causal-alignment window sweep.
- Electricity random-client population and multi-step horizon sweep.

### Manuscript
- Lean coverage map and declaration count updated (85 declarations).
- Related-work comparison matrix and quantitative "what is new" positioning.
- Reproduction instructions updated; derived EEG cache shipped with the artifact.

### Tooling
- `VERSION` and this changelog added and included in the release artifact.