#!/bin/bash
# Print the axiom dependencies of the formalized SparTheorems results.
# Runs inside the built lake project (default $HOME/SparTheorems; override
# with SPAR_PROJECT=...). The probe file is temporary and removed afterward.
set -eu
export PATH="$HOME/.elan/bin:$PATH"

PROJECT="${SPAR_PROJECT:-$HOME/SparTheorems}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

cat > "$TMP_DIR/print_axioms.lean" <<'EOF'
import SparTheorems
#print axioms ogd_regret_box_D
#print axioms hedge_regret
#print axioms window_optimum
#print axioms sum_expand_sq
#print axioms ogd_regret_segment
#print axioms safety_threshold
#print axioms pull_inverse
#print axioms refresh_closed_form
#print axioms refresh_fixed_point
#print axioms refresh_contraction_rate
#print axioms refresh_variance_lb
#print axioms deff_sub_plugin
#print axioms deff_sub_plugin_nonneg
#print axioms moving_frame_reduction
#print axioms cycle_midpoint_average
#print axioms cycle_lag_assembly
#print axioms sum_range_half
#print axioms conj_trace
#print axioms conj_det
#print axioms windup_ratio_le
#print axioms pull_weight_bounds
#print axioms eigenbasis_solve_coord
#print axioms eigenbasis_pull_identity
#print axioms de_log_identity
#print axioms de_collapse
#print axioms smooth_dim_lipschitz
#print axioms weighted_cross_le
#print axioms branch_max_lipschitz
#print axioms strength_floor_pos
#print axioms strength_floor_bound
#print axioms joint_tracking
#print axioms joint_geometric_pinned
#print axioms window_second_deriv_pos
EOF

cd "$PROJECT"
lake env lean "$TMP_DIR/print_axioms.lean"
