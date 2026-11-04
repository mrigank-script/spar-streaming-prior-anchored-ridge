"""Named numeric constants for the SPAR streaming heads.

The estimators depend on a handful of fixed constants; naming them keeps the
runtime rule and the manuscript describing the same numbers.  Values are
unchanged from the inline literals they replace.
"""

RIDGE_REF = 1e-6          # numerical ridge of the reference-branch data solve
RIDGE_TRACE = 1e-8        # ridge for the trace / inverse-trace computations
ACT_THRESH = 1e-8         # relative eigenvalue threshold for the active dimension
SLOW_ETA = 5e-5           # slow residual EMA gain (horizon ~1/SLOW_ETA)
GREF_HORIZON = 2000.0     # G_ref EMA horizon, in stream steps

# EEG head: the short streams need a shorter slow horizon and a faster baseline.
EEG_GREF_HORIZON = 150.0
EEG_SLOW_ETA = 5e-4
