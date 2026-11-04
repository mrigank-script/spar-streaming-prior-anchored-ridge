"""The online strength-and-window estimator shared by the SPAR heads.

Both ``run_spar_kf`` (the scalar SPAR-Pull head) and ``stream_spar_kf_eeg``
(the multi-class BCI head) update the same two online estimators from the same
conflict branches at each refresh block; this module holds that shared core.

The conflict branches themselves are assembled by the caller: prediction units
vs the legacy parameter units in ``run_spar_kf``, and the ``K``-class factor in
the EEG head.
"""

import numpy as np


def strength_window_update(*, C_ema, nu2_ema, V_hat, C_ref, C_resid, C_floor,
                           E_resid, mu_bar, rho, w, sigma2_base, D, cyc,
                           K_cal, eta_C, eta_nu, w_min, w_max, clip_lo, cap,
                           adaptive_rho=True, debias_variance_floor=True,
                           use_ref=True, use_cusum=True, rho_drift_floor=None,
                           rho_pin=None, w_pin=None):
    """One refresh-block update of the anchor strength ``rho`` and window ``w``.

    Returns ``(rho, w, C_ema, nu2_ema)``.  ``cap`` is the strength ceiling for
    this block (the caller lowers it during the ``K_cal`` burn-in).
    """
    cands = []
    if use_ref:
        cands.append(C_ref)
    if use_cusum:
        cands.append(C_resid)
    cands += [C_floor, 1e-10]
    C_raw = max(cands)
    C_ema = (1 - eta_C) * C_ema + eta_C * C_raw
    if adaptive_rho:
        rho_raw = mu_bar * V_hat / C_ema
        lo = clip_lo
        if rho_drift_floor is not None and w >= 0.5 * w_max:
            lo = max(lo, rho_drift_floor)
        rho = float(np.clip(rho_raw, lo, cap))
    if rho_pin is not None:
        rho = float(rho_pin)

    # estimator 2: forgetting window
    if debias_variance_floor:
        s_bar = rho / (mu_bar + rho)
        c_prime = (1.0 - s_bar) ** 2 * sigma2_base * D
        nu2_hat = max(E_resid * w * w
                      - c_prime * w ** 3 / max(2.0 - w, 1e-6), 0.0) \
            if use_cusum else 0.0
        nu2_ema = (1 - eta_nu) * nu2_ema + eta_nu * nu2_hat
        kappa = 4.0 * nu2_ema / max(c_prime, 1e-12)
    else:
        nu2_hat = E_resid * w * w / max((1.0 - w) ** 2, 1e-12) \
            if use_cusum else 0.0
        nu2_ema = (1 - eta_nu) * nu2_ema + eta_nu * nu2_hat
        s_bar = rho / (mu_bar + rho)
        kappa = 2.0 * nu2_ema / max(
            (1 - s_bar) ** 2 * sigma2_base * D, 1e-12)
    w_star = kappa ** (1.0 / 3.0)
    if cyc < K_cal:
        w = w_min
    else:
        w = float(np.clip(w_star, w_min, w_max))
    if w_pin is not None:
        w = float(w_pin)
    return rho, w, C_ema, nu2_ema
