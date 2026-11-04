"""SPAR streaming heads, with the baseline library re-exported.

``rls_baselines`` keeps the SPAR-Pull family -- the anchored, self-calibrating
streaming heads -- and re-exports the stream generator, the priors, and the
competitor baselines, so ``from rls_baselines import <name>`` keeps working:

  run_spar_diag  DIAGONAL-rho SPAR: per-direction strength (Theorem 5)
  run_spar_kf    SPAR-Pull: scalar strength + forgetting window
  run_spar_eg    SPAR-EG: expert-simplex composition

Streams/priors live in :mod:`spar_streams`; baselines in :mod:`spar_baselines`.

Sanity checks (``python scripts/rls_baselines.py``):
  RLS(lam=1) matches the batch count-scaled ridge at every step;
  DF(lam_f=1) reduces to RLS(lam=1);
  KF(q=0) reduces to RLS(lam=1) up to gain scaling;
  WiSE-FT endpoints reproduce the warm and e2e models.
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Re-exported so existing imports keep working (flake8: F401).
from spar_streams import make_stream_fac, make_prior  # noqa: E402,F401
from spar_baselines import (run_rls, run_vff, run_df, run_arow, run_kf,  # noqa: E402,F401
                            run_sgd, run_adam, wiseft_models, wiseft_preds,
                            run_kf_est, run_leaky_batch)

from spar_constants import (RIDGE_REF, RIDGE_TRACE, ACT_THRESH,  # noqa: E402
                           SLOW_ETA, GREF_HORIZON)
from spar_estimators import strength_window_update  # noqa: E402


def run_spar_diag(x, y, T_warm, G_init=None, alpha=0.997, refresh_every=50,
                  eta_delta=0.2, s_cap=0.995, resid_fast=100,
                  slow_horizon=2000, ridge=0.01, shrink=0.5,
                  return_state=False):
    """Diagonal-rho SPAR: per-direction strength from Theorem 5.

      v_i    = sigma^2 / (n_eff * mu_i)                  (data variance)
      delta_i^2 = max( (G_i - w_hat_i)^2 - v_i,  floor_i )
      rho_i* = mu_i * v_i / delta_i^2   ->   s_i = rho_i/(mu_i + rho_i)
      refresh rate beta_i = 1 - s_i    (the pull-identity coupling)

    Per-direction conflict estimates are high-variance in weakly excited
    directions (few effective samples), so each delta_i^2 is shrunk toward
    the cross-direction mean by w_i = mu_i/(mu_i + mu_bar) (James-Stein
    style; `shrink` scales the shrinkage, 0 = none, 1 = full-to-mean).
    Dead directions (mu ~ 0): s_i = s_cap (full anchor, beta ~ 0).
    A smooth residual-CUSUM gate disengages all directions under drift
    onset (gate = sigma2_slow / (sigma2_slow + max(0, fast - slow))).

    THE GATED PULL IS USED FOR BOTH THE SOLVE AND THE REFRESH — the
    pull-identity coupling must be self-consistent or the anchor locks in
    post-drift (found in smoke test).

    Cost: O(D^2) per step, O(D^3) only at refreshes.
    """
    D = x.shape[1]
    A = x[:T_warm].T @ x[:T_warm].astype(np.float64)
    B = x[:T_warm].T @ y[:T_warm]
    n_eff = float(T_warm)
    W_warm_fit = np.linalg.solve(A + n_eff * ridge * np.eye(D), B)
    G = np.asarray(G_init, dtype=np.float64).copy() if G_init is not None \
        else W_warm_fit.copy()
    r_warm = y[:T_warm] - x[:T_warm] @ W_warm_fit
    sigma2 = float(np.var(r_warm)) + 1e-12      # slow noise EMA (horizon 2000)
    sig_fast = sigma2                           # fast noise EMA (horizon 100)
    eta_f = 1.0 / resid_fast
    eta_s = 1.0 / slow_horizon

    # cached eigen state (recomputed at refresh cadence)
    mu, U = np.linalg.eigh(A / n_eff)
    mu_pos = np.maximum(mu, 1e-12)
    G_e = U.T @ G                                # anchor in eigen coords
    mu_bar = float(np.mean(mu_pos))
    floor = 0.25 * sigma2 / (T_warm * mu_pos)
    delta2 = floor.copy()                        # per-direction conflict EMA
    v_dir = sigma2 / (n_eff * mu_pos)
    rho_dir = mu_pos * v_dir / np.maximum(delta2, 1e-14)
    s_vec = np.clip(rho_dir / (mu_pos + rho_dir), 0.0, s_cap)
    s_vec[mu < 1e-9 * max(mu[-1], 1e-12)] = s_cap

    preds = np.zeros(len(y) - T_warm)
    step = 0
    if return_state:
        n_cyc = 1 + (len(y) - T_warm) // refresh_every
        state = {"s_mean": np.zeros(n_cyc), "gate": np.zeros(n_cyc),
                 "delta_over_v": np.zeros(n_cyc), "sig_fast": np.zeros(n_cyc),
                 "sigma2": np.zeros(n_cyc)}
        cyc_i = 0

    for t in range(T_warm, len(y)):
        # ---- solve in the cached eigenbasis: O(D^2) ----
        bbar_e = (U.T @ B) / n_eff
        what_e = bbar_e / mu_pos
        gate = sigma2 / (sigma2 + max(0.0, sig_fast - sigma2) + 1e-12)
        s_eff = s_vec * gate
        W_e = (1.0 - s_eff) * what_e + s_eff * G_e
        W = U @ W_e
        preds[t - T_warm] = x[t] @ W

        r_t = y[t] - x[t] @ W
        sig_fast = (1 - eta_f) * sig_fast + eta_f * (r_t * r_t)
        sigma2 = (1 - eta_s) * sigma2 + eta_s * (r_t * r_t)

        # ---- refresh at cadence: O(D^3) ----
        if step % refresh_every == 0 and step > 0:
            if return_state and cyc_i < n_cyc:
                state["s_mean"][cyc_i] = float(np.mean(s_vec))
                state["gate"][cyc_i] = float(gate)
                state["delta_over_v"][cyc_i] = float(
                    np.mean(delta2 / np.maximum(v_dir, 1e-16)))
                state["sig_fast"][cyc_i] = float(sig_fast)
                state["sigma2"][cyc_i] = float(sigma2)
                cyc_i += 1
            # 1) pull-identity refresh in the CURRENT cached basis
            #    (W_e, G_e, what_e are all old-basis: consistent)
            s_eff_r = s_vec * gate
            G_e = (1.0 - s_eff_r) * W_e + s_eff_r * G_e
            G = U @ G_e
            # 2) recompute the cached basis and re-derive state in NEW basis
            mu, U = np.linalg.eigh(A / n_eff)
            mu_pos = np.maximum(mu, 1e-12)
            mu_bar = float(np.mean(mu_pos))
            G_e = U.T @ G
            what_e = ((U.T @ B) / n_eff) / mu_pos
            v_dir = sigma2 / (n_eff * mu_pos)
            raw = (G_e - what_e) ** 2 - v_dir
            fl = 0.25 * sigma2 / (T_warm * mu_pos)
            raw = np.maximum(raw, fl)
            # cross-direction shrinkage (dead directions trust the mean)
            w_i = shrink * mu_pos / (mu_pos + mu_bar)
            raw = (1.0 - w_i) * raw + w_i * float(np.mean(raw))
            delta2 = (1 - eta_delta) * delta2 + eta_delta * raw
            rho_dir = mu_pos * v_dir / np.maximum(delta2, 1e-14)
            s_vec = np.clip(rho_dir / (mu_pos + rho_dir), 0.0, s_cap)
            s_vec[mu < 1e-9 * max(mu[-1], 1e-12)] = s_cap

        A = alpha * A + np.outer(x[t], x[t])
        B = alpha * B + x[t] * y[t]
        n_eff = alpha * n_eff + 1.0
        step += 1

    if return_state:
        return preds, state
    return preds


# ---------------------------------------------------------------- SPAR-Pull


def run_spar_kf(x, y, T_warm, G_init=None, refresh=True, refresh_every=50, gamma=1.0,
                w_min=5e-4, w_max=0.05, K_cal=10, eta_C=0.15, eta_nu=0.2,
                base_eta=SLOW_ETA, resid_fast=100, clip_lo=1e-5, clip_hi=1.0,
                rho_init=0.1, ridge=0.01, return_diag=False,
                rho_drift_floor=None, use_cusum=True, use_ref=True,
                adaptive_rho=True, debias_variance_floor=False,
                rho_pin=None, w_pin=None, prediction_units=False,
                sticky_floor=False, drift_margin=0.05, feedback_lag=0):
    """SPAR-Pull: both estimators self-calibrated from measured statistics.

    Estimator 1 (anchor strength, conflict theorem):
        rho* = mu_bar * V_hat / C_hat       (conflict -> light anchor)
    Estimator 2 (forgetting window, window-sign theorem):
        R(w) = c'*w/(2-w) + nu^2/w^2,  c' = (1-s)^2 sigma^2 D
        w* = (4 nu^2 / c')^(1/3)  (leading order; the exact variance term
        carries the factor w/(2-w))

    debias_variance_floor=False: estimator without the variance floor
        nu2 = E_resid * w^2 / (1-w)^2
    debias_variance_floor=True:  the variance-floor-subtracted inversion
        nu2 = max(E_resid * w^2 - c' * w^3/(2-w), 0)
    spar_kf2 denotes the two-estimator arm with the variance-floor-subtracted
    estimator.

    rho_pin / w_pin (diagnostics only, default None = unchanged behavior):
    when set, each refresh block overrides the respective estimator value with
    the pinned constant for the whole stream, AFTER the arm's own update
    rules; all statistics (C_ema, nu2_ema, ...) keep updating and the
    rho_drift_floor threshold keeps the arm's w_max, so a pinned run is an
    alternate trajectory of the SAME arm with one estimator held fixed.

    prediction_units=False: legacy parameter-unit conflict (reference branch
        in parameter units; residual branch C = E tr(A^-1)/(n mu_bar), which
        carries units [y^2/x^4] and makes rho scale as c^4 under phi -> c phi).
    prediction_units=True: every conflict branch in prediction-risk units
        [y^2]: V_sigma = sigma^2 d_act/n, C_ref = delta^T Abar delta - V_sigma,
        C_resid = E, C_floor = 0.5 sigma^2 d_act/T_warm, so
        rho = mu_bar V_sigma / C_sigma has units [x^2] on every branch and
        is invariant to feature rescaling; no inverse of Abar is formed.

    Returns (preds, rhos) by default, or (preds, rhos, ws) when
    ``return_diag=True``.
    """
    D = x.shape[1]
    A = x[:T_warm].T @ x[:T_warm].astype(np.float64)
    B = x[:T_warm].T @ y[:T_warm]
    n_eff = float(T_warm)

    W_warm_fit = np.linalg.solve(A + n_eff * ridge * np.eye(D), B)
    G = np.asarray(G_init, dtype=np.float64).copy() if G_init is not None \
        else W_warm_fit.copy()
    G_ref = G.copy()
    r_warm = y[:T_warm] - x[:T_warm] @ W_warm_fit
    sigma2_base = float(np.var(r_warm)) + 1e-12      # long-run noise baseline
    sigma2_fast = sigma2_base
    Abar_warm = A / n_eff
    # The trace inverse is only used by the legacy parameter-unit branch; the
    # prediction-unit path uses the active dimension instead.  Skip the O(D^3)
    # inverse there (numerically identical, materially faster at D >= 300).
    tr_inv_warm = (float(np.trace(np.linalg.inv(Abar_warm + RIDGE_TRACE * np.eye(D))))
                   if not prediction_units else 0.0)

    eta_f = 1.0 / resid_fast
    eta_ref = refresh_every / GREF_HORIZON
    C_floor_base = (0.5 * sigma2_base * D / T_warm if prediction_units
                    else 0.5 * sigma2_base * tr_inv_warm / T_warm)
    C_ema = 2 * C_floor_base
    nu2_ema = 0.0
    w = w_min
    alpha = 1.0 - w
    rho = rho_init

    preds = np.zeros(len(y) - T_warm)
    rhos = np.zeros(len(y) - T_warm)
    ws = np.zeros(len(y) - T_warm)
    step = 0
    cyc = 0

    for t in range(T_warm, len(y)):
        if step % refresh_every == 0:
            Abar = A / n_eff
            mu_bar = float(np.trace(Abar) / D) + 1e-12
            if not prediction_units:
                tr_inv = float(np.trace(np.linalg.inv(Abar + RIDGE_TRACE * np.eye(D))))
            else:
                tr_inv = 0.0
            W_data = np.linalg.solve(A + n_eff * RIDGE_REF * np.eye(D), B)
            E_resid = max(0.0, sigma2_fast - sigma2_base)

            # estimator 1: anchor strength
            if prediction_units:
                mu_all = np.linalg.eigvalsh(Abar)
                d_act = max(1, int(np.sum(mu_all > ACT_THRESH * max(float(mu_all.max()), 1e-12))))
                V_hat = sigma2_base * d_act / n_eff
                delta = G_ref - W_data
                C_ref = float(delta @ (Abar @ delta)) - V_hat
                C_resid = E_resid
                C_floor_use = 0.5 * sigma2_base * d_act / T_warm
            else:
                V_hat = sigma2_base * tr_inv / n_eff
                C_ref = float(np.sum((G_ref - W_data) ** 2)) - V_hat
                C_resid = E_resid * tr_inv / (n_eff * mu_bar)
                C_floor_use = C_floor_base
            rho, w, C_ema, nu2_ema = strength_window_update(
                C_ema=C_ema, nu2_ema=nu2_ema, V_hat=V_hat, C_ref=C_ref,
                C_resid=C_resid, C_floor=C_floor_use, E_resid=E_resid,
                mu_bar=mu_bar, rho=rho, w=w, sigma2_base=sigma2_base, D=D,
                cyc=cyc, K_cal=K_cal, eta_C=eta_C, eta_nu=eta_nu,
                w_min=w_min, w_max=w_max, clip_lo=clip_lo,
                cap=(rho_init if cyc < K_cal else clip_hi),
                adaptive_rho=adaptive_rho,
                debias_variance_floor=debias_variance_floor, use_ref=use_ref,
                use_cusum=use_cusum, rho_drift_floor=rho_drift_floor,
                rho_pin=rho_pin, w_pin=w_pin)
            alpha = 1.0 - w

        A_eff = A + n_eff * rho * np.eye(D)
        B_eff = B + n_eff * rho * gamma * G
        W = np.linalg.solve(A_eff, B_eff)
        preds[t - T_warm] = x[t] @ W
        rhos[t - T_warm] = rho
        ws[t - T_warm] = w

        u = max(0, t - feedback_lag)
        r_t = y[u] - x[u] @ W
        sigma2_fast = (1 - eta_f) * sigma2_fast + eta_f * (r_t * r_t)
        if sticky_floor:
            # Scale-free floor: update the slow baseline only while no drift
            # excess is present; under sustained drift it is frozen at the
            # pre-drift noise level, so the excess stays open-ended.
            if sigma2_fast <= sigma2_base * (1.0 + drift_margin):
                sigma2_base = (1 - base_eta) * sigma2_base + base_eta * (r_t * r_t)
        else:
            sigma2_base = (1 - base_eta) * sigma2_base + base_eta * (r_t * r_t)

        if refresh and step % refresh_every == 0 and step > 0:
            Abar = A / n_eff
            mu, U = np.linalg.eigh(Abar)
            beta = mu / (mu + rho)
            Gp = U.T @ G
            Wp = U.T @ W
            G = U @ (beta * Wp + (1 - beta) * Gp)
            G_ref = (1 - eta_ref) * G_ref + eta_ref * W
            cyc += 1

        A = alpha * A + np.outer(x[u], x[u])
        B = alpha * B + x[u] * y[u]
        n_eff = alpha * n_eff + 1.0
        step += 1

    if return_diag:
        return preds, rhos, ws
    return preds, rhos


# --------------------------------------------- baselines and the SPAR-EG arm


def run_spar_eg(x, y, T_warm, lams=(1.0, 0.997, 0.995, 0.99),
                G_init=None, ridge=0.01, extra_preds=None):
    """SPAR-EG: the expert-simplex composition.

    K expert paths -- batch-solved leaky accumulators at several windows
    (windup-immune) plus the frozen warm anchor -- combined by
    exponentiated-gradient weights on the simplex over squared losses.
    extra_preds: optional precomputed expert trajectory over the evaluation
    segment (e.g., the SPAR-Pull arm), included as an additional expert; the
    Hedge comparator class then contains that path as well.

    Guarantees (Hedge): with eta_t = sqrt(ln K / max(S_t, ln K)) where S_t
    is the accumulated mixture loss (anytime tuning),
        sum_t l_t(mix) <= min_i sum_t l_t(i) + 2 sqrt(ln(K) * sum_t l_mix)
    so the comparator class CONTAINS every leaky window we benchmark, the
    frozen anchor, and all their mixtures. Mixture-convexity of the squared
    loss ((sum w u)^2 <= sum w u^2) is the same lemma as the pull-identity
    interpolation bound (convex_interp_sq_le, formalized in Lean 4).
    """
    D = x.shape[1]
    K = len(lams) + 1 + (1 if extra_preds is not None else 0)
    # expert state: accumulators for each lam
    A_ex = []
    B_ex = []
    n_ex = []
    for lam in lams:
        A_ex.append(x[:T_warm].T @ x[:T_warm].astype(np.float64))
        B_ex.append(x[:T_warm].T @ y[:T_warm])
        n_ex.append(float(T_warm))
    # frozen anchor expert
    W_warm_fit = np.linalg.solve(
        x[:T_warm].T @ x[:T_warm] + T_warm * ridge * np.eye(D),
        x[:T_warm].T @ y[:T_warm])
    W_anchor = np.asarray(G_init, dtype=np.float64).copy() \
        if G_init is not None else W_warm_fit

    wts = np.full(K, 1.0 / K)
    log_wts = np.full(K, -np.log(K))
    S_mix = 0.0
    lnK = np.log(K)
    preds = np.zeros(len(y) - T_warm)
    for t in range(T_warm, len(y)):
        # expert predictions
        p_ex = np.zeros(K)
        for i, lam in enumerate(lams):
            W = np.linalg.solve(A_ex[i] + 0.01 * n_ex[i] * np.eye(D), B_ex[i])
            p_ex[i] = x[t] @ W
        if extra_preds is not None:
            p_ex[len(lams)] = extra_preds[t - T_warm]
        p_ex[K - 1] = x[t] @ W_anchor
        # mixture prediction (convexity: l(mix) <= sum w l_i)
        mix = float(wts @ p_ex)
        preds[t - T_warm] = mix
        # EG update on squared losses
        ell = (p_ex - y[t]) ** 2
        S_mix += (mix - y[t]) ** 2
        eta = np.sqrt(lnK / max(S_mix, lnK))
        log_wts = log_wts - eta * ell
        log_wts -= log_wts.max()
        wts = np.exp(log_wts)
        wts /= wts.sum()
        # accumulate expert paths
        for i, lam in enumerate(lams):
            A_ex[i] = lam * A_ex[i] + np.outer(x[t], x[t])
            B_ex[i] = lam * B_ex[i] + x[t] * y[t]
            n_ex[i] = lam * n_ex[i] + 1.0
    return preds


# ---------------------------------------------------------------- sanity


def _sanity() -> None:
    rng = np.random.default_rng(0)
    D, T, T_warm = 16, 600, 200
    x = rng.normal(size=(T, D))
    w = rng.normal(scale=0.5, size=D)
    y = x @ w + 0.05 * rng.normal(size=T)
    I = np.eye(D)

    # 1) RLS lam=1 equals the batch warm-ridge at every step
    #    (P0^-1 = A_warm + rho*T_warm*I stays FIXED as data accumulates)
    A = x[:T_warm].T @ x[:T_warm]
    b = x[:T_warm].T @ y[:T_warm]
    n = float(T_warm)
    W = np.linalg.solve(A + 0.1 * n * I, b)
    P = np.linalg.inv(A / n + 0.1 * I) / n
    max_dev = 0.0
    for t in range(T_warm, T):
        W_batch = np.linalg.solve(A + 0.1 * T_warm * I, b)
        max_dev = max(max_dev, float(np.max(np.abs(W - W_batch))))
        k = P @ x[t]
        g = k / (1.0 + x[t] @ k)
        W = W + g * (y[t] - x[t] @ W)
        P = P - np.outer(g, k)
        A = A + np.outer(x[t], x[t])
        b = b + x[t] * y[t]
        n += 1.0
    print(f"RLS(lam=1) vs batch warm-ridge: max |dev| = {max_dev:.2e}")

    # 2) DF(lam_f=1) reduces to RLS(lam=1)
    p_rls = run_rls(x, y, T_warm, lam=1.0, rho=0.1)
    p_df = run_df(x, y, T_warm, lam_f=1.0, rho=0.1)
    print(f"DF(lam_f=1) vs RLS(lam=1): max |dev| = "
          f"{float(np.max(np.abs(p_rls - p_df))):.2e}")

    # 3) KF(q=0, R=1, same rho init) matches RLS(lam=1)
    p_kf = run_kf(x, y, T_warm, q=0.0, R=1.0, rho_init=1.0)
    p_rls2 = run_rls(x, y, T_warm, lam=1.0, rho=1.0)
    print(f"KF(q=0,R=1,rho=1) vs RLS(lam=1,rho=1): max |dev| = "
          f"{float(np.max(np.abs(p_kf - p_rls2))):.2e}")

    # 4) WiSE-FT endpoints
    W0, Wft = wiseft_models(x, y, T_warm)
    a0 = wiseft_preds(x[T_warm:], W0, Wft, 0.0)
    a1 = wiseft_preds(x[T_warm:], W0, Wft, 1.0)
    print(f"WiSE-FT endpoints: max|a0 - xW0| = "
          f"{float(np.max(np.abs(a0 - x[T_warm:] @ W0))):.2e}, "
          f"max|a1 - xWft| = {float(np.max(np.abs(a1 - x[T_warm:] @ Wft))):.2e}")

    # 5) spar_diag runs and is finite (uses its own warm prior)
    p_sp = run_spar_diag(x, y, T_warm)
    print(f"spar_diag: finite={np.all(np.isfinite(p_sp))}, "
          f"mse={float(np.mean((p_sp - y[T_warm:])**2)):.4f}")

    # 6) stream generator axes
    for spectrum in ("iso", "decay3", "rankdef"):
        for drift in ("abrupt", "covrot", "both", "exdeath", "osc", "gradual"):
            X, Y, wt, t1, t2, Q0, lam0 = make_stream_fac(
                seed=1, D=32, T=900, spectrum=spectrum, drift=drift)
            assert np.all(np.isfinite(Y)), (spectrum, drift)
    print("stream generator: all axes finite")


if __name__ == "__main__":
    _sanity()
