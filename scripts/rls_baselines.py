"""Baseline method library for the synthetic factorial grid.

Implements the competitors that were previously only positioned against
structurally, plus the theory-exact diagonal-rho SPAR:

  run_rls        plain / leaky RLS (forgetting factor lam, ridge rho)
  run_df         DIRECTIONAL FORGETTING (Kulhavy 1987; Cao & Schwartz 2004)
                 -- the mechanism underlying SIFt-RLS (Palem et al. 2023 add
                 stabilization heuristics on top of this update; we implement
                 the canonical DF update with the standard small-gamma guard)
  run_arow       AROW (Crammer, Dredze & Pereira 2009), regression form
  run_kf         Kalman filter with random-walk state model (Q = qI, R given)
                 -- the drift-rate-oracle tracker
  run_sgd        online SGD on squared loss
  run_adam       online Adam on squared loss
  wiseft_models  WiSE-FT protocol: (zero-shot warm model, end-of-stream
                 fine-tuned model); caller sweeps the interpolation alpha
                 post-hoc (their protocol reports the alpha-envelope)
  run_spar_diag  DIAGONAL-rho SPAR: per-direction strength
                 rho_i* = mu_i * v_i / delta_i^2 (Theorem 5, theory-exact)

Also provides make_stream_fac: the multi-factor stream generator
(spectrum x drift-geometry x noise x prior axes).

Sanity checks (python scripts/rls_baselines.py):
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

from spar_test import run_spar_v2  # noqa: E402


# ---------------------------------------------------------------- streams

def make_stream_fac(seed=0, D=128, T=10000, sigma=0.1,
                    spectrum="iso", drift="abrupt", noise="gauss",
                    drift_frac=(1 / 3, 2 / 3), mismatch="none"):
    """Multi-factor three-phase stream with optional model mismatch.

    mismatch: none | nonlinear (y = tanh(x·w) + ε, linear head misspecified)
              | covshift (feature covariance doubles mid-stream)
              | hetnoise (observation-noise variance 0.01 -> 0.05 from t1)
    """
    rng = np.random.default_rng(seed)
    if spectrum == "iso":
        lam0 = np.ones(D)
    elif spectrum == "decay3":
        lam0 = np.logspace(0, -3, D)
    elif spectrum == "rankdef":
        lam0 = np.logspace(0, -2, D)
        lam0[D // 2:] = 0.0
    else:
        raise ValueError(spectrum)
    Q0, _ = np.linalg.qr(rng.normal(size=(D, D)))
    order = np.argsort(lam0)[::-1]
    lam0 = lam0[order]
    Q0 = Q0[:, order]

    t1, t2 = int(T * drift_frac[0]), int(T * drift_frac[1])
    w0 = rng.normal(scale=0.5, size=D)
    w1 = w0 + rng.normal(scale=0.7, size=D)

    w_true = np.zeros((T, D))
    if drift in ("abrupt", "both"):
        w_true[:] = w0
        w_true[t1:] = w1
    elif drift in ("gradual",):
        ramp = np.zeros(T)
        ramp[t1:t2] = np.linspace(0, 1, t2 - t1)
        ramp[t2:] = 1.0
        w_true[:] = (1 - ramp[:, None]) * w0 + ramp[:, None] * w1
    elif drift in ("covrot", "exdeath"):
        w_true[:] = w0
    elif drift == "osc":
        w_true[:] = w0
        per = 800
        for s in range(t1, T, per):
            if ((s - t1) // per) % 2 == 1:
                w_true[s:s + per] = w1
    elif drift == "none":
        w_true[:] = w0
    else:
        raise ValueError(drift)

    Z = rng.normal(size=(T, D))
    U = Z * np.sqrt(lam0)[None, :]
    if drift in ("covrot", "both"):
        theta = np.zeros(T)
        ramp = np.zeros(T)
        ramp[t1:t2] = np.linspace(0, 1, t2 - t1)
        ramp[t2:] = 1.0
        theta = (np.pi / 2) * ramp
        c, s_ = np.cos(theta), np.sin(theta)
        u0, u1 = U[:, 0].copy(), U[:, 1].copy()
        U[:, 0] = c * u0 - s_ * u1
        U[:, 1] = s_ * u0 + c * u1
    if drift == "exdeath":
        decay = np.ones(T)
        ramp = np.linspace(0, 1, t2 - t1)
        decay[t1:t2] = np.exp(-9 * ramp)
        decay[t2:] = np.exp(-9)
        U[:, :4] *= decay[:, None]
    if mismatch == "covshift":
        # double the feature scale for the second half of the stream
        U[t1:] *= 2.0

    X = U @ Q0.T

    if noise == "gauss":
        eps = sigma * rng.normal(size=T)
    elif noise == "t3":
        g = rng.standard_t(3, size=T)
        eps = sigma * g / np.sqrt(3.0 / (3.0 - 2.0))
    else:
        raise ValueError(noise)

    if mismatch == "hetnoise":
        # observation-noise scale change: sigma^2 0.01 -> 0.05 from t1 on
        eps = np.where(np.arange(T) >= t1, np.sqrt(5.0), 1.0) * eps

    if mismatch == "nonlinear":
        # y = tanh(x·w*) + ε: the linear head is misspecified
        y = np.tanh(np.einsum("td,td->t", X, w_true)) + eps
    else:
        y = np.einsum("td,td->t", X, w_true) + eps
    return X, y, w_true, t1, t2, Q0, lam0


def make_prior(prior, x, y, w_true, t1, T_warm, Q0, lam0, rng=None):
    """Anchor prior for the prior axis. Returns (G_init, label).

    warm: data-estimated from the warm-up window (None -> methods fit it)
    oracle: the true pre-drift teacher w0
    adversarial: w0 + 2*||w0||*q1  (worst case: error in the TOP direction)
    """
    D = x.shape[1]
    if prior == "warm":
        return None
    if prior == "oracle":
        return w_true[T_warm - 1].copy()
    if prior == "adversarial":
        w0 = w_true[T_warm - 1]
        return w0 + 2.0 * np.linalg.norm(w0) * Q0[:, 0]
    raise ValueError(prior)


# ---------------------------------------------------------------- methods

def run_rls(x, y, T_warm, lam=1.0, rho=0.1):
    """Plain (lam=1) or leaky (lam<1) RLS via the P recursion."""
    D = x.shape[1]
    A_w = x[:T_warm].T @ x[:T_warm]
    b_w = x[:T_warm].T @ y[:T_warm]
    W = np.linalg.solve(A_w + rho * T_warm * np.eye(D), b_w)
    P = np.linalg.inv(A_w / T_warm + rho * np.eye(D)) / T_warm
    preds = np.zeros(len(y) - T_warm)
    for t in range(T_warm, len(y)):
        preds[t - T_warm] = x[t] @ W
        k = P @ x[t]
        gain = k / (lam + x[t] @ k)
        W = W + gain * (y[t] - x[t] @ W)
        P = (P - np.outer(gain, k)) / lam
    return preds


def run_vff(x, y, T_warm, lam0=1.0, lam_min=0.95, rho=0.1):
    """Variable Forgetting Factor RLS (Fortescue, Kershenbaum & Ydstie,
    1981): the per-step forgetting factor keeps the information content
    near a target, lam_t = 1 - e_t^2 / (lam0 * sigma0), clipped to
    [lam_min, 1]. Covariance-form recursion (the standard VFF family;
    the robust variant is Paleologu, Benesty & Ciochina, 2008)."""
    D = x.shape[1]
    A_w = x[:T_warm].T @ x[:T_warm]
    b_w = x[:T_warm].T @ y[:T_warm]
    W = np.linalg.solve(A_w + rho * T_warm * np.eye(D), b_w)
    P = np.linalg.inv(A_w / T_warm + rho * np.eye(D)) / T_warm
    r_warm = y[:T_warm] - x[:T_warm] @ W
    sigma0 = float(np.var(r_warm)) + 1e-12
    preds = np.zeros(len(y) - T_warm)
    for t in range(T_warm, len(y)):
        e = y[t] - x[t] @ W
        preds[t - T_warm] = x[t] @ W
        lam = float(np.clip(1.0 - (e * e) / (lam0 * sigma0), lam_min, 1.0))
        k = P @ x[t]
        gain = k / (lam + x[t] @ k)
        W = W + gain * e
        P = (P - np.outer(gain, k)) / lam
    return preds


def run_df(x, y, T_warm, lam_f=0.98, rho=0.1, eps=1e-12):
    """Directional forgetting (Kulhavy 1987; Cao & Schwartz 2004).

    Forgets ONLY along the incoming data direction:
        P <- P + ((1-lam_f)/lam_f) (P x x^T P)/(x^T P x)
    then the standard RLS measurement update. Directions orthogonal to the
    data keep their information (no covariance windup) -- the property
    SIFt-RLS stabilizes and exploits.
    """
    D = x.shape[1]
    A_w = x[:T_warm].T @ x[:T_warm]
    b_w = x[:T_warm].T @ y[:T_warm]
    W = np.linalg.solve(A_w + rho * T_warm * np.eye(D), b_w)
    P = np.linalg.inv(A_w / T_warm + rho * np.eye(D)) / T_warm
    preds = np.zeros(len(y) - T_warm)
    for t in range(T_warm, len(y)):
        preds[t - T_warm] = x[t] @ W
        g = x[t] @ P @ x[t]
        if g > eps:
            k = P @ x[t]
            P = P + ((1.0 - lam_f) / lam_f) * np.outer(k, k) / g
        k2 = P @ x[t]
        denom = 1.0 + x[t] @ k2
        gain = k2 / denom
        W = W + gain * (y[t] - x[t] @ W)
        P = P - np.outer(gain, k2)
    return preds


def run_arow(x, y, T_warm, r=0.1):
    """AROW regression: RLS measurement update with aggressive-regularized gain."""
    D = x.shape[1]
    A_w = x[:T_warm].T @ x[:T_warm]
    b_w = x[:T_warm].T @ y[:T_warm]
    W = np.linalg.solve(A_w + 1e-3 * T_warm * np.eye(D), b_w)
    S = np.linalg.inv(A_w / T_warm + r * np.eye(D)) / T_warm
    preds = np.zeros(len(y) - T_warm)
    for t in range(T_warm, len(y)):
        preds[t - T_warm] = x[t] @ W
        k = S @ x[t]
        sig2 = x[t] @ k + r
        gain = k / sig2
        W = W + gain * (y[t] - x[t] @ W)
        S = S - np.outer(k, k) / sig2
    return preds


def run_kf(x, y, T_warm, q=1e-5, R=0.01, rho_init=1e-3):
    """Kalman filter, random-walk state model w_t = w_{t-1} + N(0, qI)."""
    D = x.shape[1]
    A_w = x[:T_warm].T @ x[:T_warm]
    b_w = x[:T_warm].T @ y[:T_warm]
    W = np.linalg.solve(A_w + rho_init * T_warm * np.eye(D), b_w)
    P = np.linalg.inv(A_w / T_warm + rho_init * np.eye(D)) / T_warm
    preds = np.zeros(len(y) - T_warm)
    for t in range(T_warm, len(y)):
        P = P + q * np.eye(D)
        preds[t - T_warm] = x[t] @ W
        k = P @ x[t]
        K = k / (x[t] @ k + R)
        W = W + K * (y[t] - x[t] @ W)
        P = P - np.outer(K, k)
    return preds


def run_sgd(x, y, T_warm, lr=0.01):
    D = x.shape[1]
    A_w = x[:T_warm].T @ x[:T_warm]
    b_w = x[:T_warm].T @ y[:T_warm]
    W = np.linalg.solve(A_w + 1e-3 * T_warm * np.eye(D), b_w)
    preds = np.zeros(len(y) - T_warm)
    for t in range(T_warm, len(y)):
        preds[t - T_warm] = x[t] @ W
        e = x[t] @ W - y[t]
        W = W - lr * e * x[t]
    return preds


def run_adam(x, y, T_warm, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
    D = x.shape[1]
    A_w = x[:T_warm].T @ x[:T_warm]
    b_w = x[:T_warm].T @ y[:T_warm]
    W = np.linalg.solve(A_w + 1e-3 * T_warm * np.eye(D), b_w)
    m = np.zeros(D)
    v = np.zeros(D)
    preds = np.zeros(len(y) - T_warm)
    for t in range(T_warm, len(y)):
        preds[t - T_warm] = x[t] @ W
        g = (x[t] @ W - y[t]) * x[t]
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** (t - T_warm + 1))
        vh = v / (1 - b2 ** (t - T_warm + 1))
        W = W - lr * mh / (np.sqrt(vh) + eps)
    return preds


def wiseft_models(x, y, T_warm, rho=0.01):
    """WiSE-FT protocol: zero-shot = warm-up ridge; fine-tuned = end-of-stream
    accumulated solve (full stream, offline). Caller sweeps alpha post-hoc."""
    D = x.shape[1]
    A_w = x[:T_warm].T @ x[:T_warm]
    b_w = x[:T_warm].T @ y[:T_warm]
    W_zero = np.linalg.solve(A_w + rho * T_warm * np.eye(D), b_w)
    n = float(len(x))
    A_all = x.T @ x
    b_all = x.T @ y
    W_ft = np.linalg.solve(A_all + rho * n * np.eye(D), b_all)
    return W_zero, W_ft


def wiseft_preds(x, W_zero, W_ft, alpha):
    return x @ ((1 - alpha) * W_zero + alpha * W_ft)


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
                base_eta=5e-5, resid_fast=100, clip_lo=1e-5, clip_hi=1.0,
                rho_init=0.1, ridge=0.01, return_diag=False,
                rho_drift_floor=None, use_cusum=True, use_ref=True,
                adaptive_rho=True, debias_variance_floor=False,
                rho_pin=None, w_pin=None, prediction_units=False):
    """SPAR-Pull: both estimators self-calibrated from measured statistics.

    Estimator 1 (anchor strength, conflict theorem):
        rho* = mu_bar * V_hat / C_hat       (conflict -> light anchor)
    Estimator 2 (forgetting window, window-sign theorem):
        R(w) = c'*w/(2-w) + nu^2/w^2,  c' = (1-s)^2 sigma^2 D
        w* = (4 nu^2 / c')^(1/3)  (leading order; the exact variance term
        w/(2-w) replaces the earlier sigma^2 D w with its factor-2 error)

    debias_variance_floor=False: the original estimator
        nu2 = E_resid * w^2 / (1-w)^2   [defective: ignores the variance
        term, no positive fixed point, climbs to the cap]
    debias_variance_floor=True:  the variance-floor-subtracted inversion
        nu2 = max(E_resid * w^2 - c' * w^3/(2-w), 0)   [exact at any w]
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
    tr_inv_warm = (float(np.trace(np.linalg.inv(Abar_warm + 1e-8 * np.eye(D))))
                   if not prediction_units else 0.0)

    eta_f = 1.0 / resid_fast
    eta_ref = refresh_every / 2000.0
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
                tr_inv = float(np.trace(np.linalg.inv(Abar + 1e-8 * np.eye(D))))
            else:
                tr_inv = 0.0
            W_data = np.linalg.solve(A + n_eff * 1e-6 * np.eye(D), B)
            E_resid = max(0.0, sigma2_fast - sigma2_base)

            # estimator 1: anchor strength
            if prediction_units:
                mu_all = np.linalg.eigvalsh(Abar)
                d_act = max(1, int(np.sum(mu_all > 1e-8 * max(float(mu_all.max()), 1e-12))))
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
            cands = []
            if use_ref:
                cands.append(C_ref)
            if use_cusum:
                cands.append(C_resid)
            cands += [C_floor_use, 1e-10]
            C_raw = max(cands)
            C_ema = (1 - eta_C) * C_ema + eta_C * C_raw
            if adaptive_rho:
                rho_raw = mu_bar * V_hat / C_ema
                cap = rho_init if cyc < K_cal else clip_hi
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
            alpha = 1.0 - w

        A_eff = A + n_eff * rho * np.eye(D)
        B_eff = B + n_eff * rho * gamma * G
        W = np.linalg.solve(A_eff, B_eff)
        preds[t - T_warm] = x[t] @ W
        rhos[t - T_warm] = rho
        ws[t - T_warm] = w

        r_t = y[t] - x[t] @ W
        sigma2_fast = (1 - eta_f) * sigma2_fast + eta_f * (r_t * r_t)
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

        A = alpha * A + np.outer(x[t], x[t])
        B = alpha * B + x[t] * y[t]
        n_eff = alpha * n_eff + 1.0
        step += 1

    if return_diag:
        return preds, rhos, ws
    return preds, rhos


# --------------------------------------------- baselines and the SPAR-EG arm

def run_kf_est(x, y, T_warm, q=1e-4, rho_init=1e-3, eta_R=0.01):
    """Kalman random-walk with R ESTIMATED online from squared innovations
    (fast EMA, horizon ~1/eta_R) -- R is estimated from residuals, not an
    oracle. The noise-adaptive fair baseline."""
    D = x.shape[1]
    A_w = x[:T_warm].T @ x[:T_warm]
    b_w = x[:T_warm].T @ y[:T_warm]
    W = np.linalg.solve(A_w + rho_init * T_warm * np.eye(D), b_w)
    P = np.linalg.inv(A_w / T_warm + rho_init * np.eye(D)) / T_warm
    r_warm = y[:T_warm] - x[:T_warm] @ W
    R_est = float(np.var(r_warm)) + 1e-12
    preds = np.zeros(len(y) - T_warm)
    for t in range(T_warm, len(y)):
        P = P + q * np.eye(D)
        preds[t - T_warm] = x[t] @ W
        r = y[t] - x[t] @ W
        R_est = (1 - eta_R) * R_est + eta_R * (r * r)
        k = P @ x[t]
        K = k / (x[t] @ k + R_est)
        W = W + K * r
        P = P - np.outer(K, k)
    return preds


def run_leaky_batch(x, y, T_warm, lam=0.995, ridge_frac=0.01):
    """Batch (information-form) solve of the exponentially-weighted LS with
    matched ridge: the cliff-figure control. Maintains the SAME alpha-window
    accumulator as leaky RLS but solves directly, immune to the covariance
    recursion's numerical windup. If the lambda-cliff is intrinsic to short
    windows (rank deficiency), this arm still suffers but must NOT explode."""
    D = x.shape[1]
    A = x[:T_warm].T @ x[:T_warm].astype(np.float64)
    B = x[:T_warm].T @ y[:T_warm]
    n_eff = float(T_warm)
    ridge = ridge_frac
    preds = np.zeros(len(y) - T_warm)
    for t in range(T_warm, len(y)):
        W = np.linalg.solve(A + ridge * n_eff * np.eye(D), B)
        preds[t - T_warm] = x[t] @ W
        A = lam * A + np.outer(x[t], x[t])
        B = lam * B + x[t] * y[t]
        n_eff = lam * n_eff + 1.0
    return preds


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
