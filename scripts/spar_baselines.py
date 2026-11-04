"""Competitor baselines for the SPAR factorial grid.

Split out of ``rls_baselines``: the closed-form and first-order baselines
(leaky RLS, VFF, directional forgetting, AROW, Kalman, SGD/Adam, WiSE-FT, the
estimated-R Kalman, and the batch leaky solve).  No SPAR code lives here.

The forward model, the priors, and the SPAR heads are in ``spar_streams`` and
``rls_baselines`` respectively.
"""

import numpy as np


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


