"""SPAR: Spectrally-Adaptive Anchor Refresh — pull-identity refresh,
adaptive strength via reference-conflict + residual CUSUM, accumulator forgetting.

Theory encoded (see paper/appendix.tex):
  Pull identity (EXACT algebra, any gamma):
     W_i - w = (1-s_i)(what_i - w) + s_i(gamma*G_i - w),   s_i = rho/(mu_i+rho)
  Risk interpolation (gamma=1, unbiased data):
     R_i = (1-s_i)^2 V_i + s_i^2 Delta_i^2
  Dominance (Delta=0): R_SPAR = (1-s_i)^2 V_i < V_i strictly for rho>0
  Global sign: optimal rho_i* = sigma^2/(n*delta_i^2); lumped
     rho* = mu_bar * V_hat / C_hat  ->  DECREASES during drift
  T5 Single estimator: rho controls pull s_i and refresh beta_i = 1 - s_i jointly.
  Robustness: the anchored solve (A + n*rho*I)^{-1} has condition number
     <= 1 + mu_max/rho ALWAYS — forgetting cannot cause covariance windup
     (plain leaky RLS diverges at lambda=0.99 under poor excitation).

Conflict estimator: the reference branch measures staleness against a slow
anchor because the working anchor co-moves with the solve, so
||G - W_data||^2 alone does not detect drift:
  - G_ref: slow reference anchor (horizon ref_horizon steps), independent
    enough of the working anchor to detect drift, converges to new
    equilibria so rho* can re-rise after drift resolves.
  - Two-scale residual CUSUM: sigma2_fast (horizon ~100) vs sigma2_slow
    (horizon ~2000); excess maps to parameter-space conflict.
  - C_hat = max(C_ref_debiased, C_resid, C_floor), EMA-smoothed.
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_stream(seed=0, D=32, T=10000, noise=0.1, spectrum_decay=3.0,
                drift_frac=(1/3, 2/3), drift_dirs="all", drift_duration=None):
    rng = np.random.default_rng(seed)
    eigvals = np.logspace(0, -spectrum_decay, D)
    Q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    Sigma = Q @ np.diag(eigvals) @ Q.T
    L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(D))
    x = rng.normal(size=(T, D)) @ L.T

    w0 = rng.normal(scale=0.5, size=D)
    w1 = w0.copy()
    if drift_dirs == "all":
        w1 += rng.normal(scale=0.7, size=D)
    elif drift_dirs == "excited":
        w1 += Q[:, :D // 2] @ rng.normal(scale=1.0, size=D // 2)
    elif drift_dirs == "dead":
        w1 += Q[:, D // 2:] @ rng.normal(scale=1.0, size=D - D // 2)

    t1, t2 = int(T * drift_frac[0]), int(T * drift_frac[1])
    if drift_duration is not None:
        # Fixed-duration drift phase: the per-step drift rate is held
        # constant across horizons (independent of T).
        t2 = min(t1 + int(drift_duration), T)
    w_true = np.zeros((T, D))
    w_true[:t1] = w0
    w_true[t2:] = w1
    ramp = np.linspace(0.0, 1.0, t2 - t1)
    w_true[t1:t2] = (1 - ramp[:, None]) * w0 + ramp[:, None] * w1

    y = np.einsum("td,td->t", x, w_true) + noise * rng.normal(size=T)
    return x, y, w_true, t1, t2, eigvals, Q


def check_pull_identity(x, y, w_true, T_warm, rho=0.1, gamma=1.0, at_step=5000):
    """Pull identity exact check: holds to machine precision."""
    D = x.shape[1]
    A = x[:at_step].T @ x[:at_step]
    B = x[:at_step].T @ y[:at_step]
    n = float(at_step)
    G = np.linalg.solve(x[:T_warm].T @ x[:T_warm] + T_warm * 0.01 * np.eye(D),
                        x[:T_warm].T @ y[:T_warm])
    mu, U = np.linalg.eigh(A / n)
    bbar = (U.T @ B) / n
    Gp = U.T @ G
    w = U.T @ w_true[at_step]
    s = rho / (mu + rho)
    what = bbar / mu
    Wp = (bbar + rho * gamma * Gp) / (mu + rho)
    lhs = Wp - w
    rhs = (1 - s) * (what - w) + s * (gamma * Gp - w)
    return float(np.max(np.abs(lhs - rhs)))


def run_spar_v2(x, y, T_warm, rho_init=0.1, alpha=1.0, adaptive_rho=True,
                refresh=True, refresh_every=50, gamma=1.0,
                clip_lo=1e-5, clip_hi=1.0, K_cal=10,
                eta_C=0.15, ref_horizon=2000, resid_fast=100,
                G_init=None, return_diag=False):
    """SPAR v2.1 runner.

    G_init: optional prior override (oracle / adversarial prior axes). None
      -> warm-up fit. The noise estimate always comes from the warm-up FIT
      so an adversarial prior cannot corrupt the variance estimate.
    return_diag: also return the CUSUM trajectory for the false-escalation test.
    """
    D = x.shape[1]
    A = x[:T_warm].T @ x[:T_warm].astype(np.float64)
    B = x[:T_warm].T @ y[:T_warm]
    n_eff = float(T_warm)

    W_warm_fit = np.linalg.solve(A + n_eff * 0.01 * np.eye(D), B)
    G = np.asarray(G_init, dtype=np.float64).copy() if G_init is not None \
        else W_warm_fit.copy()
    G_ref = G.copy()
    r_warm = y[:T_warm] - x[:T_warm] @ W_warm_fit
    sigma2_slow = float(np.var(r_warm)) + 1e-12
    sigma2_fast = sigma2_slow
    Abar_warm = A / n_eff
    tr_inv_warm = float(np.trace(np.linalg.inv(Abar_warm + 1e-8 * np.eye(D))))

    eta_f = 1.0 / resid_fast
    eta_s = 1.0 / ref_horizon
    eta_ref = refresh_every / ref_horizon
    C_floor_base = 0.5 * sigma2_slow * tr_inv_warm / T_warm
    C_ema = 2 * C_floor_base
    rho = rho_init

    preds = np.zeros(len(y) - T_warm)
    rhos = np.zeros(len(y) - T_warm)
    sig_fast = np.zeros(len(y) - T_warm)
    sig_slow = np.zeros(len(y) - T_warm)
    step = 0
    cyc = 0

    for t in range(T_warm, len(y)):
        if adaptive_rho and step % refresh_every == 0:
            Abar = A / n_eff
            mu_bar = float(np.trace(Abar) / D) + 1e-12
            tr_inv = float(np.trace(np.linalg.inv(Abar + 1e-8 * np.eye(D))))
            V_hat = sigma2_slow * tr_inv / n_eff
            W_data = np.linalg.solve(A + n_eff * 1e-6 * np.eye(D), B)

            # reference-anchor conflict (independent of working anchor's co-movement)
            C_ref = float(np.sum((G_ref - W_data) ** 2)) - V_hat
            # residual CUSUM conflict: fast excess over slow baseline
            E_resid = max(0.0, sigma2_fast - sigma2_slow)
            C_resid = E_resid * tr_inv / (n_eff * mu_bar)
            C_raw = max(C_ref, C_resid, C_floor_base, 1e-10)
            C_ema = (1 - eta_C) * C_ema + eta_C * C_raw
            rho_raw = mu_bar * V_hat / C_ema
            cap = rho_init if cyc < K_cal else clip_hi
            rho = float(np.clip(rho_raw, clip_lo, cap))

        A_eff = A + n_eff * rho * np.eye(D)
        B_eff = B + n_eff * rho * gamma * G
        W = np.linalg.solve(A_eff, B_eff)
        preds[t - T_warm] = x[t] @ W
        rhos[t - T_warm] = rho
        sig_fast[t - T_warm] = sigma2_fast
        sig_slow[t - T_warm] = sigma2_slow

        r_t = y[t] - x[t] @ W
        sigma2_fast = (1 - eta_f) * sigma2_fast + eta_f * (r_t * r_t)
        sigma2_slow = (1 - eta_s) * sigma2_slow + eta_s * (r_t * r_t)

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
        return preds, rhos, (sig_fast, sig_slow)
    return preds, rhos


def run_no_anchor(x, y, T_warm, rho=0.01, alpha=1.0):
    D = x.shape[1]
    A = x[:T_warm].T @ x[:T_warm].astype(np.float64)
    B = x[:T_warm].T @ y[:T_warm]
    n_eff = float(T_warm)
    preds = np.zeros(len(y) - T_warm)
    for t in range(T_warm, len(y)):
        W = np.linalg.solve(A + n_eff * rho * np.eye(D), B)
        preds[t - T_warm] = x[t] @ W
        A = alpha * A + np.outer(x[t], x[t])
        B = alpha * B + x[t] * y[t]
        n_eff = alpha * n_eff + 1.0
    return preds


def run_fixed_anchor(x, y, T_warm, rho, gamma=1.0):
    D = x.shape[1]
    A = x[:T_warm].T @ x[:T_warm].astype(np.float64)
    B = x[:T_warm].T @ y[:T_warm]
    n_eff = float(T_warm)
    G = np.linalg.solve(A + n_eff * 0.01 * np.eye(D), B)
    preds = np.zeros(len(y) - T_warm)
    for t in range(T_warm, len(y)):
        W = np.linalg.solve(A + n_eff * rho * np.eye(D), B + n_eff * rho * gamma * G)
        preds[t - T_warm] = x[t] @ W
        A = A + np.outer(x[t], x[t])
        B = B + x[t] * y[t]
        n_eff += 1.0
    return preds


def run_leaky_rls(x, y, T_warm, lam, rho=0.1):
    """TRUE leaky RLS with forgetting factor lam (lam=1 -> plain RLS)."""
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


def main() -> None:
    t0 = time.perf_counter()
    T_warm = 2000
    D = 32
    noise = 0.1
    n_seeds = 5

    configs = [
        {"name": "ill-conditioned (decay=3, kappa~1000)", "decay": 3.0},
        {"name": "isotropic (decay=0)", "decay": 0.0},
    ]

    for cfg in configs:
        print(f"\n{'='*78}")
        print(f"=== {cfg['name']} ===")
        print(f"{'='*78}")

        all_results = {}
        rho_trajs = {k: [] for k in ("pre", "drift", "post")}
        for seed in range(n_seeds):
            x, y, w_true, t1, t2, eigvals, Q = make_stream(
                seed=seed, D=D, noise=noise, spectrum_decay=cfg["decay"])
            y_te = y[T_warm:]
            regions = {
                "pre": slice(0, t1 - T_warm),
                "drift": slice(t1 - T_warm, t2 - T_warm),
                "post": slice(t2 - T_warm, None),
            }

            methods = {
                "no_anchor": run_no_anchor(x, y, T_warm, 0.01),
                "fixed_anchor": run_fixed_anchor(x, y, T_warm, 0.1),
                "spar_fix": run_spar_v2(x, y, T_warm, rho_init=0.1, alpha=1.0,
                                         adaptive_rho=False, refresh=True)[0],
                "spar_adp": run_spar_v2(x, y, T_warm, rho_init=0.1, alpha=1.0,
                                         adaptive_rho=True, refresh=True)[0],
                "spar_forg_fixrho": run_spar_v2(x, y, T_warm, rho_init=0.1,
                                                 alpha=0.997, adaptive_rho=False,
                                                 refresh=True)[0],
                "spar_full": run_spar_v2(x, y, T_warm, rho_init=0.1, alpha=0.997,
                                          adaptive_rho=True, refresh=True)[0],
            }

            # alpha sweep for spar_full
            for a in (1.0, 0.999, 0.998, 0.997, 0.995):
                methods[f"spar_full[{a}]"] = run_spar_v2(
                    x, y, T_warm, rho_init=0.1, alpha=a,
                    adaptive_rho=True, refresh=True)[0]

            # leaky RLS sweep
            for lam in (1.0, 0.999, 0.998, 0.997, 0.996, 0.995, 0.99):
                methods[f"leaky[{lam}]"] = run_leaky_rls(x, y, T_warm, lam)

            # post-hoc oracle: best (rho, alpha) per phase
            grid = [(float(r), a) for r in np.logspace(-4, 0, 13)
                    for a in (1.0, 0.998, 0.995)]
            per_phase = {rn: (np.inf, None) for rn in regions}
            for rho_t, a in grid:
                p, _ = run_spar_v2(x, y, T_warm, rho_init=rho_t, alpha=a,
                                   adaptive_rho=False, refresh=True)
                for rn, reg in regions.items():
                    mse = float(np.mean((p[reg] - y_te[reg]) ** 2))
                    if mse < per_phase[rn][0]:
                        per_phase[rn] = (mse, (rho_t, a))
            op = np.full(len(y_te), np.nan)
            for rn, reg in regions.items():
                rho_b, a_b = per_phase[rn][1]
                p, _ = run_spar_v2(x, y, T_warm, rho_init=rho_b, alpha=a_b,
                                   adaptive_rho=False, refresh=True)
                op[reg] = p[reg]
            methods["oracle_phase"] = op

            _, rhos_adp = run_spar_v2(x, y, T_warm, rho_init=0.1, alpha=0.997,
                                       adaptive_rho=True, refresh=True)
            for rn, reg in regions.items():
                rho_trajs[rn].append(float(np.mean(rhos_adp[reg])))

            for name, preds in methods.items():
                for rn, reg in regions.items():
                    all_results.setdefault((name, rn), []).append(
                        float(np.nanmean((preds[reg] - y_te[reg]) ** 2)))

        # print table: reference arms + best-per-family
        def agg(name, rn):
            return np.mean(all_results[(name, rn)])

        fam_spar = sorted({n[0] for n in all_results if n[0].startswith("spar_full[")})
        fam_leaky = sorted({n[0] for n in all_results if n[0].startswith("leaky[")})

        print(f"\n{'method':<18} {'pre-drift':>12} {'during':>12} {'post':>12} {'overall':>12}")
        print("-" * 72)
        rows = [("no_anchor", "no_anchor"), ("fixed_anchor", "fixed_anchor"),
                ("spar_fix", "spar_fix"), ("spar_adp", "spar_adp"),
                ("spar_forg_fixrho", "spar_forg_fixrho"), ("spar_full", "spar_full"),
                ("oracle_phase", "oracle_phase")]
        for label, name in rows:
            v = [agg(name, rn) for rn in ("pre", "drift", "post")]
            print(f"{label:<18} {v[0]:>12.6f} {v[1]:>12.6f} {v[2]:>12.6f} "
                  f"{(v[0]+v[1]+v[2])/3:>12.6f}")

        print(f"\n  best single-config by overall:")
        for fam, fname in ((fam_spar, "spar_full"), (fam_leaky, "leaky")):
            best = min(fam, key=lambda n: sum(agg(n, rn) for rn in ("pre", "drift", "post")))
            v = [agg(best, rn) for rn in ("pre", "drift", "post")]
            print(f"    {best} : {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} overall {(v[0]+v[1]+v[2])/3:.6f}")

        print(f"\n  best per-phase (post-hoc oracle references):")
        for rn in ("pre", "drift", "post"):
            bs = min(fam_spar, key=lambda n: agg(n, rn))
            bl = min(fam_leaky, key=lambda n: agg(n, rn))
            print(f"    {rn:>6}: spar {bs}={agg(bs, rn):.6f} | leaky {bl}={agg(bl, rn):.6f}")

        pre_r = np.mean(rho_trajs["pre"]); dr_r = np.mean(rho_trajs["drift"])
        po_r = np.mean(rho_trajs["post"])
        print(f"\n  adaptive rho* (alpha=0.997): pre={pre_r:.5f} drift={dr_r:.5f} post={po_r:.5f}")
        print(f"  T4 sign check: rho*(drift) < rho*(pre): {dr_r < pre_r}"
              f" | rho*(post) > rho*(drift): {po_r > dr_r}")

        # divergence count (robustness check)
        div = [n for n in fam_leaky if agg(n, "post") > 100]
        print(f"  leaky arms diverged (post MSE > 100): {len(div)}/{len(fam_leaky)}"
              f" — anchored arms: provably immune (kappa <= 1 + mu_max/rho)")

    x, y, w_true, t1, t2, _, _ = make_stream(seed=0, D=D, noise=noise, spectrum_decay=3.0)
    resid = check_pull_identity(x, y, w_true, T_warm, rho=0.1, gamma=1.0, at_step=5000)
    print(f"\nPull identity residual: {resid:.2e}")
    print(f"total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
