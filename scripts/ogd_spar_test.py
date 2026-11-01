"""Empirical validation of the no-regret composition (Proposition 4): OGD on the
anchor pull s achieves regret O(sqrt(T)) vs min(data path, anchor path).

The pull identity makes the loss CONVEX in s exactly:
    W(s) = (1-s)*w_hat + s*G   (segment family, per direction)
    l(s) = (x @ W(s) - y)^2    (convex in s: affine composed with square)
Online gradient descent on s_t in [0,1] against ADVERSARIAL/endogenous
component streams gives the Zinkevich guarantee:
    sum l(s_t) <= min_s sum l(s) + G_b sqrt(T)
and since s=0 (pure data) and s=1 (pure anchor) are in the class:
    R_T(OGD) <= min(R_T(data path), R_T(anchor path)) + G_b sqrt(T).

This script measures the actual regret on the three-phase stream: the
experienced loss of OGD-s vs the best FIXED s in hindsight (grid), vs the
two endpoints, and vs the per-phase oracle.
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from spar_test import make_stream, run_leaky_rls, run_spar_v2  # noqa: E402


def run_ogd_spar(x, y, T_warm, alpha=0.997, rho_ref=0.1, eps0=0.01,
                  refresh_every=50, eta0=0.05, s_init=0.5, G_init=None):
    """OGD on the pull s over the segment family, with SPAR-refreshed anchor.

    Component streams (both endogenous to the algorithm's own history —
    the Zinkevich bound tolerates this):
      w_hat_t : eps0-ridge data solve from the forgetting accumulator
      G_t     : SPAR-refreshed anchor (pull-identity rates at rho_ref)
    The learner plays s_t, predicts with (1-s)w_hat + s G, then updates
    s by OGD on the revealed convex loss. G_init selects the anchor
    prior (None = the warm-up fit; oracle/adversarial priors pass the
    corresponding vector).
    """
    D = x.shape[1]
    A = x[:T_warm].T @ x[:T_warm].astype(np.float64)
    B = x[:T_warm].T @ y[:T_warm]
    n_eff = float(T_warm)
    G = np.asarray(G_init, dtype=np.float64).copy() if G_init is not None \
        else np.linalg.solve(A + n_eff * 0.01 * np.eye(D), B)

    preds = np.zeros(len(y) - T_warm)
    s_traj = np.zeros(len(y) - T_warm)
    s = s_init
    step = 0

    for t in range(T_warm, len(y)):
        w_hat = np.linalg.solve(A + n_eff * eps0 * np.eye(D), B)
        W = (1 - s) * w_hat + s * G
        preds[t - T_warm] = x[t] @ W
        s_traj[t - T_warm] = s

        # revealed loss and exact gradient in s
        err = x[t] @ W - y[t]
        grad = 2.0 * err * (x[t] @ (G - w_hat))
        eta = eta0 / np.sqrt(step + 1.0)
        s = float(np.clip(s - eta * grad, 0.0, 1.0))

        # SPAR refresh at reference rates
        if step % refresh_every == 0 and step > 0:
            Abar = A / n_eff
            mu, U = np.linalg.eigh(Abar)
            beta = mu / (mu + rho_ref)
            Gp = U.T @ G
            Wp = U.T @ W
            G = U @ (beta * Wp + (1 - beta) * Gp)

        A = alpha * A + np.outer(x[t], x[t])
        B = alpha * B + x[t] * y[t]
        n_eff = alpha * n_eff + 1.0
        step += 1

    return preds, s_traj


def run_segment_fixed(x, y, T_warm, s_fixed, alpha=0.997, rho_ref=0.1,
                      eps0=0.01, refresh_every=50):
    """A FIXED s on the same segment family (the comparison class)."""
    D = x.shape[1]
    A = x[:T_warm].T @ x[:T_warm].astype(np.float64)
    B = x[:T_warm].T @ y[:T_warm]
    n_eff = float(T_warm)
    G = np.linalg.solve(A + n_eff * 0.01 * np.eye(D), B)
    preds = np.zeros(len(y) - T_warm)
    step = 0
    for t in range(T_warm, len(y)):
        w_hat = np.linalg.solve(A + n_eff * eps0 * np.eye(D), B)
        W = (1 - s_fixed) * w_hat + s_fixed * G
        preds[t - T_warm] = x[t] @ W
        if step % refresh_every == 0 and step > 0:
            Abar = A / n_eff
            mu, U = np.linalg.eigh(Abar)
            beta = mu / (mu + rho_ref)
            Gp = U.T @ G
            Wp = U.T @ W
            G = U @ (beta * Wp + (1 - beta) * Gp)
        A = alpha * A + np.outer(x[t], x[t])
        B = alpha * B + x[t] * y[t]
        n_eff = alpha * n_eff + 1.0
        step += 1
    return preds


def run_ogd_profile(x, y, T_warm, alpha=0.997, rho_ref=0.1, eps0=0.01,
                    refresh_every=50, eta0=0.05, s_init=0.5, G_init=None):
    """D-dimensional projected OGD on the pull profile s in [0,1]^D.

    The full Proposition 4(i) arm: one coordinate per eigen-direction, so
    the feasible set contains every fixed scalar strength and every fixed
    per-direction profile. Same component streams and anchor refresh as
    the scalar segment arm (run_ogd_spar); the anytime step carries the
    sqrt(D) dimension factor of the Proposition 4(i) bound:

        eta_t = eta0 * sqrt(D) / sqrt(t + 1).

    Per step the prediction interpolates the ridge data solve and the
    anchor per eigen-direction (the pull-identity family), the exact
    coordinate gradient is 2 * err * z_i * (G_e - w_hat_e)_i with
    z = U^T x_t, and each coordinate is projected back onto [0, 1].
    G_init selects the anchor prior (None = the warm-up fit).
    """
    D = x.shape[1]
    A = x[:T_warm].T @ x[:T_warm].astype(np.float64)
    B = x[:T_warm].T @ y[:T_warm]
    n_eff = float(T_warm)
    G = np.asarray(G_init, dtype=np.float64).copy() if G_init is not None \
        else np.linalg.solve(A + n_eff * 0.01 * np.eye(D), B)

    mu, U = np.linalg.eigh(A / n_eff)
    s = np.full(D, float(s_init))
    eta_scale = eta0 * np.sqrt(D)
    preds = np.zeros(len(y) - T_warm)
    step = 0

    for t in range(T_warm, len(y)):
        w_hat = np.linalg.solve(A + n_eff * eps0 * np.eye(D), B)
        w_hat_e = U.T @ w_hat
        G_e = U.T @ G
        W = U @ ((1.0 - s) * w_hat_e + s * G_e)
        preds[t - T_warm] = x[t] @ W

        err = preds[t - T_warm] - y[t]
        z = U.T @ x[t]
        grad = 2.0 * err * z * (G_e - w_hat_e)
        eta = eta_scale / np.sqrt(step + 1.0)
        s = np.clip(s - eta * grad, 0.0, 1.0)

        if step % refresh_every == 0 and step > 0:
            mu, U = np.linalg.eigh(A / n_eff)
            beta = mu / (mu + rho_ref)
            Gp = U.T @ G
            Wp = U.T @ W
            G = U @ (beta * Wp + (1 - beta) * Gp)

        A = alpha * A + np.outer(x[t], x[t])
        B = alpha * B + x[t] * y[t]
        n_eff = alpha * n_eff + 1.0
        step += 1

    return preds, s


def main() -> None:
    t0 = time.perf_counter()
    T_warm = 2000
    n_seeds = 5

    for decay in (3.0, 0.0):
        print(f"\n{'='*74}")
        print(f"=== OGD-s regret validation | spectrum decay={decay} ===")
        print(f"{'='*74}")

        all_res = {}
        s_trajs = {k: [] for k in ("pre", "drift", "post")}
        regret_pre = []
        for seed in range(n_seeds):
            x, y, w_true, t1, t2, _, _ = make_stream(
                seed=seed, D=32, noise=0.1, spectrum_decay=decay)
            y_te = y[T_warm:]
            regions = {
                "pre": slice(0, t1 - T_warm),
                "drift": slice(t1 - T_warm, t2 - T_warm),
                "post": slice(t2 - T_warm, None),
            }
            T_total = len(y_te)

            # OGD-s
            preds_ogd, s_traj = run_ogd_spar(x, y, T_warm, alpha=0.997)
            # fixed-s comparison class (fine grid)
            grid = {f"s={sv:.2f}": run_segment_fixed(x, y, T_warm, sv, alpha=0.997)
                    for sv in np.linspace(0.0, 1.0, 21)}
            # references
            leaky = run_leaky_rls(x, y, T_warm, 0.996)
            spar_full = run_spar_v2(x, y, T_warm, rho_init=0.1, alpha=0.997,
                                    adaptive_rho=True, refresh=True)[0]

            methods = {"OGD_s": preds_ogd, "leaky.996": leaky,
                       "spar_full": spar_full, **grid}

            # best fixed s in hindsight (full stream and per phase)
            names = list(methods)
            mse = {nm: np.array([float(np.mean((methods[nm][reg] - y_te[reg])**2))
                                 for reg in regions.values()])
                   for nm in names}
            overall = {nm: float(np.mean((methods[nm] - y_te)**2)) for nm in names}
            s_names = [nm for nm in names if nm.startswith("s=")]
            best_fixed_total = min(s_names, key=lambda nm: overall[nm])
            # OGD vs best fixed s: the regret
            regret = overall["OGD_s"] - overall[best_fixed_total]
            regret_pre.append(regret)

            for nm in names:
                for rn, reg in regions.items():
                    all_results = all_res.setdefault(nm, {})
                    all_results.setdefault(rn, []).append(
                        float(np.mean((methods[nm][reg] - y_te[reg])**2)))

            for rn, reg in regions.items():
                s_trajs[rn].append(float(np.mean(s_traj[reg])))

        def agg(nm, rn):
            return float(np.mean(all_res[nm][rn]))

        print(f"\n  regret (OGD_s overall - best fixed s overall), mean over seeds:"
              f" {np.mean(regret_pre):.5f}  (Prop 4 predicts O(sqrt(T)/T) -> small)")
        print(f"\n  {'method':<12} {'pre':>10} {'drift':>10} {'post':>10} {'overall':>10}")
        print("  " + "-" * 56)
        for nm in ("s=0.00", "s=0.25", "s=0.50", "s=0.75", "s=1.00",
                   "OGD_s", "leaky.996", "spar_full"):
            v = [agg(nm, rn) for rn in ("pre", "drift", "post")]
            print(f"  {nm:<12} {v[0]:>10.6f} {v[1]:>10.6f} {v[2]:>10.6f} "
                  f"{(v[0]+v[1]+v[2])/3:>10.6f}")

        best_s = min((nm for nm in all_res if nm.startswith("s=")),
                     key=lambda nm: sum(agg(nm, rn) for rn in ("pre", "drift", "post")))
        print(f"\n  best fixed s overall: {best_s}")
        print(f"  s_t trajectory: pre={np.mean(s_trajs['pre']):.4f}"
              f" drift={np.mean(s_trajs['drift']):.4f}"
              f" post={np.mean(s_trajs['post']):.4f}")
        print(f"  sign check: s(drift) < s(pre): "
              f"{np.mean(s_trajs['drift']) < np.mean(s_trajs['pre'])}")

    print(f"\ntotal {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
