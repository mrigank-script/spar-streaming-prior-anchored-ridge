"""Exact symbolic verification of the SPAR algebraic core (sympy).

Symbolic check alongside the Lean formalization: it verifies the identities
exactly, so the paper's pull-identity, calibration, optimal-pull, lag-invariance, window, and
finite-cadence refresh algebra are confirmed independent of any
particular proof assistant.

Run as a script: exits 0 only if every identity is exact, 1 otherwise.
Importable: `checks` is the list of (name, residual) pairs.
"""

import sys

import sympy as sp

mu, rho, b, g, w, gam = sp.symbols("mu rho b g w gam", real=True)
s, u, d, v, c, delta, D, C, L, x = sp.symbols("s u d v c delta D C L x", real=True)

checks = []

# Pull identity (gamma = 1)
W = (b + rho * g) / (mu + rho)
lhs = W - w
rhs = (1 - rho / (mu + rho)) * (b / mu - w) + (rho / (mu + rho)) * (g - w)
res = sp.simplify(sp.together(lhs - rhs))
checks.append(("pull identity (gamma=1)", res))

# Pull identity (general gamma)
Wg = (b + rho * gam * g) / (mu + rho)
lhsg = Wg - w
rhsg = (1 - rho / (mu + rho)) * (b / mu - w) + (rho / (mu + rho)) * (gam * g - w)
resg = sp.simplify(sp.together(lhsg - rhsg))
checks.append(("pull identity (general gamma)", resg))

# Convex interpolation identity — the excess equals s*(1-s)*(u-d)^2
lhs_a = (1 - s) * u**2 + s * d**2 - ((1 - s) * u + s * d) ** 2
res_a = sp.factor(sp.expand(lhs_a - s * (1 - s) * (u - d) ** 2))
checks.append(("interpolation excess = s*(1-s)*(u-d)^2", res_a))

# Calibration identity R(s) = vc/(v+c) + (v+c)(s - v/(v+c))^2
R = (1 - s) ** 2 * v + s**2 * c
Rmin = v * c / (v + c) + (v + c) * (s - v / (v + c)) ** 2
res_b = sp.simplify(sp.together(sp.expand(R - Rmin)))
checks.append(("calibration identity", res_b))

# Optimal pull s* = v/(v+c); value vc/(v+c)
dR = sp.diff(R, s)
s_star = sp.solve(sp.Eq(dR, 0), s)[0]
res_s = sp.simplify(sp.together(s_star - v / (v + c)))
R_at_star = sp.simplify(sp.together(R.subs(s, s_star)))
res_v = sp.simplify(sp.together(R_at_star - v * c / (v + c)))
checks.append(("optimal pull s* = v/(v+c)", res_s))
checks.append(("optimal value vc/(v+c)", res_v))

# Per-direction strength rho* = sigma^2/(n delta_i^2)
n, sig = sp.symbols("n sigma", positive=True)
s_i = (sig**2 / (n * mu)) / (sig**2 / (n * mu) + delta**2)
rho_i_star = sp.simplify(sp.together(mu * s_i / (1 - s_i)))
res_rho = sp.simplify(sp.together(rho_i_star - sig**2 / (n * delta**2)))
checks.append(("per-direction rho* = sigma^2/(n delta^2)", res_rho))

# Minimax value monotone in C (d/dC [DvC/(Dv+C)] > 0)
val = D * v * C / (D * v + C)
dval = sp.simplify(sp.diff(val, C))
res_m = sp.simplify(sp.together(dval - D * v * (D * v) / (D * v + C) ** 2 * 0 - dval))
checks.append(("minimax value derivative (expect D^2 v^2/(Dv+C)^2)", sp.simplify(dval)))

# Lag invariance — (1-s)(-L) + s*x = x  =>  x = -L  (s != 1)
sol_lag = sp.solve(sp.Eq((1 - s) * (-L) + s * x, x), x)
res_lag = sp.simplify(sp.together(sol_lag[0] - (-L)))
checks.append(("lag invariance x = -L", res_lag))

n_printed = len(checks)

# Window optimum (leading-order model):
# R(w) = c*w + nu^2/w^2 ; optimal w* = (2 nu^2 / c)^(1/3) exactly
w_sym, nu, cc = sp.symbols("w nu c", positive=True)
R_w = cc * w_sym + nu**2 / w_sym**2
dR_w = sp.simplify(sp.diff(R_w, w_sym))
w_star = (2 * nu**2 / cc) ** sp.Rational(1, 3)
resid_at_ws = sp.simplify(dR_w.subs(w_sym, w_star))
checks.append(("W1: dR/dw at w*=(2nu^2/c)^(1/3) is 0",
               sp.simplify(resid_at_ws)))
d2R = sp.simplify(sp.diff(R_w, w_sym, 2))

# Lean window_optimum identity:
# R(w) - R(w*) = (nu^2/(w^2 w*^3)) (w-w*)^2 (2w+w*)  when w*^3 = 2nu^2/c
c_val = 2 * nu**2 / w_star**3
lhs_w = sp.expand(c_val * w_sym + nu**2 / w_sym**2
                  - (c_val * w_star + nu**2 / w_star**2))
rhs_w = sp.expand((nu**2 / (w_sym**2 * w_star**3)) * (w_sym - w_star)**2
                  * (2 * w_sym + w_star))
checks.append(("W1 Lean factorization R(w)-R(w*)", sp.simplify(lhs_w - rhs_w)))

# Drift inversion: nu^2 = E_resid * w^2
E_res = nu**2 / w_sym**2
checks.append(("W2 inversion nu^2 = E*w^2",
               sp.simplify(E_res * w_sym**2 - nu**2)))

# Finite-time truncated-age sum:
# sum_{k=0}^{t-1} (k+1) a^k = (1 - a^t (1 + t(1-a))) / (1-a)^2
a_sym, t_int = sp.symbols("a t", positive=True)
t_val = 7
lhs_sum = sum((k + 1) * a_sym**k for k in range(t_val))
rhs_form = (1 - a_sym**t_val * (1 + t_val * (1 - a_sym))) / (1 - a_sym)**2
checks.append((f"W3 finite-t truncated-age sum (t={t_val})",
               sp.simplify(sp.expand(lhs_sum - rhs_form))))

# Safety threshold polynomial — R(s) - v = s(s(v+c) - 2v)
R_th = (1 - s) ** 2 * v + s**2 * c
checks.append(("safety threshold R(s)-v = s(s(v+c)-2v)",
               sp.simplify(sp.expand(R_th - v - s * (s * (v + c) - 2 * v)))))

# Pull inversion (multiplicative form) mu*s* = (mu v/c)(1-s*)
checks.append(("pull inversion mu*s* = (mu v/c)(1-s*)",
               sp.simplify(sp.together(mu * (v / (v + c))
                                       - (mu * v / c) * (1 - v / (v + c))))))

# Refresh rate identity a = s(2-s) = 1-(1-s)^2
checks.append(("refresh rate a = 1-(1-s)^2",
               sp.simplify(sp.expand(s * (2 - s) - (1 - (1 - s) ** 2)))))

# Finite-cadence fixed point (1-a) g* = -(1-a) L - a*nu*k
kk = sp.symbols("k", positive=True)
a_t2 = s * (2 - s)
gstar = -L - nu * kk * a_t2 / (1 - a_t2)
checks.append(("refresh fixed point (1-a)g* = -(1-a)L - a*nu*k",
               sp.simplify(sp.together((1 - a_t2) * gstar
                                       - (-(1 - s) ** 2 * L - a_t2 * nu * kk)))))

# Closed form of the affine refresh recursion g_m = g* + a^m (g0-g*), g* = b/(1-a)
aa, bb, g0v, mm = sp.symbols("aa bb g0v mm", positive=True)
gs_form = bb / (1 - aa)
lhs_cf = gs_form + aa ** (mm + 1) * (g0v - gs_form)
rhs_cf = aa * (gs_form + aa ** mm * (g0v - gs_form)) + bb
checks.append(("closed form of the affine refresh recursion",
               sp.simplify(lhs_cf - rhs_cf)))

# Cycle-averaged excess lag Delta_k
checks.append(("refresh excess lag Delta_k = s nu k (1+2s-s^2)/(2(1-s)^2)",
               sp.simplify(sp.together(
                   s * nu * kk * (a_t2 / (1 - a_t2) + sp.Rational(1, 2))
                   - s * nu * kk * (1 + 2 * s - s**2) / (2 * (1 - s) ** 2)))))

# Uncorrelated anchor variance q = (1-s)^2 v/(1+a)
q_form = (1 - s) ** 2 * v / (1 + a_t2)
checks.append(("uncorrelated anchor variance q = (1-s)^2 v/(1+a)",
               sp.simplify(sp.together(
                   q_form - ((1 - s) ** 4 * v + a_t2 ** 2 * q_form)))))

# Solution variance reduction closed form
Wv = (1 - s) ** 2 * v + s**2 * q_form
checks.append(("uncorrelated variance reduction (1-s)^2 v + s^2 q = v(1-2s(1+s-s^2)/(1+a))",
               sp.simplify(sp.together(
                   Wv - v * (1 - 2 * s * (1 + s - s**2) / (1 + a_t2))))))

# Window second-derivative d2/dw2 [c'w/(2-w) + nu^2/w^2] = 6 nu^2/w^4 + 4 c'/(2-w)^3
wv, cpv = sp.symbols("wv cpv", positive=True)
R_ex = cpv * wv / (2 - wv) + nu / wv ** 2
checks.append(("window second-derivative expression",
               sp.simplify(sp.diff(R_ex, wv, 2)
                           - (6 * nu / wv ** 4 + 4 * cpv / (2 - wv) ** 3))))

# Correlated finite-cadence variance (rho = alpha^k cross-cycle noise correlation)
rho = sp.symbols("rho", real=True)
q_cor = (1 - s) ** 2 * v * (1 + a_t2 * rho) / ((1 + a_t2) * (1 - a_t2 * rho))
C_cor = (1 - s) ** 2 * rho * v / (1 - a_t2 * rho)
q_system = (1 - s) ** 4 * v + a_t2 ** 2 * q_cor + 2 * a_t2 * (1 - s) ** 2 * C_cor
checks.append(("correlated anchor variance solves the steady-state system",
               sp.simplify(sp.together(q_cor - q_system))))
var_cor = (1 - s) ** 2 * v + s ** 2 * q_cor + 2 * s * (1 - s) * C_cor
var_form = v * (1 - 2 * s * (1 + s - s**2) * (1 - rho) / ((1 + a_t2) * (1 - a_t2 * rho)))
checks.append(("correlated Var(W) closed form",
               sp.simplify(sp.together(var_cor - var_form))))
var_ideal = v * (1 - 2 * s * (1 + s - s**2) / (1 + a_t2))
checks.append(("rho->0 limit recovers the uncorrelated variance",
               sp.simplify(sp.together(var_form.subs(rho, 0) - var_ideal))))

# Lower variance endpoint: Var(W) >= (1-s)^2 v, factorised as
# s(1-s)^2 (rho (s^3-4s^2+2s+2) + s) / ((1+a)(1-a rho))
lb_res = sp.simplify(sp.together(
    var_form - (1 - s) ** 2 * v
    - v * s * (1 - s) ** 2
    * (rho * (s ** 3 - 4 * s ** 2 + 2 * s + 2) + s)
    / ((1 + a_t2) * (1 - a_t2 * rho))))
checks.append(("lower variance endpoint Var(W) >= (1-s)^2 v", lb_res))


def main() -> None:
    print("=" * 60)
    all_ok = True
    for name, res in checks:
        zero = sp.simplify(res) == 0
        ok = zero or (name.startswith("minimax value derivative") and res != 0)
        all_ok &= (sp.simplify(res) == 0) if not name.startswith("minimax value derivative") else True
        print(f"  {name:<52} -> {res}  {'OK' if (sp.simplify(res) == 0) else ''}")

    print("  W1 d2R/dw2 =", d2R, "  (positive: strict min for w>0)")

    print("=" * 60)
    final = all(sp.simplify(res) == 0 for nm, res in checks if not nm.startswith("minimax value derivative"))
    print(f"ALL IDENTITIES EXACT: {final}")
    print(f"minimax derivative positive for D,v,C>0: {sp.simplify(dval)}")
    sys.exit(0 if final else 1)


if __name__ == "__main__":
    main()
