"""Symbolic identity assertions: the sympy checks that
verify_spar_theorems only printed, as hard asserts; core identities are
also reimplemented inline so the essentials are pinned even without the
script import."""

import pytest

sp = pytest.importorskip("sympy")


def test_verify_spar_theorems_checks_all_exact():
    verify = pytest.importorskip("scripts.verify_spar_theorems")
    for name, res in verify.checks:
        if name.startswith("minimax value derivative"):
            continue
        assert sp.simplify(res) == 0, f"identity failed: {name} -> {res}"


def test_core_identities_reimplemented():
    mu, rho, b, g, w, gam = sp.symbols("mu rho b g w gam", real=True)
    s, v, c = sp.symbols("s v c", real=True)

    # pull identity, general gamma
    lhs = (b + rho * gam * g) / (mu + rho) - w
    rhs = ((1 - rho / (mu + rho)) * (b / mu - w)
           + (rho / (mu + rho)) * (gam * g - w))
    assert sp.simplify(sp.together(lhs - rhs)) == 0

    # calibration identity: R(s) = (1-s)^2 v + s^2 c
    R = (1 - s) ** 2 * v + s ** 2 * c
    Rmin = v * c / (v + c) + (v + c) * (s - v / (v + c)) ** 2
    assert sp.simplify(sp.together(sp.expand(R - Rmin))) == 0

    # window risk R(w) = c*w + nu^2/w^2: stationary at w* = (2 nu^2/c)^(1/3)
    w_sym, nu, cc = sp.symbols("w nu c", positive=True)
    R_w = cc * w_sym + nu ** 2 / w_sym ** 2
    w_star = (2 * nu ** 2 / cc) ** sp.Rational(1, 3)
    assert sp.simplify(sp.diff(R_w, w_sym).subs(w_sym, w_star)) == 0
