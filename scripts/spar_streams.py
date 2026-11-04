"""Stream generator and prior constructors for the SPAR experiments.

Split out of ``rls_baselines``: data generation (``make_stream_fac``) and the
prior constructors (``make_prior``) only; no method code.

Sanity check: ``python scripts/rls_baselines.py`` exercises every axis.
"""

import numpy as np


def make_stream_fac(seed=0, D=128, T=10000, sigma=0.1,
                    spectrum="iso", drift="abrupt", noise="gauss",
                    drift_frac=(1 / 3, 2 / 3), mismatch="none"):
    """Multi-factor three-phase stream with optional model mismatch.

    mismatch: none | nonlinear (y = tanh(x·w) + ε, linear head misspecified)
              | covshift (feature scale doubles mid-stream; covariance x4)
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

