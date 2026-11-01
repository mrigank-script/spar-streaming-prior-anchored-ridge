"""Online-EA causality: the whitener for
stream trial i is built only from trials < i, so perturbing future trials
must leave earlier trials' features bit-identical."""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("pyriemann")
pytest.importorskip("mne")
tier1 = pytest.importorskip("scripts.bench_eeg_tier1")


def _feature_fn():
    import jax.numpy as jnp

    def fn(x):
        return jnp.concatenate([jnp.mean(x, axis=0), jnp.std(x, axis=0)])

    return fn


def test_online_ea_features_inert_to_future_trials():
    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(60, 4, 16))
    X_st = rng.normal(size=(50, 4, 16))
    window, every = 20, 10

    phi1, feats1 = tier1.online_ea_features(
        X_tr, X_st, _feature_fn(), window=window, every=every)

    X_st2 = X_st.copy()
    X_st2[30:] += 5.0  # perturb only future trials
    phi2, feats2 = tier1.online_ea_features(
        X_tr, X_st2, _feature_fn(), window=window, every=every)

    np.testing.assert_array_equal(phi1, phi2)
    np.testing.assert_array_equal(feats1[:30], feats2[:30])
    # the perturbation must actually reach the future features (non-vacuous)
    assert not np.array_equal(feats1[30:], feats2[30:])
