"""Live drift-trigger loop: composite PAD + moment drift -> closed-form re-stitch."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from core import features as featuremod
from core.metrics import pad_score
from dataio import splits
from runner.align import align_windows, cov_subspace


class TriggerConfig(NamedTuple):
    block: int = 100
    cal_len: int = 500
    k_sigma: float = 3.0
    use_pad: bool = True
    use_moment: bool = True
    rank: int = 8
    method: str = "diagonal"
    max_stitches: int = 3


def _ridge(phi, y, ridge, gamma, prior):
    D = phi.shape[1]
    A = phi.T @ phi / len(phi) + (gamma + ridge) * np.eye(D)
    b = phi.T @ y / len(phi) + gamma * prior
    return np.linalg.solve(A, b)


def run_live(values, train_end, feat_cfg, trig: TriggerConfig, lru_params, lookback: int = 24,
             ridge: float = 0.01, gamma: float = 0.1, oracle_onset_step=None):
    """Stream with composite drift detection; on trigger, collect a calibration
    window from the post-onset region, then closed-form re-stitch.

    oracle_onset_step: if given, trigger exactly once at that step (oracle).
    """
    values = np.asarray(values, dtype=np.float64)
    y_all = values[:, -1]
    ctx_tr, tgt_tr = splits.stream_arrays(train_end, lookback=lookback, start=lookback)
    win_tr = values[ctx_tr]
    y_tr = y_all[tgt_tr]

    feature_fn = featuremod.make_feature_fn(feat_cfg, lru_params)
    phi_tr = np.asarray(jax.vmap(feature_fn)(jnp.asarray(win_tr)).block_until_ready())
    mu = phi_tr.mean(0)
    sd = phi_tr.std(0) + 1e-8
    phi_tr_z = (phi_tr - mu) / sd
    D = phi_tr_z.shape[1]
    W = _ridge(phi_tr_z, y_tr, ridge, gamma, np.zeros(D))

    def state_fn(win):
        s = featuremod.lru_last_state(win.astype(jnp.float32), lru_params["lambda_diag"], lru_params["B"])
        return jnp.concatenate([jnp.real(s), jnp.imag(s)])

    state_jit = jax.jit(state_fn)
    st_tr_raw = np.asarray(jax.vmap(state_jit)(jnp.asarray(win_tr)).block_until_ready())
    st_mu = st_tr_raw.mean(0)
    st_sd = st_tr_raw.std(0) + 1e-8
    st_tr = (st_tr_raw - st_mu) / st_sd
    cov_ref = np.cov(st_tr, rowvar=False)
    cov_scale = np.linalg.norm(cov_ref, "fro") + 1e-12
    u_st = cov_subspace(st_tr, trig.rank)

    def scores_of(st_block_z):
        pad_s = pad_score(u_st, cov_subspace(st_block_z, trig.rank), "energy")
        mom_s = float(np.linalg.norm(st_block_z.mean(0)))
        cov_s = float(np.linalg.norm(np.cov(st_block_z, rowvar=False) - cov_ref, "fro") / cov_scale)
        return pad_s, mom_s, cov_s

    ref = [scores_of(st_tr[s0:s0 + trig.block])
           for s0 in range(0, len(st_tr) - trig.block + 1, trig.block)]
    ref_pad = np.array([r[0] for r in ref])
    ref_mom = np.array([r[1] for r in ref])
    thr_pad = float(ref_pad.mean() + trig.k_sigma * ref_pad.std())
    thr_mom = float(ref_mom.mean() + trig.k_sigma * ref_mom.std())

    preds = np.full(len(values), np.nan)
    triggers = []
    apply_align = None
    pending_start = None
    cooldown_until = -1
    n_stitched = 0
    for s0 in range(train_end, len(values), trig.block):
        s1 = min(s0 + trig.block, len(values))
        ctx, tgt = splits.stream_arrays(s1, lookback=lookback, start=max(s0, lookback))
        if len(tgt) == 0:
            continue
        win_raw = values[ctx]
        win = apply_align(win_raw) if apply_align is not None else win_raw
        phi = np.asarray(jax.vmap(feature_fn)(jnp.asarray(win)).block_until_ready())
        preds[tgt] = ((phi - mu) / sd) @ W

        if pending_start is not None:
            if s1 - pending_start >= trig.cal_len:
                cctx, _ = splits.stream_arrays(s1, lookback=lookback, start=pending_start)
                apply_align, _ = align_windows(win_tr, values[cctx], method=trig.method)
                triggers.append({"start": int(s0), "end": int(s1), "pad": None, "mom": None})
                cooldown_until = s1 + trig.cal_len
                pending_start = None
                n_stitched += 1
            continue

        st_b = (np.asarray(jax.vmap(state_jit)(jnp.asarray(win)).block_until_ready()) - st_mu) / st_sd
        pad_s, mom_s, _ = scores_of(st_b)
        if oracle_onset_step is not None:
            fire = bool(s0 <= oracle_onset_step < s1) and not triggers
        else:
            fire = ((trig.use_pad and pad_s > thr_pad) or (trig.use_moment and mom_s > thr_mom)) \
                and s0 >= cooldown_until and n_stitched < trig.max_stitches
        if fire:
            pending_start = s1
            triggers.append({"start": int(s0), "end": int(s1), "pad": float(pad_s), "mom": float(mom_s)})

    return {
        "preds": preds,
        "targets": y_all,
        "triggers": triggers,
        "thresholds": (thr_pad, thr_mom),
        "train_end": train_end,
    }
