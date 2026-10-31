"""Evaluation metrics for streaming, drift, and closed-form learning experiments.

Analysis-time utilities (numpy, not jax): a run records prediction/error streams,
tracked subspaces, and weight snapshots, and these functions score them.

Metrics introduced for this thesis:
  PAD  principal-angle drift (drift detection and trigger quality)
  BAS  backward alignment score (compressibility of backprop into block maps)
  EMH  effective memory horizon of a forgetting factor
  SR   spectral reliability (per-block conditioning as confidence)
  SPR  signature preservation ratio (Woodbury core stability)
"""

import numpy as np

__all__ = [
    "effective_memory_horizon",
    "principal_angles",
    "pad_score",
    "pad_trigger",
    "recovery_time",
    "adaptation_regret",
    "parameter_tracking_error",
    "spectral_reliability",
    "eig_signature",
    "signature_preservation_ratio",
    "block_diagonal_of",
    "embed_block_diagonal",
    "backward_alignment_score",
    "mse",
    "mae",
    "mase",
    "rank_ic",
    "sharpe_ratio",
    "max_drawdown",
    "mean_turnover",
    "net_returns",
    "balanced_accuracy",
    "cohen_kappa",
    "expected_calibration_error",
]


# --------------------------------------------------------------------------
# Drift detection and adaptation speed
# --------------------------------------------------------------------------

def effective_memory_horizon(alpha: float) -> float:
    """EMH = 1/(1-alpha): expected lookback in samples of forgetting factor alpha."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]; alpha=1 means infinite memory")
    if alpha == 1.0:
        return float("inf")
    return 1.0 / (1.0 - alpha)


def principal_angles(U_ref: np.ndarray, U_t: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between the column spaces of U_ref and U_t."""
    Q1, _ = np.linalg.qr(U_ref)
    Q2, _ = np.linalg.qr(U_t)
    sv = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    sv = np.clip(sv, -1.0, 1.0)
    return np.arccos(sv)


def pad_score(U_ref: np.ndarray, U_t: np.ndarray, mode: str = "energy") -> float:
    """Principal-Angle Drift statistic.

    mode: 'max' -> largest sine (worst rotated direction);
          'mean' -> mean sine;
          'energy' -> sum of squared sines (Grassmann projection distance).
    """
    sines = np.sin(principal_angles(U_ref, U_t))
    if mode == "max":
        return float(sines[-1])
    if mode == "mean":
        return float(np.mean(sines))
    if mode == "energy":
        return float(np.sum(sines ** 2))
    raise ValueError(f"unknown mode: {mode}")


def pad_trigger(scores: np.ndarray, tau_max: float | None = None, tau_energy: float | None = None) -> np.ndarray:
    """Boolean firing mask for a PAD score stream; fires on either exceeded threshold."""
    scores = np.asarray(scores, dtype=float)
    if tau_max is None and tau_energy is None:
        raise ValueError("set at least one threshold")
    fire = np.zeros(scores.shape, dtype=bool)
    if tau_max is not None:
        fire |= scores > tau_max
    if tau_energy is not None:
        fire |= scores > tau_energy
    return fire


def recovery_time(signal: np.ndarray, drift_index: int, baseline: float | None = None,
                  tol: float = 0.1, patience: int = 5) -> float:
    """Samples after drift onset until signal returns within tol*|baseline| and stays.

    Returns inf if the signal never recovers within the stream.
    """
    signal = np.asarray(signal, dtype=float)
    if baseline is None:
        baseline = float(np.median(signal[:drift_index]))
    if drift_index >= len(signal):
        return float("inf")
    band = abs(baseline) * tol + 1e-12
    hit = 0
    for i in range(drift_index, len(signal)):
        if abs(signal[i] - baseline) <= band:
            hit += 1
            if hit >= patience:
                return float(i - patience + 1 - drift_index)
        else:
            hit = 0
    return float("inf")


def adaptation_regret(errors: np.ndarray, oracle_errors: np.ndarray) -> float:
    """Cumulative excess loss of a stream versus a per-step oracle."""
    errors = np.asarray(errors, dtype=float)
    oracle = np.asarray(oracle_errors, dtype=float)
    return float(np.sum(errors - oracle))


def parameter_tracking_error(W_est: np.ndarray, W_true: np.ndarray) -> float:
    """Mean Frobenius distance between estimated and true weight streams (T, ...)."""
    W_est = np.asarray(W_est, dtype=float)
    W_true = np.asarray(W_true, dtype=float)
    diff = (W_est - W_true).reshape(len(W_est), -1)
    return float(np.mean(np.linalg.norm(diff, axis=1)))


# --------------------------------------------------------------------------
# Conditioning and numerical stability
# --------------------------------------------------------------------------

def spectral_reliability(A: np.ndarray) -> np.ndarray:
    """SR = sigma_min / sigma_max per matrix (1/condition number), supports stacked input."""
    sv = np.linalg.svd(A, compute_uv=False)
    return sv[..., -1] / (sv[..., 0] + 1e-12)


def eig_signature(M: np.ndarray) -> tuple[int, int]:
    """(n_positive, n_negative) eigenvalue counts of a symmetric matrix."""
    w = np.linalg.eigvalsh(M)
    return int(np.sum(w > 1e-8)), int(np.sum(w < -1e-8))


def signature_preservation_ratio(M_before: np.ndarray, M_after: np.ndarray) -> float:
    """1.0 if the eigenvalue-sign counts (p, q) of the two symmetric matrices agree."""
    w1 = np.linalg.eigvalsh(M_before)
    w2 = np.linalg.eigvalsh(M_after)
    p1, q1 = int(np.sum(w1 > 1e-8)), int(np.sum(w1 < -1e-8))
    p2, q2 = int(np.sum(w2 > 1e-8)), int(np.sum(w2 < -1e-8))
    return 1.0 if (p1 == p2 and q1 == q2) else 0.0


# --------------------------------------------------------------------------
# Backward-map analysis
# --------------------------------------------------------------------------

def block_diagonal_of(W: np.ndarray, block_size: int) -> np.ndarray:
    """Extract diagonal blocks from a square matrix -> (num_blocks, bs, bs)."""
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be square")
    D = W.shape[0]
    if D % block_size != 0:
        raise ValueError("D must be divisible by block_size")
    nb = D // block_size
    return np.stack([W[i * block_size:(i + 1) * block_size,
                       i * block_size:(i + 1) * block_size] for i in range(nb)])


def embed_block_diagonal(blocks: np.ndarray) -> np.ndarray:
    """Scatter (num_blocks, bs_in, bs_out) blocks into a full block-diagonal matrix."""
    nb, b_in, b_out = blocks.shape
    W = np.zeros((nb * b_in, nb * b_out), dtype=blocks.dtype)
    for i in range(nb):
        W[i * b_in:(i + 1) * b_in, i * b_out:(i + 1) * b_out] = blocks[i]
    return W


def backward_alignment_score(G_ref: np.ndarray, G_approx: np.ndarray, mode: str = "relative") -> float:
    """BAS: how faithfully an approximation reproduces the reference map (1 = perfect)."""
    G_ref = np.asarray(G_ref, dtype=float)
    G_approx = np.asarray(G_approx, dtype=float)
    if mode == "relative":
        denom = np.linalg.norm(G_ref) + 1e-12
        return float(1.0 - np.linalg.norm(G_ref - G_approx) / denom)
    if mode == "cosine":
        a = G_ref.ravel()
        b = G_approx.ravel()
        return float(a @ b / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))
    raise ValueError(f"unknown mode: {mode}")


# --------------------------------------------------------------------------
# Forecast quality
# --------------------------------------------------------------------------

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray) -> float:
    """MAE scaled by the MAE of a seasonal-naive forecast (scale = mean |y - y_naive|)."""
    scale = np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_naive, dtype=float)))
    return mae(y_true, y_pred) / (scale + 1e-12)


# --------------------------------------------------------------------------
# Finance
# --------------------------------------------------------------------------

def _rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def rank_ic(pred: np.ndarray, target: np.ndarray) -> float:
    """Spearman rank information coefficient."""
    return float(np.corrcoef(_rankdata(pred), _rankdata(target))[0, 1])


def sharpe_ratio(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    if sd == 0.0:
        return 0.0
    return float(np.mean(r) / sd * np.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    wealth = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peak - 1.0))


def mean_turnover(weights: np.ndarray, prev_weights: np.ndarray) -> float:
    """Mean L1 change of the weight vector per period (two-sided)."""
    weights = np.asarray(weights, dtype=float)
    prev_weights = np.asarray(prev_weights, dtype=float)
    return float(np.mean(np.abs(weights - prev_weights).sum(axis=1)))


def net_returns(gross: np.ndarray, weights: np.ndarray, prev_weights: np.ndarray,
                cost_bps: float = 10.0) -> np.ndarray:
    """Gross period returns minus proportional transaction costs on L1 turnover."""
    gross = np.asarray(gross, dtype=float)
    weights = np.asarray(weights, dtype=float)
    prev_weights = np.asarray(prev_weights, dtype=float)
    turnover = np.abs(weights - prev_weights).sum(axis=1)
    return gross - (cost_bps / 1e4) * turnover


# --------------------------------------------------------------------------
# Classification (EEG / BCI)
# --------------------------------------------------------------------------

def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = [np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)]
    return float(np.mean(recalls))


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = np.unique(np.concatenate([y_true, y_pred]))
    index = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=float)
    for t, p in zip(y_true, y_pred):
        cm[index[t], index[p]] += 1.0
    n = cm.sum()
    po = np.trace(cm) / n
    pe = float((cm.sum(axis=0) * cm.sum(axis=1)).sum() / n ** 2)
    if pe >= 1.0:
        return 0.0
    return float((po - pe) / (1.0 - pe))


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """ECE over max-probability confidence bins; probs may be (N, K) or (N,)."""
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true).ravel()
    if probs.ndim == 2:
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
    else:
        conf = probs
        pred = (probs >= 0.5).astype(int)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)
