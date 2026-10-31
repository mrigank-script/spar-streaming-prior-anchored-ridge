"""Input-side closed-form alignment and principal-angle drift diagnostics."""

import numpy as np

from core.information_alignment import AffineInputStitcher
from core.metrics import pad_score


def align_windows(win_ref, win_cal, method: str = "covariance_matching"):
    """Fit the affine stitcher on calibration windows against a reference window set.

    Returns a callable that aligns new windows of shape (S, L, C).
    """
    n_channels = win_ref.shape[-1]
    stitcher = AffineInputStitcher(n_channels)
    stitcher.fit_from_calibration(
        win_cal.reshape(-1, n_channels),
        X_ref=win_ref.reshape(-1, n_channels),
        method=method,
    )

    def apply(win):
        shape = win.shape
        return stitcher.align(win.reshape(-1, n_channels)).reshape(shape).astype(win.dtype)

    return apply, stitcher


def feature_subspace(phi: np.ndarray, rank: int = 8) -> np.ndarray:
    """Basis of the dominant feature-space subspace (right singular vectors)."""
    _, _, vt = np.linalg.svd(np.asarray(phi), full_matrices=False)
    return vt[:rank].T


def cov_subspace(phi: np.ndarray, rank: int = 8) -> np.ndarray:
    """Top-`rank` eigenvectors of the feature covariance."""
    cov = np.cov(np.asarray(phi), rowvar=False)
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, -rank:]


def pad_blocks(phi_ref: np.ndarray, phi_stream: np.ndarray, block: int = 100, rank: int = 8) -> np.ndarray:
    """Per-block PAD (energy) between the reference subspace and stream blocks."""
    u_ref = feature_subspace(phi_ref, rank)
    scores = []
    for start in range(0, len(phi_stream) - block + 1, block):
        u_block = feature_subspace(phi_stream[start:start + block], rank)
        scores.append(pad_score(u_ref, u_block, mode="energy"))
    return np.asarray(scores)


def pad_blocks_cov(phi_ref: np.ndarray, phi_stream: np.ndarray, block: int = 100, rank: int = 8) -> np.ndarray:
    """Per-block PAD (energy) between top covariance eigenspaces."""
    u_ref = cov_subspace(phi_ref, rank)
    scores = []
    for start in range(0, len(phi_stream) - block + 1, block):
        u_block = cov_subspace(phi_stream[start:start + block], rank)
        scores.append(pad_score(u_ref, u_block, mode="energy"))
    return np.asarray(scores)


def moment_drift_blocks(phi_ref: np.ndarray, phi_stream: np.ndarray, block: int = 100):
    """Per-block mean drift (normalized by reference feature RMS) and relative
    covariance drift versus reference moments."""
    phi_ref = np.asarray(phi_ref)
    phi_stream = np.asarray(phi_stream)
    mu_ref = phi_ref.mean(0)
    cov_ref = np.cov(phi_ref, rowvar=False)
    cov_scale = np.linalg.norm(cov_ref, "fro") + 1e-12
    mu_scale = float(np.sqrt(np.mean(np.var(phi_ref, axis=0)))) + 1e-12
    mu_scores, cov_scores = [], []
    for start in range(0, len(phi_stream) - block + 1, block):
        blk = phi_stream[start:start + block]
        mu_scores.append(np.linalg.norm(blk.mean(0) - mu_ref) / mu_scale)
        cov_scores.append(np.linalg.norm(np.cov(blk, rowvar=False) - cov_ref, "fro") / cov_scale)
    return np.asarray(mu_scores), np.asarray(cov_scores)


def trigger_metrics(scores: np.ndarray, onset_block: int, k: float = 3.0):
    """Threshold k-sigma above the pre-onset mean: (threshold, FAR, delay in blocks)."""
    scores = np.asarray(scores)
    pre = scores[:onset_block]
    threshold = pre.mean() + k * pre.std()
    alarms = scores > threshold
    far = float(alarms[:onset_block].mean())
    post = np.where(alarms[onset_block:])[0]
    delay = float(post[0]) if len(post) else float("inf")
    return float(threshold), far, delay
