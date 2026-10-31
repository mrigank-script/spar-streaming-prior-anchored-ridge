"""Information Alignment (IA) augmenting objective.

Adds a soft coupling to the backward projection G to bound the drift of
local target propagation: B_IA = B_raw + γ·G while A_IA = A_raw + (γ+λ) I.
γ=0 recovers plain block ridge regression.
"""

from typing import Optional, Tuple
import jax.numpy as jnp
from jax import Array


def ia_regularise(
    A_raw: Array,
    B_raw: Array,
    G: Array | None = None,
    gamma: float = 0.01,
    lambda_reg: float = 0.01,
    use_transpose: bool = True,
    use_spectral: bool = False,
    use_exact_map: bool = False,
) -> tuple[Array, Array]:
    """Apply IA regularisation: A_IA = A_raw + (γ+λ) I, B_IA = B_raw + γ·G (or γ·G^T / γ·Q).

    Coefficient note (handoff §11.1): the exact MAP estimate under the prior
    W ~ N(G, (γ+λ)^{-1} I) weights the prior mean by (γ+λ), not γ:
    W_MAP = (A + (γ+λ)I)^{-1} (B + (γ+λ) G). The default path (γ only)
    under-weights the prior mean by (γ+λ)/γ (= 2 at the 0.01/0.01 defaults) and
    is kept for backward compatibility; pass use_exact_map=True for the exact
    MAP coefficient. gamma = 0 recovers plain block ridge in both versions."""
    bs = A_raw.shape[-1]
    I = jnp.eye(bs, dtype=A_raw.dtype)
    if A_raw.ndim == 2:
        I_blocks = I
    else:
        I_blocks = jnp.broadcast_to(I[None], A_raw.shape)

    A_IA = A_raw + (gamma + lambda_reg) * I_blocks
    prior_scale = (gamma + lambda_reg) if use_exact_map else gamma

    if use_spectral:
        # Polar spectral anchor (F-12 fix): Q = U V^T of B_raw guarantees
        # sigma_min(B_IA) >= gamma + sigma_min(B_raw) exactly (§11.2):
        # B + gamma U V^T = U (Sigma + gamma I) V^T. This branch must be
        # checked BEFORE `G is None`: Q derives from B_raw and needs no G.
        # (Previously `G is None` short-circuited and the anchor was silently
        # never applied whenever G was not passed.)
        u, _, vt = jnp.linalg.svd(B_raw, full_matrices=False)
        Q = u @ vt
        B_IA = B_raw + prior_scale * Q
    elif G is None:
        B_IA = B_raw
    elif use_transpose:
        # Adjoint coupling: B_IA = B_raw + prior_scale * G^T. Only meaningful
        # when the per-block coupling map is square (bs x bs target-propagation
        # case); for rectangular (bs, K) targets G is used untransposed.
        if G.ndim >= 2 and G.shape[-1] == G.shape[-2]:
            G_coupling = jnp.swapaxes(G, -1, -2)
        else:
            G_coupling = G
        B_IA = B_raw + prior_scale * G_coupling
    else:
        B_IA = B_raw + prior_scale * G

    return A_IA, B_IA


def check_positive_definite(A: Array, tol: float = 1e-6) -> Array:
    """Per-block positive-definiteness check via minimum eigenvalue."""
    eigvals = jnp.linalg.eigvalsh(A)
    return jnp.all(eigvals > tol, axis=-1)


def condition_numbers(A: Array) -> Array:
    """Per-block condition number (σ_max / σ_min) for monitoring (supports 2D and 3D arrays)."""
    sv = jnp.linalg.svd(A, compute_uv=False)
    return sv[..., 0] / (sv[..., -1] + 1e-12)


import numpy as np

class AffineInputStitcher:
    """
    Unsupervised Input-Side Affine Alignment Layer (F-04 Remediation).
    
    Transforms shifted test inputs before entering the frozen neural backbone:
        X_aligned = (X - mu_cal) @ W_align + mu_ref
    
    Addresses the nonlinear latent manifold rotation bottleneck where linear output
    readouts cannot un-saturate upstream frozen LSTM gates.
    """
    def __init__(self, num_channels: int, reg_eps: float = 1e-4):
        self.num_channels = num_channels
        self.reg_eps = reg_eps
        self.mu_ref = None
        self.Sigma_ref = None
        self.mu_cal = None
        self.Sigma_cal = None
        self.W_align = np.eye(num_channels, dtype=np.float32)
        self.b_align = np.zeros(num_channels, dtype=np.float32)
        self.is_fitted = False

    def set_reference_stats(self, mu_ref: np.ndarray, Sigma_ref: Optional[np.ndarray] = None):
        """Set Day 1 reference in-distribution mean and covariance."""
        self.mu_ref = np.asarray(mu_ref, dtype=np.float32)
        if Sigma_ref is not None:
            self.Sigma_ref = np.asarray(Sigma_ref, dtype=np.float32)
        else:
            self.Sigma_ref = np.eye(self.num_channels, dtype=np.float32)

    def fit_from_calibration(
        self,
        X_cal: np.ndarray,
        X_ref: Optional[np.ndarray] = None,
        method: str = "covariance_matching"
    ):
        """
        Fit affine stitcher strictly using calibration window data [0, n_cal].
        Methods:
          - 'covariance_matching': W = Sigma_cal^{-1/2} @ Sigma_ref^{1/2}
          - 'diagonal': Channel-wise mean centering and variance rescaling
        """
        X_cal = np.asarray(X_cal, dtype=np.float32)
        N, C = X_cal.shape
        self.mu_cal = np.mean(X_cal, axis=0)

        if X_ref is not None:
            self.mu_ref = np.mean(X_ref, axis=0)
            X_ref_c = X_ref - self.mu_ref
            self.Sigma_ref = (X_ref_c.T @ X_ref_c) / len(X_ref) + self.reg_eps * np.eye(C)
        elif self.mu_ref is None:
            self.mu_ref = np.zeros(C, dtype=np.float32)
            self.Sigma_ref = np.eye(C, dtype=np.float32)

        X_cal_c = X_cal - self.mu_cal
        self.Sigma_cal = (X_cal_c.T @ X_cal_c) / max(N, 1) + self.reg_eps * np.eye(C)

        if method == "covariance_matching" and N >= C:
            # Symmetric matrix square root and inverse square root
            u1, s1, _ = np.linalg.svd(self.Sigma_cal)
            Sigma_cal_inv_half = u1 @ np.diag(1.0 / np.sqrt(np.maximum(s1, self.reg_eps))) @ u1.T

            u0, s0, _ = np.linalg.svd(self.Sigma_ref)
            Sigma_ref_half = u0 @ np.diag(np.sqrt(np.maximum(s0, self.reg_eps))) @ u0.T

            self.W_align = (Sigma_cal_inv_half @ Sigma_ref_half).astype(np.float32)
        else:
            # Diagonal variance matching (robust when N < C)
            std_cal = np.sqrt(np.diag(self.Sigma_cal)) + 1e-6
            std_ref = np.sqrt(np.diag(self.Sigma_ref)) + 1e-6
            self.W_align = np.diag(std_ref / std_cal).astype(np.float32)

        self.b_align = self.mu_ref - self.mu_cal @ self.W_align
        self.is_fitted = True

    def align(self, X: np.ndarray) -> np.ndarray:
        """Apply input-side affine alignment to test features."""
        if not self.is_fitted:
            return X
        return (np.asarray(X, dtype=np.float32) @ self.W_align + self.b_align).astype(np.float32)