"""Complex-diagonal Linear Recurrent Unit (LRU).

Processes a sequence in O(log L) parallel steps via JAX's associative scan.
The diagonal structure of Λ makes the scan element-wise.

Keep every |λ| < 1.0: a radius of 1 turns the recurrence into an undamped
integrator and blows up to NaN. Initialise with r ∈ [0.9, 0.999].
"""

import jax
import jax.numpy as jnp
from jax import Array


def lru_combine(carry_a, carry_b):
    """Associative combine for the parallel scan: (h', λ') = (λ_b h_a + h_b, λ_b λ_a)."""
    h_a, lambda_a = carry_a
    h_b, lambda_b = carry_b
    h_new = lambda_b * h_a + h_b
    lambda_new = lambda_b * lambda_a
    return (h_new, lambda_new)


def lru_forward(x_sequence: Array, lambda_diag: Array, B: Array, return_complex: bool = False) -> Array:
    """Run the recurrence h_t = Λ h_{t-1} + B x_t for the whole sequence at
    once via a binary reduction tree (10 steps for L=1024, not 1024)."""
    L = x_sequence.shape[0]

    B32 = B.astype(jnp.float32)
    x32 = x_sequence.astype(jnp.float32)
    if x32.ndim == 1:
        x32 = x32[:, None]
    if B32.ndim == 1:
        B32 = B32[:, None]

    h_init = jnp.einsum("di,li->ld", B32, x32).astype(jnp.complex64)

    lambda_c64 = lambda_diag.astype(jnp.complex64)
    lambdas = jnp.broadcast_to(lambda_c64[None, :], (L, lambda_c64.shape[0]))
    lambdas = jnp.array(lambdas)

    elements = (h_init, lambdas)
    h_all, _ = jax.lax.associative_scan(lru_combine, elements, axis=0)

    if return_complex:
        return h_all
    return jnp.real(h_all)


def init_lru_params(key: Array, D: int, D_in: int, r_min: float = 0.9, r_max: float = 0.999):
    """Random LRU parameters: stable eigenvalue magnitudes |λ| ∈ [r_min, r_max)
    (clipped below 1.0), uniform phases, Xavier-scaled input projection B."""
    key_r, key_phi, key_B = jax.random.split(key, 3)

    r = jax.random.uniform(key_r, (D,), minval=r_min, maxval=r_max)
    r = jnp.clip(r, 0.0, 0.9999)

    phi = jax.random.uniform(key_phi, (D,), minval=0.0, maxval=2.0 * jnp.pi)

    lambda_diag = r * jnp.exp(1j * phi)
    B = jax.random.normal(key_B, (D, D_in)) / jnp.sqrt(float(D_in))

    return {"lambda_diag": lambda_diag, "B": B}


def lru_forward_batched(x_batch: Array, lambda_diag: Array, B: Array) -> Array:
    """Batched forward pass: vmap over the batch dim."""
    return jax.vmap(lambda x: lru_forward(x, lambda_diag, B))(x_batch)
