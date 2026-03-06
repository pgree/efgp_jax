"""Conjugate Gradients solver (single and batched), JIT-compatible."""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array


def cg_solve(
    A_apply: Callable[[Array], Array],
    b: Array,
    x0: Optional[Array] = None,
    tol: float = 1e-6,
    max_iter: Optional[int] = None,
    M_inv_apply: Optional[Callable[[Array], Array]] = None,
) -> Array:
    """Solve A x = b via Conjugate Gradients (single system).

    Uses jax.lax.while_loop for JIT compatibility.

    Parameters
    ----------
    A_apply : callable
        Function v -> A @ v.
    b : Array, shape (n,)
    x0 : Array, shape (n,), initial guess (zeros if None).
    tol : float
        Relative residual tolerance.
    max_iter : int or None
        Defaults to 2*n.
    M_inv_apply : callable or None
        Preconditioner M^{-1} v.

    Returns
    -------
    x : Array, shape (n,)
    """
    n = b.shape[0]
    if x0 is None:
        x0 = jnp.zeros_like(b)
    if max_iter is None:
        max_iter = 2 * n

    div_eps = 1e-16
    precond = M_inv_apply if M_inv_apply is not None else lambda v: v

    r0 = b - A_apply(x0)
    z0 = precond(r0)
    p0 = z0
    r_dot_z0 = jnp.vdot(r0, z0).real
    r0_norm = jnp.sqrt(r_dot_z0)

    # State: (x, r, z, p, r_dot_z, r0_norm, iter_count)
    init_state = (x0, r0, z0, p0, r_dot_z0, r0_norm, 0)

    def cond_fn(state):
        x, r, z, p, r_dot_z, r0_norm, i = state
        rel_res = jnp.sqrt(jnp.abs(r_dot_z)) / (r0_norm + div_eps)
        return (i < max_iter) & (rel_res > tol)

    def body_fn(state):
        x, r, z, p, r_dot_z, r0_norm, i = state
        Ap = A_apply(p)
        pAp = jnp.vdot(p, Ap).real + div_eps
        alpha = r_dot_z / pAp
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        z_new = precond(r_new)
        r_dot_z_new = jnp.vdot(r_new, z_new).real
        beta = r_dot_z_new / (r_dot_z + div_eps)
        p_new = z_new + beta * p
        return (x_new, r_new, z_new, p_new, r_dot_z_new, r0_norm, i + 1)

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return final_state[0]


def cg_solve_batched(
    A_apply: Callable[[Array], Array],
    b: Array,
    x0: Optional[Array] = None,
    tol: float = 1e-6,
    max_iter: Optional[int] = None,
    M_inv_apply: Optional[Callable[[Array], Array]] = None,
) -> Array:
    """Solve A x_i = b_i for a batch of right-hand sides.

    Uses jax.lax.while_loop for JIT compatibility.

    Parameters
    ----------
    A_apply : callable
        Function (B, n) -> (B, n).
    b : Array, shape (B, n)
    x0 : Array, shape (B, n), initial guess (zeros if None).
    tol : float
    max_iter : int or None
    M_inv_apply : callable or None

    Returns
    -------
    x : Array, shape (B, n)
    """
    B, n = b.shape
    if x0 is None:
        x0 = jnp.zeros_like(b)
    if max_iter is None:
        max_iter = 2 * n

    div_eps = 1e-16
    precond = M_inv_apply if M_inv_apply is not None else lambda v: v

    r0 = b - A_apply(x0)
    z0 = precond(r0)
    p0 = z0
    r_dot_z0 = jnp.sum(jnp.conj(r0) * z0, axis=1).real  # (B,)
    r0_norm = jnp.sqrt(r_dot_z0)

    # State: (x, r, z, p, r_dot_z, r0_norm, iter_count)
    init_state = (x0, r0, z0, p0, r_dot_z0, r0_norm, 0)

    def cond_fn(state):
        x, r, z, p, r_dot_z, r0_norm, i = state
        rel_res = jnp.sqrt(jnp.abs(r_dot_z)) / (r0_norm + div_eps)
        return (i < max_iter) & jnp.any(rel_res > tol)

    def body_fn(state):
        x, r, z, p, r_dot_z, r0_norm, i = state
        Ap = A_apply(p)
        pAp = jnp.sum(jnp.conj(p) * Ap, axis=1).real + div_eps  # (B,)
        alpha = r_dot_z / pAp  # (B,)
        x_new = x + alpha[:, None] * p
        r_new = r - alpha[:, None] * Ap
        z_new = precond(r_new)
        r_dot_z_new = jnp.sum(jnp.conj(r_new) * z_new, axis=1).real
        beta = r_dot_z_new / (r_dot_z + div_eps)
        p_new = z_new + beta[:, None] * p
        return (x_new, r_new, z_new, p_new, r_dot_z_new, r0_norm, i + 1)

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return final_state[0]
