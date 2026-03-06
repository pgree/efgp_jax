"""Tests for the Conjugate Gradients solver."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from efgp_jax.cg import cg_solve, cg_solve_batched


def test_cg_solve_spd():
    """Solve Ax = b with a known SPD matrix."""
    n = 50
    key = jax.random.PRNGKey(0)
    A_half = jax.random.normal(key, (n, n))
    A = A_half.T @ A_half + 5.0 * jnp.eye(n)  # well-conditioned SPD
    b = jnp.ones(n)
    x_true = jnp.linalg.solve(A, b)

    A_apply = lambda v: A @ v
    x_cg = cg_solve(A_apply, b, tol=1e-10)

    assert jnp.allclose(x_cg, x_true, atol=1e-6)


def test_cg_solve_residual():
    """Check that the residual ||Ax - b|| is small."""
    n = 30
    key = jax.random.PRNGKey(1)
    A_half = jax.random.normal(key, (n, n))
    A = A_half.T @ A_half + 3.0 * jnp.eye(n)
    b = jax.random.normal(jax.random.PRNGKey(2), (n,))

    x = cg_solve(lambda v: A @ v, b, tol=1e-10)
    residual = jnp.linalg.norm(A @ x - b) / jnp.linalg.norm(b)
    assert residual < 1e-6


def test_cg_solve_batched():
    """Batched CG should solve multiple systems."""
    n = 30
    B = 5
    key = jax.random.PRNGKey(3)
    A_half = jax.random.normal(key, (n, n))
    A = A_half.T @ A_half + 5.0 * jnp.eye(n)
    b = jax.random.normal(jax.random.PRNGKey(4), (B, n))

    A_apply = lambda v: (v @ A.T)  # (B, n) @ (n, n) -> (B, n)
    x = cg_solve_batched(A_apply, b, tol=1e-10)

    for i in range(B):
        residual = jnp.linalg.norm(A @ x[i] - b[i]) / jnp.linalg.norm(b[i])
        assert residual < 1e-5


def test_cg_with_preconditioner():
    """CG with diagonal preconditioner."""
    n = 50
    key = jax.random.PRNGKey(5)
    A_half = jax.random.normal(key, (n, n))
    A = A_half.T @ A_half + 2.0 * jnp.eye(n)
    b = jax.random.normal(jax.random.PRNGKey(6), (n,))
    diag_A = jnp.diag(A)

    M_inv = lambda v: v / diag_A
    x = cg_solve(lambda v: A @ v, b, tol=1e-10, M_inv_apply=M_inv)
    residual = jnp.linalg.norm(A @ x - b) / jnp.linalg.norm(b)
    assert residual < 1e-6


def test_cg_jit():
    """Verify cg_solve works under jax.jit."""
    n = 30
    key = jax.random.PRNGKey(7)
    A_half = jax.random.normal(key, (n, n))
    A = A_half.T @ A_half + 5.0 * jnp.eye(n)
    b = jnp.ones(n)
    x_true = jnp.linalg.solve(A, b)

    @jax.jit
    def solve_jit(b):
        return cg_solve(lambda v: A @ v, b, tol=1e-10, max_iter=200)

    x_cg = solve_jit(b)
    assert jnp.allclose(x_cg, x_true, atol=1e-5)
