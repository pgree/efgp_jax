"""Tests for NUFFT wrapper."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from efgp_jax.nufft import nufft_type1, nufft_type2, _make_phi


def test_nufft_type1_batch():
    """Batched type-1 NUFFT (vmap) should match single calls."""
    N = 20
    B = 3
    m = 15
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (N, 1))
    phi = _make_phi(x, jnp.zeros(1), 1.0)
    vals = jax.random.normal(jax.random.PRNGKey(1), (B, N)).astype(jnp.complex128)
    out_shape = (m,)

    # Batched
    result_batch = nufft_type1(phi, vals, out_shape, eps=1e-8)

    # Single calls
    singles = []
    for i in range(B):
        singles.append(nufft_type1(phi, vals[i], out_shape, eps=1e-8))
    result_single = jnp.stack(singles)

    assert jnp.allclose(result_batch, result_single, atol=1e-10)


def test_nufft_type2_batch():
    """Batched type-2 NUFFT (vmap) should match single calls."""
    N = 20
    B = 3
    m = 15
    key = jax.random.PRNGKey(2)
    x = jax.random.uniform(key, (N, 1))
    phi = _make_phi(x, jnp.zeros(1), 1.0)
    fk = jax.random.normal(jax.random.PRNGKey(3), (B, m)).astype(jnp.complex128)

    # Batched
    result_batch = nufft_type2(phi, fk, eps=1e-8)

    # Single calls
    singles = []
    for i in range(B):
        singles.append(nufft_type2(phi, fk[i], eps=1e-8))
    result_single = jnp.stack(singles)

    assert jnp.allclose(result_batch, result_single, atol=1e-10)
