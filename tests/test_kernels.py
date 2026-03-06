"""Tests for kernel functions."""

import math
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from efgp_jax.kernels import (
    SE,
    Matern,
    kernel_matrix,
    log_marginal,
    estimate_hyperparameters,
    pairwise_distances,
)


# ---------------------------------------------------------------------------
# SE kernel
# ---------------------------------------------------------------------------

def test_se_kernel_at_zero():
    k = SE(lengthscale=0.5, variance=2.0, dim=1)
    assert float(k(jnp.array(0.0))) == pytest.approx(2.0)


def test_se_kernel_symmetry():
    k = SE(lengthscale=0.3, variance=1.0, dim=1)
    r = jnp.linspace(0, 2, 50)
    vals = k(r)
    assert jnp.all(vals <= 1.0 + 1e-12)
    assert jnp.all(vals >= 0.0)


def test_se_kernel_matrix_shape():
    k = SE(lengthscale=0.5, variance=1.0, dim=1)
    x = jnp.linspace(0, 1, 10)
    K = kernel_matrix(x, x, k)
    assert K.shape == (10, 10)
    assert jnp.allclose(K, K.T, atol=1e-12)


def test_se_spectral_grad_finite_diff():
    k = SE(lengthscale=0.5, variance=1.0, dim=1)
    xi = jnp.linspace(-2, 2, 20)
    dl, dvar = k.spectral_grad(xi)

    h = 1e-6
    # d/dl
    k_plus = SE(lengthscale=k.lengthscale + h, variance=k.variance, dim=1)
    k_minus = SE(lengthscale=k.lengthscale - h, variance=k.variance, dim=1)
    dl_fd = (k_plus.spectral_density(xi) - k_minus.spectral_density(xi)) / (2 * h)
    assert jnp.allclose(dl, dl_fd, rtol=1e-4)

    # d/dvar
    k_plus = SE(lengthscale=k.lengthscale, variance=k.variance + h, dim=1)
    k_minus = SE(lengthscale=k.lengthscale, variance=k.variance - h, dim=1)
    dvar_fd = (k_plus.spectral_density(xi) - k_minus.spectral_density(xi)) / (2 * h)
    assert jnp.allclose(dvar, dvar_fd, rtol=1e-4)


# ---------------------------------------------------------------------------
# Matérn kernel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_matern_at_zero(nu):
    k = Matern(lengthscale=0.5, variance=2.0, nu=nu, dim=1)
    assert float(k(jnp.array(0.0))) == pytest.approx(2.0)


def test_matern_kernel_matrix_symmetric():
    k = Matern(lengthscale=0.5, variance=1.0, nu=2.5, dim=1)
    x = jnp.linspace(0, 1, 10)
    K = kernel_matrix(x, x, k)
    assert K.shape == (10, 10)
    assert jnp.allclose(K, K.T, atol=1e-12)


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_matern_spectral_grad_finite_diff(nu):
    k = Matern(lengthscale=0.5, variance=1.0, nu=nu, dim=1)
    xi = jnp.linspace(-2, 2, 20)
    dl, dvar = k.spectral_grad(xi)

    h = 1e-6
    k_plus = Matern(lengthscale=k.lengthscale + h, variance=k.variance, nu=nu, dim=1)
    k_minus = Matern(lengthscale=k.lengthscale - h, variance=k.variance, nu=nu, dim=1)
    dl_fd = (k_plus.spectral_density(xi) - k_minus.spectral_density(xi)) / (2 * h)
    assert jnp.allclose(dl, dl_fd, rtol=1e-3)


# ---------------------------------------------------------------------------
# Log marginal
# ---------------------------------------------------------------------------

def test_log_marginal_finite():
    key = jax.random.PRNGKey(0)
    x = jnp.linspace(0, 1, 20)
    y = jnp.sin(2 * jnp.pi * x) + 0.1 * jax.random.normal(key, (20,))
    k = SE(lengthscale=0.3, variance=1.0, dim=1)
    lml = log_marginal(x, y, 0.01, k)
    assert jnp.isfinite(lml)


# ---------------------------------------------------------------------------
# Estimate hyperparameters
# ---------------------------------------------------------------------------

def test_estimate_hyperparameters():
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (50, 2))
    y = jnp.sum(x, axis=1)
    l, v, nv = estimate_hyperparameters(x, y)
    assert l > 0
    assert v > 0
    assert nv > 0
