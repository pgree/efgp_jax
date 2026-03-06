"""Tests for the main EFGP algorithm."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from efgp_jax.kernels import SE, Matern
from efgp_jax.efgp import efgp_gradient, efgp_predict, efgp_predict_var
from efgp_jax.gp import predict as gp_predict, posterior_covariance


def _toy_data(n=50, seed=0):
    key = jax.random.PRNGKey(seed)
    x = jnp.linspace(0, 1, n)
    y = jnp.sin(2 * jnp.pi * x) + 0.1 * jax.random.normal(key, (n,))
    return x, y


def test_efgp_gradient_shape():
    x, y = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    key = jax.random.PRNGKey(1)
    grad = efgp_gradient(
        x, y, sigmasq=0.01,
        kernel=kernel,
        eps=1e-2, trace_samples=5,
        key=key,
    )
    assert grad.shape == (3,)
    assert jnp.all(jnp.isfinite(grad))


def test_efgp_gradient_with_log_marginal():
    x, y = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    key = jax.random.PRNGKey(2)
    grad, lml = efgp_gradient(
        x, y, sigmasq=0.01,
        kernel=kernel,
        eps=1e-2, trace_samples=5,
        key=key,
        compute_log_marginal=True,
        log_marginal_probes=10,
        log_marginal_steps=10,
    )
    assert grad.shape == (3,)
    assert jnp.isfinite(lml)


def test_efgp_predict_matches_vanilla():
    """EFGP posterior mean should be close to vanilla GP on small data."""
    x, y = _toy_data(n=30)
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    sigmasq = 0.01
    x_new = jnp.linspace(0.1, 0.9, 10)

    # Vanilla GP
    mean_gp = gp_predict(x, y, x_new, sigmasq, kernel)

    # EFGP
    mean_efgp = efgp_predict(
        x, y, x_new,
        kernel=kernel,
        sigmasq=sigmasq, eps=1e-6,
    )

    assert jnp.allclose(mean_gp, mean_efgp, atol=0.05)


def test_efgp_predict_var():
    """Check variance shape, non-negativity, and agreement with vanilla GP."""
    x, y = _toy_data(n=30)
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    sigmasq = 0.01
    x_new = jnp.linspace(0.1, 0.9, 5)

    yhat, var = efgp_predict_var(
        x, y, x_new, kernel=kernel,
        sigmasq=sigmasq, eps=1e-6,
    )
    assert yhat.shape == (5,)
    assert var.shape == (5,)
    assert jnp.all(var >= 0)

    # Compare with vanilla GP posterior variance
    cov_gp = posterior_covariance(x, x_new, sigmasq, kernel)
    var_gp = jnp.diag(cov_gp)
    assert jnp.allclose(var, var_gp, atol=0.05)


def test_efgp_matern():
    """Run efgp_predict with Matérn kernel."""
    x, y = _toy_data(n=30)
    kernel = Matern(lengthscale=0.3, variance=1.0, nu=2.5, dim=1)
    sigmasq = 0.01
    x_new = jnp.linspace(0.1, 0.9, 10)

    mean_efgp = efgp_predict(
        x, y, x_new, kernel=kernel,
        sigmasq=sigmasq, eps=1e-4,
    )
    assert mean_efgp.shape == (10,)
    assert jnp.all(jnp.isfinite(mean_efgp))


def test_efgp_2d():
    """Small 2D dataset, check efgp_predict shape."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    x = jax.random.uniform(k1, (30, 2))
    y = jnp.sin(2 * jnp.pi * x[:, 0]) + jnp.cos(2 * jnp.pi * x[:, 1])
    x_new = jax.random.uniform(k2, (5, 2))

    kernel = SE(lengthscale=0.3, variance=1.0, dim=2)
    sigmasq = 0.01

    mean_efgp = efgp_predict(
        x, y, x_new, kernel=kernel,
        sigmasq=sigmasq, eps=1e-3,
    )
    assert mean_efgp.shape == (5,)
    assert jnp.all(jnp.isfinite(mean_efgp))
