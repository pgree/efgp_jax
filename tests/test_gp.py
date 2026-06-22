"""Tests for the vanilla GP module."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from efgp_jax.kernels import SE, Matern
from efgp_jax.gp import posterior_covariance, predict, sample_posterior


def _toy_data(n=30, seed=0):
    key = jax.random.PRNGKey(seed)
    x = jnp.linspace(0, 1, n)
    y = jnp.sin(2 * jnp.pi * x) + 0.1 * jax.random.normal(key, (n,))
    return x, y


def test_predict_shape():
    x, y = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    x_new = jnp.linspace(0, 1, 20)
    mean = predict(x, y, x_new, 0.01, kernel)
    assert mean.shape == (20,)


def test_predict_with_variance():
    x, y = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    x_new = jnp.linspace(0, 1, 20)
    mean, var = predict(x, y, x_new, 0.01, kernel, compute_var=True)
    assert mean.shape == (20,)
    assert var.shape == (20,)
    assert jnp.all(var >= -1e-8)  # variance should be non-negative (up to numerics)


def test_posterior_covariance_shape():
    x, _ = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    x_trgs = jnp.linspace(0, 1, 15)
    cov = posterior_covariance(x, x_trgs, 0.01, kernel)
    assert cov.shape == (15, 15)
    # Should be approximately symmetric
    assert jnp.allclose(cov, cov.T, atol=1e-10)


def test_posterior_covariance_positive_diagonal():
    x, _ = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    x_trgs = jnp.linspace(0, 1, 15)
    cov = posterior_covariance(x, x_trgs, 0.01, kernel)
    assert jnp.all(jnp.diag(cov) > 0)


def test_sample_posterior_shape():
    x, y = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    x_new = jnp.linspace(0, 1, 10)
    key = jax.random.PRNGKey(42)
    samples = sample_posterior(x, y, x_new, 0.01, kernel, key, n_samples=5)
    assert samples.shape == (10, 5)


def test_predict_interpolates():
    """Posterior mean should pass close to training data when noise is small."""
    x, y = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    mean = predict(x, y, x, 1e-4, kernel)
    # With small noise, predictions at training points should be close to y
    assert jnp.max(jnp.abs(mean - y)) < 0.25


def test_predict_matern():
    x, y = _toy_data()
    kernel = Matern(lengthscale=0.3, variance=1.0, nu=2.5, dim=1)
    x_new = jnp.linspace(0, 1, 20)
    mean = predict(x, y, x_new, 0.01, kernel)
    assert mean.shape == (20,)
    assert jnp.all(jnp.isfinite(mean))


# ---------------------------------------------------------------------------
# JAX traceability tests
# ---------------------------------------------------------------------------


def test_predict_jittable():
    """jit(predict) with kernel as a regular pytree arg — no static_argnames."""
    x, y = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    x_new = jnp.linspace(0, 1, 20)

    jitted = jax.jit(predict)
    m_ref = predict(x, y, x_new, 0.01, kernel)
    m_jit = jitted(x, y, x_new, 0.01, kernel)

    assert m_jit.shape == m_ref.shape
    assert jnp.allclose(m_jit, m_ref, rtol=1e-12, atol=1e-12)

    kernel2 = SE(lengthscale=0.1, variance=1.0, dim=1)
    m2 = jitted(x, y, x_new, 0.01, kernel2)
    assert not jnp.allclose(m2, m_jit)


def test_predict_jittable_with_variance():
    """jit(predict, compute_var=True) — compute_var is a static Python bool."""
    x, y = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    x_new = jnp.linspace(0, 1, 20)

    jitted = jax.jit(predict, static_argnames=("compute_var",))
    mean_ref, var_ref = predict(x, y, x_new, 0.01, kernel, compute_var=True)
    mean_jit, var_jit = jitted(x, y, x_new, 0.01, kernel, compute_var=True)

    assert jnp.allclose(mean_jit, mean_ref, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(var_jit, var_ref, rtol=1e-10, atol=1e-12)


def test_posterior_covariance_jittable():
    x, _ = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    x_trgs = jnp.linspace(0, 1, 15)

    jitted = jax.jit(posterior_covariance)
    c_ref = posterior_covariance(x, x_trgs, 0.01, kernel)
    c_jit = jitted(x, x_trgs, 0.01, kernel)

    assert c_jit.shape == c_ref.shape
    assert jnp.allclose(c_jit, c_ref, rtol=1e-12, atol=1e-12)


def test_sample_posterior_jittable():
    """jit(sample_posterior) — n_samples is static (it sets a shape)."""
    x, y = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    x_new = jnp.linspace(0, 1, 10)
    key = jax.random.PRNGKey(42)

    jitted = jax.jit(sample_posterior, static_argnames=("n_samples",))
    s_ref = sample_posterior(x, y, x_new, 0.01, kernel, key, n_samples=5)
    s_jit = jitted(x, y, x_new, 0.01, kernel, key, n_samples=5)

    assert s_jit.shape == (10, 5)
    assert jnp.allclose(s_jit, s_ref, rtol=1e-10, atol=1e-10)
