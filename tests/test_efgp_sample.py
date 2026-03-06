"""Tests for efgp_sample_posterior (Matheron rule posterior sampling)."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from efgp_jax.kernels import SE, pairwise_distances
from efgp_jax.efgp import efgp_predict, efgp_predict_var, efgp_sample_posterior
from efgp_jax.gp import sample_posterior


EPSILON = 1e-4


def _make_small_dataset(n=50, seed=0):
    """Small 1-D dataset for testing."""
    key = jax.random.PRNGKey(seed)
    kernel = SE(lengthscale=0.2, variance=1.0, dim=1)
    x = jnp.linspace(0.05, 0.95, n)
    K = kernel(pairwise_distances(x[:, None], x[:, None]))
    K_noise = K + 0.01 * jnp.eye(n)
    L = jnp.linalg.cholesky(K_noise)
    y = L @ jax.random.normal(key, (n,))
    return x, y, kernel


def test_shape_single():
    """n_samples=1 returns shape (n_new,)."""
    x, y, kernel = _make_small_dataset()
    x_new = jnp.linspace(0.0, 1.0, 30)
    key = jax.random.PRNGKey(42)

    s = efgp_sample_posterior(
        x, y, x_new, kernel, sigmasq=0.01, eps=EPSILON,
        key=key, n_samples=1,
    )
    assert s.shape == (30,)


def test_shape_multi():
    """n_samples>1 returns shape (n_samples, n_new)."""
    x, y, kernel = _make_small_dataset()
    x_new = jnp.linspace(0.0, 1.0, 30)
    key = jax.random.PRNGKey(42)

    s = efgp_sample_posterior(
        x, y, x_new, kernel, sigmasq=0.01, eps=EPSILON,
        key=key, n_samples=7,
    )
    assert s.shape == (7, 30)


def test_statistics_match_efgp_predict():
    """Empirical mean/variance of many samples should match efgp_predict / efgp_predict_var."""
    x, y, kernel = _make_small_dataset(n=80, seed=1)
    x_new = jnp.linspace(0.1, 0.9, 20)
    sigmasq = 0.01

    # Reference mean and variance
    yhat = efgp_predict(
        x, y, x_new, kernel, sigmasq=sigmasq, eps=EPSILON,
    )
    _, var = efgp_predict_var(
        x, y, x_new, kernel, sigmasq=sigmasq, eps=EPSILON,
    )

    # Draw many posterior samples
    n_samples = 5_000
    key = jax.random.PRNGKey(123)
    samples = efgp_sample_posterior(
        x, y, x_new, kernel, sigmasq=sigmasq, eps=EPSILON,
        key=key, n_samples=n_samples,
    )
    assert samples.shape == (n_samples, 20)

    emp_mean = jnp.mean(samples, axis=0)
    emp_var = jnp.var(samples, axis=0)

    # Check mean (allow generous tolerance for Monte Carlo)
    mean_err = jnp.max(jnp.abs(emp_mean - yhat))
    assert mean_err < 0.15, f"max mean error = {mean_err}"

    # Check variance (relative)
    # Skip points where true variance is very small
    mask = var > 0.01
    if jnp.any(mask):
        rel_var_err = jnp.max(jnp.abs(emp_var[mask] - var[mask]) / var[mask])
        assert rel_var_err < 0.5, f"max relative variance error = {rel_var_err}"


def test_small_data_vs_cholesky():
    """On a small dataset, EFGP posterior samples should have similar statistics to Cholesky."""
    n = 30
    x, y, kernel = _make_small_dataset(n=n, seed=2)
    x_new = jnp.linspace(0.15, 0.85, 15)
    sigmasq = 0.05

    n_samples = 5_000

    # EFGP samples
    key = jax.random.PRNGKey(0)
    efgp_samples = efgp_sample_posterior(
        x, y, x_new, kernel, sigmasq=sigmasq, eps=EPSILON,
        key=key, n_samples=n_samples,
    )

    # Cholesky samples
    key = jax.random.PRNGKey(1)
    chol_samples = sample_posterior(
        x, y, x_new, sigmasq, kernel, key, n_samples=n_samples,
    )  # (n_new, n_samples)
    chol_samples = chol_samples.T  # (n_samples, n_new)

    # Compare means
    efgp_mean = jnp.mean(efgp_samples, axis=0)
    chol_mean = jnp.mean(chol_samples, axis=0)
    mean_diff = jnp.max(jnp.abs(efgp_mean - chol_mean))
    assert mean_diff < 0.15, f"mean difference = {mean_diff}"

    # Compare variances
    efgp_var = jnp.var(efgp_samples, axis=0)
    chol_var = jnp.var(chol_samples, axis=0)
    mask = chol_var > 0.01
    if jnp.any(mask):
        rel_var_diff = jnp.max(jnp.abs(efgp_var[mask] - chol_var[mask]) / chol_var[mask])
        assert rel_var_diff < 0.5, f"relative variance difference = {rel_var_diff}"
