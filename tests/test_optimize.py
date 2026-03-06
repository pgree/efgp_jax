"""Tests for hyperparameter optimization."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from efgp_jax.kernels import SE
from efgp_jax.optimize import optimize_hyperparameters


def test_optimize_runs():
    """Run a few optimization steps and check output types."""
    key = jax.random.PRNGKey(0)
    x = jnp.linspace(0, 1, 50)
    y = jnp.sin(2 * jnp.pi * x) + 0.1 * jax.random.normal(key, (50,))

    kernel0 = SE(lengthscale=0.5, variance=1.0, dim=1)
    sigmasq0 = 0.1

    kernel_opt, sigmasq_opt, info = optimize_hyperparameters(
        x, y,
        kernel0=kernel0,
        sigmasq0=sigmasq0,
        eps=1e-2,
        key=jax.random.PRNGKey(1),
        maxiter=5,
        trace_samples=5,
        log_marginal_probes=10,
        log_marginal_steps=10,
        verbose=False,
    )

    assert kernel_opt.lengthscale > 0
    assert kernel_opt.variance > 0
    assert sigmasq_opt > 0
    assert 'nll' in info
    assert 'nfev' in info
