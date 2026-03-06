"""Vanilla GP: predict, posterior covariance, sample."""

import math
from typing import Optional, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .kernels import Kernel, pairwise_distances


def posterior_covariance(
    x: Array,
    x_trgs: Array,
    sigmasq: float,
    kernel: Kernel,
) -> Array:
    """Full posterior covariance matrix at target locations.

    Parameters
    ----------
    x : Array, shape (n, d)
    x_trgs : Array, shape (m, d)
    sigmasq : float
    kernel : Kernel

    Returns
    -------
    cov : Array, shape (m, m)
    """
    if x.ndim == 1:
        x = x[:, None]
    if x_trgs.ndim == 1:
        x_trgs = x_trgs[:, None]

    n = x.shape[0]

    K_obs = kernel(pairwise_distances(x, x))
    K_noise = K_obs + sigmasq * jnp.eye(n)

    K_cross = kernel(pairwise_distances(x_trgs, x))
    K_trgs = kernel(pairwise_distances(x_trgs, x_trgs))

    cov = K_trgs - K_cross @ jnp.linalg.solve(K_noise, K_cross.T)
    cov = cov + 1e-10 * jnp.eye(x_trgs.shape[0])
    return cov


def predict(
    x: Array,
    y: Array,
    x_trgs: Array,
    sigmasq: float,
    kernel: Kernel,
    compute_var: bool = False,
) -> Array:
    """Posterior mean (and optionally variance) at target locations.

    Parameters
    ----------
    x : Array, shape (n, d) or (n,)
    y : Array, shape (n,)
    x_trgs : Array, shape (m, d) or (m,)
    sigmasq : float
    kernel : Kernel
    compute_var : bool

    Returns
    -------
    mean : Array, shape (m,)
    var : Array, shape (m,)  (only if compute_var=True)
    """
    if x.ndim == 1:
        x = x[:, None]
    if x_trgs.ndim == 1:
        x_trgs = x_trgs[:, None]

    n = x.shape[0]

    K = kernel(pairwise_distances(x, x))
    K_noise = K + sigmasq * jnp.eye(n)

    L = jnp.linalg.cholesky(K_noise)
    alpha = jax.scipy.linalg.cho_solve((L, True), y)

    K_cross = kernel(pairwise_distances(x_trgs, x))
    mean = K_cross @ alpha

    if compute_var:
        cov = posterior_covariance(x, x_trgs, sigmasq, kernel)
        var = jnp.diag(cov)
        return mean, var

    return mean


def sample_posterior(
    x: Array,
    y: Array,
    x_new: Array,
    sigmasq: float,
    kernel: Kernel,
    key: Array,
    n_samples: int = 1,
) -> Array:
    """Draw samples from the GP posterior.

    Returns
    -------
    samples : Array, shape (n_new, n_samples)
    """
    if x.ndim == 1:
        x = x[:, None]
    if x_new.ndim == 1:
        x_new = x_new[:, None]

    n_new = x_new.shape[0]
    cov = posterior_covariance(x, x_new, sigmasq, kernel)
    L = jnp.linalg.cholesky(cov)
    Z = jax.random.normal(key, shape=(n_new, n_samples))

    mean = predict(x, y, x_new, sigmasq, kernel)
    samples = mean[:, None] + L @ Z
    return samples
