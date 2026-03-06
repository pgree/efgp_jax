"""Kernel functions: Squared Exponential and Matérn."""

import math
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Kernel:
    """Base class for kernels with spectral methods."""

    def __init__(self, lengthscale: float, variance: float, dim: int = 1):
        self.lengthscale = lengthscale
        self.variance = variance
        self.dim = dim

    def __call__(self, r: Array) -> Array:
        raise NotImplementedError

    def spectral_density(self, xi: Array) -> Array:
        raise NotImplementedError

    def spectral_grad(self, xi: Array) -> Tuple[Array, Array]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Squared Exponential
# ---------------------------------------------------------------------------

class SE(Kernel):
    """Squared Exponential (RBF) kernel.

    k(r) = variance * exp(-0.5 * r^2 / lengthscale^2)
    """

    def __init__(self, lengthscale: float, variance: float, dim: int = 1):
        super().__init__(lengthscale, variance, dim)

    def __call__(self, r: Array) -> Array:
        scaled = r / self.lengthscale
        return self.variance * jnp.exp(-0.5 * scaled ** 2)

    def spectral_density(self, xi: Array) -> Array:
        l = self.lengthscale
        var = self.variance
        dim = self.dim
        xi = jnp.atleast_1d(xi)
        if xi.ndim == 1:
            xi_norm_sq = xi ** 2
        else:
            xi_norm_sq = jnp.sum(xi ** 2, axis=-1)
        two_pi_sq = (2 * jnp.pi) ** 2
        prefactor = ((2 * jnp.pi) * l ** 2) ** (dim / 2) * var
        return prefactor * jnp.exp(-two_pi_sq * l ** 2 * xi_norm_sq / 2)

    def spectral_grad(self, xi: Array) -> Tuple[Array, Array]:
        l = self.lengthscale
        var = self.variance
        dim = self.dim
        if xi.ndim == 1:
            xi_norm_sq = xi ** 2
        else:
            xi_norm_sq = jnp.sum(xi ** 2, axis=-1)
        two_pi_sq = (2 * jnp.pi) ** 2
        s_w = self.spectral_density(xi)
        dl = s_w * (dim / l - two_pi_sq * l * xi_norm_sq)
        dvar = s_w / var
        return dl, dvar

    @property
    def name(self) -> str:
        return "se"


# ---------------------------------------------------------------------------
# Matérn
# ---------------------------------------------------------------------------

class Matern(Kernel):
    """Matérn kernel for nu in {0.5, 1.5, 2.5}."""

    def __init__(self, lengthscale: float, variance: float, dim: int = 1,
                 nu: float = 2.5):
        super().__init__(lengthscale, variance, dim)
        self.nu = nu

    def __call__(self, r: Array) -> Array:
        l = self.lengthscale
        var = self.variance
        nu = self.nu
        scaled = jnp.abs(r) / l

        if nu == 0.5:
            return var * jnp.exp(-scaled)
        elif nu == 1.5:
            s3 = math.sqrt(3) * scaled
            return var * (1 + s3) * jnp.exp(-s3)
        elif nu == 2.5:
            s5 = math.sqrt(5) * scaled
            return var * (1 + s5 + 5 * scaled ** 2 / 3) * jnp.exp(-s5)
        else:
            raise NotImplementedError(
                f"Matérn kernel only implemented for nu in {{0.5, 1.5, 2.5}}, got {nu}"
            )

    def spectral_density(self, xi: Array) -> Array:
        l = self.lengthscale
        var = self.variance
        nu = self.nu
        dim = self.dim

        xi = jnp.atleast_1d(xi)
        if xi.ndim == 1:
            xi_norm_sq = xi ** 2
        else:
            xi_norm_sq = jnp.sum(xi ** 2, axis=-1)

        scaling = (
            (2 * math.sqrt(math.pi)) ** dim
            * math.gamma(nu + dim / 2)
            * (2 * nu) ** nu
            / (math.gamma(nu) * l ** (2 * nu))
        )
        return var * scaling * (2 * nu / l ** 2 + (4 * math.pi ** 2) * xi_norm_sq) ** (-(nu + dim / 2))

    def spectral_grad(self, xi: Array) -> Tuple[Array, Array]:
        l = self.lengthscale
        var = self.variance
        nu = self.nu
        dim = self.dim

        if xi.ndim == 1:
            xi_norm_sq = xi ** 2
        else:
            xi_norm_sq = jnp.sum(xi ** 2, axis=-1)

        S = self.spectral_density(xi)
        dS_dvar = S / var

        denominator = 2 * nu / l ** 2 + (4 * math.pi ** 2) * xi_norm_sq
        power = -(nu + dim / 2)
        exponent_grad = power * (-4 * nu / l ** 3) / denominator
        dS_dl = S * (-2 * nu / l + exponent_grad)

        return dS_dl, dS_dvar

    @property
    def name(self) -> str:
        return "matern"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def pairwise_distances(x: Array, y: Array) -> Array:
    """Euclidean pairwise distance matrix."""
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    return jnp.sqrt(jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1))


def kernel_matrix(x: Array, y: Array, kernel: Kernel) -> Array:
    """Generic kernel matrix from a kernel object."""
    dists = pairwise_distances(x, y)
    return kernel(dists)


def log_marginal(x: Array, y: Array, sigmasq: float, kernel: Kernel) -> float:
    """Log marginal likelihood via Cholesky.

    log p(y|X) = -0.5 * (y^T K_noise^{-1} y + log|K_noise| + n log(2pi))
    """
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    K = kernel_matrix(x, x, kernel)
    K_noise = K + sigmasq * jnp.eye(n)
    L = jnp.linalg.cholesky(K_noise)
    alpha = jax.scipy.linalg.cho_solve((L, True), y)
    data_fit = 0.5 * jnp.sum(y * alpha)
    complexity = jnp.sum(jnp.log(jnp.diag(L)))
    constant = 0.5 * n * math.log(2 * math.pi)
    return -(data_fit + complexity + constant)


def estimate_hyperparameters(x: Array, y: Array, K: int = 1000) -> Tuple[float, float, float]:
    """Median-distance heuristic for initial hyperparameters.

    Returns (lengthscale, variance, noise_var).
    """
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    y_var = float(jnp.var(y))
    if n > K:
        idx = jax.random.permutation(jax.random.PRNGKey(0), n)[:K]
        x_sample = x[idx]
    else:
        x_sample = x
    dists = pairwise_distances(x_sample, x_sample)
    mask = dists > 0
    median_dist = float(jnp.median(dists[mask]))
    lengthscale = 0.5 * median_dist
    variance = y_var
    noise_var = 0.2 * y_var
    return lengthscale, variance, noise_var
