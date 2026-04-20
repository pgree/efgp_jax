"""Kernel functions: Squared Exponential and Matérn."""

import math
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class


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

@register_pytree_node_class
class SE(Kernel):
    """Squared Exponential (RBF) kernel.

    k(r) = variance * exp(-0.5 * r^2 / lengthscale^2)

    For anisotropic kernels, pass ``lengthscale`` as an array of length
    ``dim``.  The spectral density factorizes over dimensions.  The
    ``__call__`` method (which takes scalar distances) only works for
    the isotropic case.
    """

    def __init__(self, lengthscale, variance, dim: int = 1):
        l_arr = jnp.asarray(lengthscale)
        # Detect isotropic vs anisotropic based on shape/size.
        if l_arr.ndim == 0 or l_arr.size == 1:
            # scalar — store as 0-d jax array, isotropic
            l_stored = l_arr.reshape(())
            is_anisotropic = False
        else:
            if l_arr.size != dim:
                raise ValueError(
                    f"lengthscale array has {l_arr.size} entries but dim={dim}"
                )
            l_stored = l_arr.reshape((dim,))
            is_anisotropic = True

        self.lengthscale = l_stored
        self.variance = jnp.asarray(variance)
        self.dim = dim
        self.is_anisotropic = is_anisotropic

    # --- pytree protocol ---------------------------------------------------

    def tree_flatten(self):
        children = (self.lengthscale, self.variance)
        aux = (self.dim, self.is_anisotropic)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        dim, is_anisotropic = aux
        lengthscale, variance = children
        obj = cls.__new__(cls)
        obj.lengthscale = lengthscale
        obj.variance = variance
        obj.dim = dim
        obj.is_anisotropic = is_anisotropic
        return obj

    # --- kernel methods ----------------------------------------------------

    def __call__(self, r: Array) -> Array:
        if self.is_anisotropic:
            raise NotImplementedError(
                "SE.__call__(r) requires scalar distances; use spectral methods "
                "for anisotropic kernels"
            )
        scaled = r / self.lengthscale
        return self.variance * jnp.exp(-0.5 * scaled ** 2)

    def spectral_density(self, xi: Array) -> Array:
        l = self.lengthscale
        var = self.variance
        dim = self.dim
        xi = jnp.atleast_1d(xi)
        two_pi_sq = (2 * jnp.pi) ** 2
        if self.is_anisotropic:
            l = jnp.asarray(l)
            # S(xi) = var * prod_d(sqrt(2*pi) * l_d) * exp(-2*pi^2 * sum_d(l_d^2 * xi_d^2))
            prefactor = var * jnp.prod(jnp.sqrt(2 * jnp.pi) * l)
            weighted_sq = jnp.sum(l ** 2 * xi ** 2, axis=-1)
            return prefactor * jnp.exp(-two_pi_sq * weighted_sq / 2)
        else:
            if xi.ndim == 1:
                xi_norm_sq = xi ** 2
            else:
                xi_norm_sq = jnp.sum(xi ** 2, axis=-1)
            prefactor = ((2 * jnp.pi) * l ** 2) ** (dim / 2) * var
            return prefactor * jnp.exp(-two_pi_sq * l ** 2 * xi_norm_sq / 2)

    def spectral_grad(self, xi: Array) -> Tuple[Array, Array]:
        var = self.variance
        two_pi_sq = (2 * jnp.pi) ** 2
        s_w = self.spectral_density(xi)
        dvar = s_w / var
        if self.is_anisotropic:
            l = jnp.asarray(self.lengthscale)
            # dS/dl_d = S * (1/l_d - (2*pi)^2 * l_d * xi_d^2)
            dl = s_w[..., None] * (1.0 / l - two_pi_sq * l * xi ** 2)  # (M, d)
            return dl, dvar
        else:
            l = self.lengthscale
            dim = self.dim
            if xi.ndim == 1:
                xi_norm_sq = xi ** 2
            else:
                xi_norm_sq = jnp.sum(xi ** 2, axis=-1)
            dl = s_w * (dim / l - two_pi_sq * l * xi_norm_sq)
            return dl, dvar

    @property
    def name(self) -> str:
        return "se"


# ---------------------------------------------------------------------------
# Matérn
# ---------------------------------------------------------------------------

@register_pytree_node_class
class Matern(Kernel):
    """Matérn kernel for nu in {0.5, 1.5, 2.5}."""

    def __init__(self, lengthscale, variance, dim: int = 1,
                 nu: float = 2.5):
        self.lengthscale = jnp.asarray(lengthscale)
        self.variance = jnp.asarray(variance)
        self.dim = dim
        self.nu = nu

    # --- pytree protocol ---------------------------------------------------

    def tree_flatten(self):
        children = (self.lengthscale, self.variance)
        aux = (self.dim, self.nu)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        dim, nu = aux
        lengthscale, variance = children
        obj = cls.__new__(cls)
        obj.lengthscale = lengthscale
        obj.variance = variance
        obj.dim = dim
        obj.nu = nu
        return obj

    # --- kernel methods ----------------------------------------------------

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
