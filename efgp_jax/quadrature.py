"""Frequency grid generation for EFGP quadrature."""

import math
from typing import Callable, Tuple

import jax.numpy as jnp
from jax import Array

from .kernels import Kernel


def find_truncation_bound(
    f: Callable[[float], float],
    eps: float,
    initial_upper: float = 1000.0,
    max_iter: int = 200,
) -> float:
    """Find L such that f(L) ~ eps for monotonically decreasing f via bisection."""
    b = initial_upper
    for _ in range(10):
        if f(b) > eps:
            b *= 2
        else:
            break
    a = 0.0
    for _ in range(max_iter):
        mid = (a + b) / 2
        if f(mid) > eps:
            a = mid
        else:
            b = mid
    return mid


def get_xis_se(
    kernel: Kernel,
    eps: float,
    L: float,
) -> Tuple[Array, float, int]:
    """1-D equispaced Fourier nodes for SE kernel (heuristic)."""
    l = kernel.lengthscale
    var = kernel.variance
    dim = kernel.dim
    eps_use = eps / var

    h = 1 / (L + l * math.sqrt(2 * math.log(4 * dim * 3 ** dim / eps_use)))
    hm = math.ceil(
        math.sqrt(math.log(dim * (4 ** (dim + 1)) / eps_use) / 2)
        / math.pi
        / l
        / h
    )
    xis = jnp.arange(-hm, hm + 1) * h
    mtot = 2 * hm + 1
    return xis, h, mtot


def get_xis_matern(
    kernel: Kernel,
    eps: float,
    L: float,
) -> Tuple[Array, float, int]:
    """1-D equispaced Fourier nodes for Matérn kernel (heuristic)."""
    l = kernel.lengthscale
    var = kernel.variance
    nu = kernel.nu
    dim = kernel.dim
    eps_use = eps / var

    h = 1 / (L + 0.85 * l / math.sqrt(nu) * math.log(1 / eps_use))
    hm = math.ceil(
        (math.pi ** (nu + dim / 2) * l ** (2 * nu) * eps_use / 0.15) ** (-1 / (2 * nu + dim / 2))
        / h
    )
    xis = jnp.arange(-hm, hm + 1) * h
    mtot = 2 * hm + 1
    return xis, h, mtot


def get_xis_integral(
    kernel: Kernel,
    eps: float,
    L: float,
) -> Tuple[Array, float, int]:
    """1-D equispaced Fourier nodes using integral (bisection) method."""
    dim = kernel.dim

    # Spatial truncation bound
    Ltime = find_truncation_bound(lambda r: float(kernel(jnp.array(r))), eps)
    h = 1 / (L + Ltime)

    # Frequency truncation bound
    s0 = float(jnp.squeeze(kernel.spectral_density(jnp.array(0.0))))

    def khat_modified(r):
        val = abs(r) ** (dim - 1) * float(jnp.squeeze(kernel.spectral_density(jnp.array(r)))) / s0
        return val

    Lfreq = find_truncation_bound(khat_modified, eps)
    hm = math.ceil(Lfreq / h)

    xis = jnp.arange(-hm, hm + 1) * h
    mtot = xis.size
    return xis, h, mtot


def get_xis(
    kernel: Kernel,
    eps: float,
    L: float,
    use_integral: bool = False,
) -> Tuple[Array, float, int]:
    """Return 1-D equispaced Fourier quadrature nodes.

    Parameters
    ----------
    kernel : Kernel
    eps : float
        Tolerance parameter.
    L : float
        Max size of spatial domain so all inter-point differences lie in [-L, L].
    use_integral : bool
        If True, use bisection-based integral method.

    Returns
    -------
    xis : Array, shape (2*hm+1,)
    h : float
    mtot : int
    """
    if use_integral:
        return get_xis_integral(kernel, eps, L)

    if kernel.name == "se":
        return get_xis_se(kernel, eps, L)
    elif kernel.name == "matern":
        return get_xis_matern(kernel, eps, L)
    else:
        raise ValueError(f"Unknown kernel name: {kernel.name}")


def get_xis_nd(
    kernel: Kernel,
    eps: float,
    L: float,
    use_integral: bool = False,
) -> Tuple[Array, float, int]:
    """d-dimensional equispaced Fourier nodes (tensor product of 1-D nodes).

    Returns
    -------
    xis : Array, shape (mtot^d, d)
    h : float
    mtot : int  (per-dimension count)
    """
    dim = kernel.dim
    xis_1d, h, mtot = get_xis(kernel, eps, L, use_integral)

    if dim == 1:
        return xis_1d.reshape(-1, 1), h, mtot

    grids = jnp.meshgrid(*[xis_1d for _ in range(dim)], indexing="ij")
    xis = jnp.stack([g.ravel() for g in grids], axis=-1)
    return xis, h, mtot
