"""Fast evaluation of a Fourier basis on [0, 1] via NUFFT.

Given kernel hyperparameters, builds a 1-D Fourier basis (frequency grid
+ spectral weights) and evaluates it at arbitrary target points using
a type-2 NUFFT (uniform coefficients -> nonuniform points).

The basis functions are phi_j(x) = w_j * exp(2*pi*i*xi_j*x),  j = -M..M,
where xi_j = j*h are equispaced frequencies and w_j = sqrt(S(xi_j)*h).

Evaluating all M basis functions at N points costs O(M log M + N) via NUFFT,
compared to O(N*M) for the naive dense product.
"""

import math
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array

from .kernels import Kernel, SE, Matern
from .quadrature import get_xis
from .nufft import _make_phi, nufft_type2, _cmplx


class FourierBasis1D(NamedTuple):
    """1-D equispaced Fourier basis descriptor.

    Attributes
    ----------
    xis : Array, shape (M,)
        Frequency nodes xi_j = j * h.
    weights : Array, shape (M,)
        Spectral weights w_j = sqrt(S(xi_j) * h).
    h : float
        Frequency spacing.
    mtot : int
        Number of basis functions (= 2*hm + 1).
    """
    xis: Array
    weights: Array
    h: float
    mtot: int


def make_basis(
    kernel: Kernel,
    L: float = 1.0,
    eps: float = 1e-6,
    *,
    use_integral: bool = False,
) -> FourierBasis1D:
    """Build a 1-D Fourier basis from a kernel.

    Parameters
    ----------
    kernel : Kernel
        Kernel object (e.g. ``SE(...)`` or ``Matern(...)``).
    L : float
        Spatial domain half-width (points should lie in [0, L]).
    eps : float
        Truncation tolerance for the frequency grid.
    use_integral : bool
        Use integral (bisection) method for grid construction.

    Returns
    -------
    FourierBasis1D
    """
    xis, h, mtot = get_xis(kernel, eps, L, use_integral=use_integral)
    weights = jnp.sqrt(kernel.spectral_density(xis) * h)
    return FourierBasis1D(xis=xis, weights=weights, h=h, mtot=mtot)


def evaluate_basis_nufft(basis: FourierBasis1D, x: Array) -> Array:
    """Evaluate every basis function at target points x via NUFFT.

    Computes  f_j(x_i) = w_j * exp(2*pi*i * xi_j * x_i)  for all i, j
    using a type-2 NUFFT (one transform per "virtual coefficient vector").

    For a single coefficient vector c of shape (M,), the type-2 NUFFT
    computes  sum_j c_j exp(2*pi*i * xi_j * x_i).  To extract the full
    basis matrix Phi (shape N x M), we run M transforms with c = e_j * w_j.
    That is expensive (M transforms), so instead we return the result of
    applying the basis to a set of coefficient vectors.

    This function returns the full N x M basis matrix via a batched NUFFT:
    each column j is obtained by transforming the unit vector e_j scaled
    by w_j.  When M is moderate this is practical and much faster than
    the dense O(NM) matmul for large N.

    Parameters
    ----------
    basis : FourierBasis1D
    x : Array, shape (N,)
        Target points (e.g. on [0, 1]).

    Returns
    -------
    Phi : Array, shape (N, M), complex
        Phi[i, j] = w_j * exp(2*pi*i * xi_j * x_i).
    """
    x = jnp.atleast_1d(x)
    N = x.shape[0]
    M = basis.mtot
    cdtype = _cmplx(x.dtype)

    # NUFFT phase coordinates: phi = 2*pi*h*x  (centered at 0)
    phi = (2 * math.pi * basis.h * x,)

    # Batched type-2 NUFFT: transform each column of diag(w) = w_j * e_j
    # fk has shape (M, M) = diag(weights)
    fk = jnp.diag(basis.weights.astype(cdtype))  # (M, M)
    # nufft_type2 with batched fk of shape (M, M) treats first dim as batch
    Phi_T = nufft_type2(phi, fk, eps=1e-12)  # (M, N)
    return Phi_T.T  # (N, M)


def apply_basis_nufft(
    basis: FourierBasis1D,
    x: Array,
    coeffs: Array,
    *,
    nufft_eps: float = 1e-12,
) -> Array:
    """Evaluate  f(x) = sum_j coeffs_j * w_j * exp(2*pi*i * xi_j * x)  via NUFFT.

    This is the efficient operation: a single type-2 NUFFT that computes
    the weighted sum at all N target points in O(M log M + N) time.

    Parameters
    ----------
    basis : FourierBasis1D
    x : Array, shape (N,)
        Target points.
    coeffs : Array, shape (M,) or (B, M)
        Coefficient vector(s).  If batched, returns (B, N).
    nufft_eps : float
        NUFFT accuracy.

    Returns
    -------
    values : Array, shape (N,) or (B, N), complex
    """
    x = jnp.atleast_1d(x)
    cdtype = _cmplx(x.dtype)
    phi = (2 * math.pi * basis.h * x,)

    # Scale coefficients by weights
    wc = (basis.weights.astype(cdtype) * coeffs.astype(cdtype))
    return nufft_type2(phi, wc, eps=nufft_eps)


def sample_gp_nufft(
    basis: FourierBasis1D,
    x: Array,
    key: Array,
    n_samples: int = 1,
    *,
    nufft_eps: float = 1e-12,
) -> Array:
    """Draw samples from the GP prior via NUFFT-accelerated basis expansion.

    Uses O(M log M + N) per sample instead of O(N*M) for the dense method.

    Parameters
    ----------
    basis : FourierBasis1D
        Fourier basis (from ``make_basis``).
    x : Array, shape (N,)
        Evaluation points.
    key : Array
        JAX PRNG key.
    n_samples : int
        Number of independent samples.
    nufft_eps : float
        NUFFT accuracy.

    Returns
    -------
    Array, shape (n_samples, N) or (N,) if n_samples == 1.
    """
    M = basis.mtot
    cdtype = _cmplx(x.dtype)

    # Complex normal z = a + ib, a,b ~ N(0,1).
    # E[|z|^2] = 2, E[z^2] = 0, so Re(Phi @ z) has covariance K.
    key1, key2 = jax.random.split(key)
    z_real = jax.random.normal(key1, shape=(n_samples, M))
    z_imag = jax.random.normal(key2, shape=(n_samples, M))
    z = (z_real + 1j * z_imag).astype(cdtype)

    values = apply_basis_nufft(basis, x, z, nufft_eps=nufft_eps)  # (n_samples, N)
    samples = jnp.real(values)

    if n_samples == 1:
        return samples[0]
    return samples


def evaluate_basis_dense(basis: FourierBasis1D, x: Array) -> Array:
    """Evaluate basis via direct dense computation (for reference/testing).

    Parameters
    ----------
    basis : FourierBasis1D
    x : Array, shape (N,)

    Returns
    -------
    Phi : Array, shape (N, M), complex
    """
    x = jnp.atleast_1d(x)
    phases = 2 * jnp.pi * jnp.outer(x, basis.xis)  # (N, M)
    return basis.weights[None, :] * jnp.exp(1j * phases)
