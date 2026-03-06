"""NUFFT wrapper around jax-finufft."""

import math

import jax
import jax.numpy as jnp
from jax import Array
import jax_finufft


def _make_phi(x: Array, xcen: Array, h: float) -> Array:
    """Compute NUFFT phase coordinates phi = 2*pi*h*(x - xcen).

    Parameters
    ----------
    x : Array, shape (N, d)
    xcen : Array, shape (d,)
    h : float

    Returns
    -------
    phi : tuple of Array, each shape (N,) — one per dimension.
    """
    TWO_PI = 2 * math.pi
    # (N, d)
    coords = TWO_PI * h * (x - xcen)
    # jax-finufft expects one array per dimension
    return tuple(coords[:, i] for i in range(coords.shape[1]))


def _nufft1_single(phi: tuple, vals: Array, out_shape: tuple, eps: float) -> Array:
    """Single (unbatched) type-1 NUFFT call."""
    d = len(phi)
    if d == 1:
        return jax_finufft.nufft1(out_shape[0], vals, phi[0], eps=eps, iflag=-1)
    elif d == 2:
        return jax_finufft.nufft1(out_shape, vals, phi[0], phi[1], eps=eps, iflag=-1)
    elif d == 3:
        return jax_finufft.nufft1(out_shape, vals, phi[0], phi[1], phi[2], eps=eps, iflag=-1)


def _nufft2_single(phi: tuple, fk: Array, eps: float) -> Array:
    """Single (unbatched) type-2 NUFFT call."""
    d = len(phi)
    if d == 1:
        return jax_finufft.nufft2(fk, phi[0], eps=eps, iflag=1)
    elif d == 2:
        return jax_finufft.nufft2(fk, phi[0], phi[1], eps=eps, iflag=1)
    elif d == 3:
        return jax_finufft.nufft2(fk, phi[0], phi[1], phi[2], eps=eps, iflag=1)


def _cmplx(dtype):
    """Map a real or complex dtype to the corresponding complex dtype."""
    if dtype == jnp.float32 or dtype == jnp.complex64:
        return jnp.complex64
    return jnp.complex128


def nufft_type1(
    phi: tuple,
    vals: Array,
    out_shape: tuple,
    eps: float = 1e-6,
) -> Array:
    """Type 1 NUFFT: nonuniform -> uniform (adjoint).

    Parameters
    ----------
    phi : tuple of Array
        Phase coordinates, one (N,) array per dimension.
    vals : Array
        Values at nonuniform points, shape (N,) or (B, N).
    out_shape : tuple of int
        Output grid shape (m1, ..., md).
    eps : float
        NUFFT accuracy.

    Returns
    -------
    Array of shape out_shape or (B, *out_shape).
    """
    cdtype = _cmplx(vals.dtype)
    if not jnp.iscomplexobj(vals):
        vals = vals.astype(cdtype)

    if vals.ndim == 1:
        return _nufft1_single(phi, vals, out_shape, eps)
    else:
        return jax.vmap(lambda v: _nufft1_single(phi, v, out_shape, eps))(vals)


def nufft_type2(
    phi: tuple,
    fk: Array,
    eps: float = 1e-6,
) -> Array:
    """Type 2 NUFFT: uniform -> nonuniform (forward).

    Parameters
    ----------
    phi : tuple of Array
        Phase coordinates, one (N,) array per dimension.
    fk : Array
        Coefficients on uniform grid, shape (m1, ..., md) or (B, m1, ..., md).
    eps : float
        NUFFT accuracy.

    Returns
    -------
    Array of shape (N,) or (B, N).
    """
    d = len(phi)
    cdtype = _cmplx(fk.dtype)
    if not jnp.iscomplexobj(fk):
        fk = fk.astype(cdtype)

    is_batched = fk.ndim > d

    if not is_batched:
        return _nufft2_single(phi, fk, eps)
    else:
        return jax.vmap(lambda f: _nufft2_single(phi, f, eps))(fk)
