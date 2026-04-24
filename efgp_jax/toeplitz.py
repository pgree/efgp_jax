"""Toeplitz-ND FFT-based matrix-vector product and convolution vector."""

import math
from math import prod
from typing import List

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class ToeplitzND:
    """Cached FFT of the Toeplitz kernel for fast mat-vec products.

    Attributes:
        v_fft: FFT of zero-padded kernel, shape fft_shape.
        ns: Block sizes [n1, ..., nd].
        fft_shape: Padded FFT grid dimensions.
        starts: Start indices for central-block extraction.
        ends: End indices for central-block extraction.

    Registered as a JAX pytree: ``v_fft`` is the sole child (leaf), while
    the integer-list metadata is stored as static aux data so it is
    hashable and does not show up as leaves.
    """

    __slots__ = ("v_fft", "ns", "fft_shape", "starts", "ends")

    def __init__(self, v_fft: Array, ns, fft_shape, starts, ends):
        self.v_fft = v_fft
        self.ns = tuple(ns)
        self.fft_shape = tuple(fft_shape)
        self.starts = tuple(starts)
        self.ends = tuple(ends)

    # --- pytree protocol ---------------------------------------------------

    def tree_flatten(self):
        children = (self.v_fft,)
        aux = (self.ns, self.fft_shape, self.starts, self.ends)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        ns, fft_shape, starts, ends = aux
        (v_fft,) = children
        obj = cls.__new__(cls)
        obj.v_fft = v_fft
        obj.ns = ns
        obj.fft_shape = fft_shape
        obj.starts = starts
        obj.ends = ends
        return obj


def make_toeplitz(v: Array, force_pow2: bool = True) -> ToeplitzND:
    """Build a ToeplitzND operator from kernel vector v.

    Parameters
    ----------
    v : Array
        Toeplitz generating vector, shape (L1, ..., Ld) with Li = 2*ni - 1.
    force_pow2 : bool
        If True, pad each FFT dimension to the next power of 2.
    """
    if not jnp.iscomplexobj(v):
        from .nufft import _cmplx
        v = v.astype(_cmplx(v.dtype))

    Ls = list(v.shape)
    ns = [(L + 1) // 2 for L in Ls]
    d = len(Ls)

    if force_pow2:
        fft_shape = [1 << (L - 1).bit_length() for L in Ls]
    else:
        fft_shape = list(Ls)

    # Zero-pad v to fft_shape
    pad_widths = [(0, F - L) for L, F in zip(Ls, fft_shape)]
    v_pad = jnp.pad(v, pad_widths)

    # FFT of kernel
    fft_dims = tuple(range(v_pad.ndim))
    v_fft = jnp.fft.fftn(v_pad, axes=fft_dims)

    starts = [n - 1 for n in ns]
    ends = [st + n for st, n in zip(starts, ns)]

    return ToeplitzND(v_fft=v_fft, ns=ns, fft_shape=fft_shape, starts=starts, ends=ends)


def toeplitz_apply(toeplitz: ToeplitzND, x: Array) -> Array:
    """Apply the Toeplitz matrix-vector product via FFT convolution.

    x can be:
      - flat: shape (..., prod(ns))
      - block: shape (..., n1, ..., nd)
    """
    ns = toeplitz.ns
    d = len(ns)
    size = prod(ns)

    orig_flat = False
    if x.shape[-1] == size and (x.ndim == 1 or d > 1):
        orig_flat = True
        batch_shape = x.shape[:-1]
        x = x.reshape(*batch_shape, *ns)
    else:
        batch_shape = x.shape[:-d]

    if not jnp.iscomplexobj(x):
        x = x.astype(toeplitz.v_fft.dtype)

    # Zero-pad x
    pad_widths = [(0, 0)] * len(batch_shape) + [
        (0, F - n) for n, F in zip(ns, toeplitz.fft_shape)
    ]
    x_pad = jnp.pad(x, pad_widths)

    # FFT convolution
    fft_dims = tuple(range(len(batch_shape), len(batch_shape) + d))
    x_fft = jnp.fft.fftn(x_pad, axes=fft_dims)
    y_fft = x_fft * toeplitz.v_fft
    y = jnp.fft.ifftn(y_fft, axes=fft_dims)

    # Extract central block
    slices = [slice(None)] * len(batch_shape)
    for st, en in zip(toeplitz.starts, toeplitz.ends):
        slices.append(slice(st, en))
    y = y[tuple(slices)]

    if orig_flat:
        y = y.reshape(*batch_shape, size)

    return y
