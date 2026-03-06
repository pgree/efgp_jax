"""Tests for quadrature (frequency grid generation)."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from efgp_jax.quadrature import find_truncation_bound, get_xis, get_xis_nd
from efgp_jax.kernels import SE, Matern


def test_find_truncation_bound():
    """Bisection should find where exp(-x) ~ eps."""
    f = lambda x: float(jnp.exp(-x))
    eps = 1e-6
    L = find_truncation_bound(f, eps)
    assert abs(f(L) - eps) < 1e-4


def test_get_xis_se():
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    xis, h, mtot = get_xis(kernel, eps=1e-4, L=1.0)
    assert mtot == xis.size
    assert h > 0
    assert mtot % 2 == 1  # 2*hm + 1


def test_get_xis_matern():
    kernel = Matern(lengthscale=0.3, variance=1.0, nu=2.5, dim=1)
    xis, h, mtot = get_xis(kernel, eps=1e-4, L=1.0)
    assert mtot == xis.size
    assert h > 0


def test_get_xis_nd_shape():
    kernel = SE(lengthscale=0.3, variance=1.0, dim=2)
    xis, h, mtot = get_xis_nd(kernel, eps=1e-3, L=1.0)
    assert xis.shape[1] == 2
    assert xis.shape[0] == mtot ** 2


def test_get_xis_integral():
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    xis, h, mtot = get_xis(kernel, eps=1e-4, L=1.0, use_integral=True)
    assert mtot == xis.size
    assert h > 0
