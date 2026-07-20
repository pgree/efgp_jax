"""Tests for fixed-frequency (amortized) hyperparameter optimization."""

import math

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from efgp_jax.kernels import SE, Matern, log_marginal
from efgp_jax.fixed_freq import (
    build_fixed_grid,
    _precompute,
    _make_nll,
    _kernel_factory,
    optimize_hyperparameters_fixed_freq,
)
from efgp_jax.nufft import _cmplx


def _dense_weightspace_nll(x, y, grid, kernel, sigmasq):
    """Brute-force NLL of Sigma = Phi Phi^H + sigma^2 I on the fixed grid.

    Reference that shares the *same* frequencies as the algorithm, so it tests
    the Woodbury/Cholesky algebra exactly (not the spectral approximation).
    """
    xis = np.asarray(grid["xis"])          # (M, d)
    xcen = np.asarray(grid["xcen"])
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(y)
    N = x.shape[0]
    hd = grid["h"] ** grid["d"]
    w = np.sqrt(np.asarray(kernel.spectral_density(grid["xis"])) * hd)  # (M,)
    E = np.exp(2j * np.pi * (x - xcen) @ xis.T)   # (N, M)
    Phi = E * w[None, :]
    Sigma = Phi @ Phi.conj().T + sigmasq * np.eye(N)
    Sigma = 0.5 * (Sigma + Sigma.conj().T)
    Lc = np.linalg.cholesky(Sigma)
    alpha = np.linalg.solve(Sigma, y)
    quad = float(np.real(y @ alpha))
    logdet = 2.0 * float(np.sum(np.log(np.real(np.diag(Lc)))))
    return 0.5 * (quad + logdet + N * math.log(2 * math.pi))


def _make_vg(x, y, kernel, grid):
    cdtype = _cmplx(x[:, None].dtype)
    G_dense, b, yTy, N = _precompute(x[:, None], y, grid, 6e-8, cdtype)
    return _make_nll(grid, G_dense, b, yTy, N, _kernel_factory(kernel), cdtype)


@pytest.mark.parametrize("kernel", [
    SE(lengthscale=0.15, variance=1.3, dim=1),
    Matern(lengthscale=0.2, variance=0.8, dim=1, nu=1.5),
])
def test_nll_matches_dense_weightspace(kernel):
    """Weight-space NLL matches the brute-force dense NLL on the same grid."""
    rng = np.random.default_rng(0)
    x = jnp.asarray(np.sort(rng.uniform(0, 1, size=40)))
    y = jnp.asarray(rng.standard_normal(40))
    sigmasq = 0.1
    grid = build_fixed_grid(kernel, (0.0, 1.0), 1e-6,
                               l_bounds=(0.1, 0.4))

    vg = _make_vg(x, y, kernel, grid)
    log_theta = jnp.array([math.log(float(kernel.lengthscale)),
                           math.log(float(kernel.variance)),
                           math.log(sigmasq)])
    nll_ff = float(vg(log_theta)[0])
    nll_dense = _dense_weightspace_nll(x, y, grid, kernel, sigmasq)

    assert nll_ff == pytest.approx(nll_dense, rel=1e-6)


def test_nll_matches_true_kernel_se():
    """For SE (fast spectral convergence) the NLL matches the exact kernel."""
    rng = np.random.default_rng(3)
    x = jnp.asarray(np.sort(rng.uniform(0, 1, size=50)))
    y = jnp.asarray(rng.standard_normal(50))
    kernel = SE(lengthscale=0.15, variance=1.0, dim=1)
    sigmasq = 0.1
    grid = build_fixed_grid(kernel, (0.0, 1.0), 1e-9,
                               l_bounds=(0.1, 0.3))

    vg = _make_vg(x, y, kernel, grid)
    log_theta = jnp.array([math.log(0.15), math.log(1.0), math.log(sigmasq)])
    nll_ff = float(vg(log_theta)[0])
    nll_dense = -float(log_marginal(x[:, None], y, sigmasq, kernel))

    assert nll_ff == pytest.approx(nll_dense, rel=1e-4)


def test_gradient_matches_finite_difference():
    """Autodiff gradient of the NLL matches finite differences."""
    rng = np.random.default_rng(1)
    x = jnp.asarray(np.sort(rng.uniform(0, 1, size=50)))
    y = jnp.asarray(rng.standard_normal(50))
    kernel = SE(lengthscale=0.2, variance=1.0, dim=1)
    grid = build_fixed_grid(kernel, (0.0, 1.0), 1e-6,
                               l_bounds=(0.1, 0.4))
    vg = _make_vg(x, y, kernel, grid)

    theta = jnp.array([math.log(0.2), math.log(1.0), math.log(0.05)])
    grad = np.asarray(vg(theta)[1])

    h = 1e-6
    fd = np.array([
        (float(vg(theta.at[i].add(h))[0]) - float(vg(theta.at[i].add(-h))[0]))
        / (2 * h)
        for i in range(3)
    ])
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-5)


def test_recovers_hyperparameters():
    """Optimizer converges and lands near the data-generating scale."""
    rng = np.random.default_rng(2)
    n = 400
    x = jnp.asarray(np.sort(rng.uniform(0, 1, size=n)))

    true_kernel = SE(lengthscale=0.08, variance=1.0, dim=1)
    true_sig2 = 0.04
    K = np.asarray(
        true_kernel(np.abs(np.asarray(x)[:, None] - np.asarray(x)[None, :]))
    )
    Lc = np.linalg.cholesky(K + 1e-8 * np.eye(n))
    f = Lc @ rng.standard_normal(n)
    y = jnp.asarray(f + math.sqrt(true_sig2) * rng.standard_normal(n))

    kernel0 = SE(lengthscale=0.3, variance=0.5, dim=1)
    kernel_fit, sig2_fit, info = optimize_hyperparameters_fixed_freq(
        x, y, kernel0, sigmasq0=0.1, eps=1e-6, domain=(0.0, 1.0),
        l_bounds=(0.03, 3.0), maxiter=60, verbose=False,
    )

    assert info["success"]
    assert 0.04 < float(kernel_fit.lengthscale) < 0.16
    assert sig2_fit < 0.15
